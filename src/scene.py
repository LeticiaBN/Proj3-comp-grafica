"""Monta a cena: posiciona todas as entidades em escala/local coerente.

Layout (vista de cima, X→leste, Z→sul, Y→cima):

  z=-100 ────────────────────────────────── z=+100
   .  mtn(-62,-52)                mtn(58,-65) .
   .                                          .
   .   sphere  connector  SCIENCEBASE         .
   .   (-37)   (-18.7)    (0,0,0)             .
   .              (interior: cama 5 partes,   .
   .               matress, desk, robot,      .
   .               babyyoda, storage_box,     .
   .               neon, bottle, trash)       .
   .                                          .
   .   supply_box(-22,5)  cart_ext(20,10)     .
   .   satelite(-15,25)   rover(15,15) — móv  .
   .                                          .
   .  mtn(-48,62)                mtn(68,42)   .
   .                                          .
   .   nave (orbita raio 55, altura 35)       .
   .   planeta (120, 70, -150) — ao fundo     .
   .   skybox (cubemap nebulosa/espaço)       .

Tudo apoiado em piso de Marte com relevo senoidal (200×200).
Piso interno: disco procedural de azulejos sci-fi (raio 10.4).
4 montanhas cônicas nos quadrantes: (-62,-52) (58,-65) (68,42) (-48,62).
"""
import math
from pathlib import Path
from typing import List

import numpy as np
from OpenGL.GL import GL_BACK, GL_CULL_FACE, glCullFace, glDisable, glEnable

from src.entity import Entity
from src.floor import MarsFloorWithCircularHole, TiledFloorDisk, MarsMountain, _height
from src.mesh import Mesh
from src.skybox import Skybox

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "objetos"
SKY_DIR = ROOT / "objetos" / "skybox"

# Tamanho do mundo (chão e câmera limits)
WORLD_HALF = 100.0
SKY_HEIGHT = 80.0

# Footprint da sciencebase (cúpula circular). A base modelada tem ~20.9
# de diâmetro depois do scale=10, o que dá raio externo ~10.45. Usamos
# 11.0 como raio único para:
#   - o buraco circular no chão de Marte (precisa ser >= raio externo
#     pra não sobrar Marte por baixo da base);
#   - o disco do piso interno (precisa cobrir a parede para esconder a
#     transição);
# A pequena folga (0.55) é coberta pela espessura da própria parede
# circular da base.
BASE_FOOTPRINT_RADIUS = 10.4


class Scene:
    def __init__(self):
        # ---- Skybox ----
        # ordem GL: +X (right), -X (left), +Y (top), -Y (bottom), +Z (front), -Z (back)
        self.skybox = Skybox([
            str(SKY_DIR / "bkg1_right.png"),
            str(SKY_DIR / "bkg1_left.png"),
            str(SKY_DIR / "bkg1_top.png"),
            str(SKY_DIR / "bkg1_bot.png"),
            str(SKY_DIR / "bkg1_front.png"),
            str(SKY_DIR / "bkg1_back.png"),
        ])

        # ---- Chão externo (Marte) com buraco circular sob a base ----
        # A sciencebase é uma cúpula CIRCULAR — um buraco quadrado deixava
        # os 4 cantos descobertos (sem chão de Marte e fora do raio da
        # parede da base). Aqui o buraco é circular, casado com a forma
        # da cúpula. Resultado: a borda do buraco coincide com a parede
        # externa da base, sem vazio visível.
        self.mars = MarsFloorWithCircularHole(
            str(ASSETS / "mars" / "mars_colormap.png"),
            world_half=WORLD_HALF,
            hole_radius=BASE_FOOTPRINT_RADIUS,
            segments=64,
        )

        # ---- Chão interno (disco circular, dentro da base) ----
        # Mesmo raio do buraco em Marte: assim o disco interno preenche
        # exatamente o vazio aberto. Textura procedural de azulejos sci-fi
        # (gerada em runtime, tileable) para dar impressão de chão real
        # sem depender de um asset externo. Y=0.3 cria uma "plataforma"
        # leve sobre o nível de Marte (mesma altura usada por bed/desk/
        # robot/storage/babyyoda no interior).
        # world_tiles_across_diameter=8 → cada azulejo tem ~22/8 ≈ 2.75
        # unidades (escala "metro"), tamanho realista de piso industrial.
        self.indoor_floor = TiledFloorDisk(
            radius=BASE_FOOTPRINT_RADIUS,
            segments=64,
            world_tiles_across_diameter=8.0,
            base_color=(0.62, 0.63, 0.66),
            grout_color=(0.18, 0.18, 0.20),
        )
        self.indoor_floor.position = np.array([0.0, 0.3, 0.0], dtype=np.float32)

        # ---- Base modular (delimitador interno) ----
        self.sciencebase = Entity(
            Mesh.from_obj(str(ASSETS / "modularbase" / "sciencebase.obj"),
                          center_xz=True, floor_y=True),
            position=(0, 0, 0),
            scale=(10, 10, 10),
            disable_culling=True,  # vê paredes por dentro também
        )
        self.connector = Entity(
            Mesh.from_obj(str(ASSETS / "modularbase" / "connector.obj"),
                          center_xz=True, floor_y=True),
            position=(-18.7, 0, 0),
            scale=(10, 10, 10),
        )
        self.spherebase = Entity(
            Mesh.from_obj(str(ASSETS / "modularbase" / "spherebase.obj"),
                          center_xz=True, floor_y=True),
            position=(-37.4, 0, 0),
            scale=(10, 10, 10),
        )

        # ---- Externos ----
        # Nave (animada — posição definida no update())
        self.nave = Entity(
            Mesh.from_obj(str(ASSETS / "nave" / "attackship.obj"), center_xz=True),
            position=(0, 25, 50),
            rotation=(0, 0, 0),
            scale=(1, 1, 1),
        )

        # Rover — translação por teclado
        self.rover = Entity(
            Mesh.from_obj(str(ASSETS / "rover" / "rover.obj"), floor_y=True),
            position=(15, 0, 15),
            scale=(1.5, 1.5, 1.5),
        )

        # Satélite — rotação por teclado (R/T giram eixo Y)
        # Rotação inicial em X e Z deixa o prato inclinado como uma antena
        # real buscando sinal; o giro em Y varre o céu em diferentes direções.
        self.satelite = Entity(
            Mesh.from_obj(str(ASSETS / "satelite" / "satelite.obj"), floor_y=True),
            position=(-15, 0, 25),
            rotation=(math.radians(30), 0, math.radians(-15)),
            scale=(1.5, 1.5, 1.5),
        )

        # Planeta de fundo — escala por teclado
        self.planet = Entity(
            Mesh.from_obj(str(ASSETS / "planet" / "planet.obj")),
            position=(120, 70, -150),
            rotation=(0, 0, 0),
            scale=(30, 30, 30),
            disable_culling=True,  # Mostra a esfera completa (frente e costas)
        )

        # Carrinho de carga perto da área do rover
        self.cart_ext = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "cart.obj"), floor_y=True),
            position=(20, 0, 10), scale=(1.8, 1.8, 1.8), rotation=(0, math.radians(-30), 0),
        )
        # Caixas de suprimentos perto do conector
        self.supply_box = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "box_01.obj"), floor_y=True),
            position=(-22, 0, 5), scale=(2.0, 2.0, 2.0),
        )

        # Cama composta: partes modeladas juntas mas em arquivos separados.
        # Movida mais para o centro para não atravessar as paredes da base.
        bed_pos = (-2.5, 0.3, -3.5)
        bed_scale = (1.8, 1.8, 1.8)
        bed_rot = (0, math.radians(225), 0)
        bed_offset = (0.13, 0.0, -0.42)  # Centro global das partes da cama
        
        # O offset do colchão subtrai ~0.65 do Y, o que levanta os vértices localmente
        # O offset do colchão foi recalculado para que, após a rotação
        # de 90 graus, seu centro (X e Z) se alinhe perfeitamente à cama.
        # Diminuir o Z (-0.555 em vez de -0.255) empurra o colchão para a direita.
        matress_offset = (0.15, -0.65, -0.055)
        
        # O colchão foi modelado virado de lado (90 graus em relação à estrutura).
        # Como o offset já centraliza tudo, girar ele localmente o mantém no lugar certo.
        matress_rot = (bed_rot[0], bed_rot[1] + math.radians(90), bed_rot[2])
        
        self.bed       = Entity(Mesh.from_obj(str(ASSETS / "scifi" / "bed.obj"), offset=bed_offset),
                                position=bed_pos, scale=bed_scale, rotation=bed_rot)
        self.matress   = Entity(Mesh.from_obj(str(ASSETS / "scifi" / "matress.obj"), offset=matress_offset),
                                position=bed_pos, scale=bed_scale, rotation=matress_rot)
        self.bed_back  = Entity(Mesh.from_obj(str(ASSETS / "scifi" / "bed_back.obj"), offset=bed_offset),
                                position=bed_pos, scale=bed_scale, rotation=bed_rot)
        self.bed_front = Entity(Mesh.from_obj(str(ASSETS / "scifi" / "bed_front.obj"), offset=bed_offset),
                                position=bed_pos, scale=bed_scale, rotation=bed_rot)
        self.pipes_bed = Entity(Mesh.from_obj(str(ASSETS / "scifi" / "pipes_bed.obj"), offset=bed_offset),
                                position=bed_pos, scale=bed_scale, rotation=bed_rot)

        # Mesa de trabalho — maior e mais perto do centro
        self.desk = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "desk_01.obj"), floor_y=True),
            position=(4.5, 0.3, -7.0), rotation=(0, math.radians(-120), 0),
            scale=(1.5, 1.5, 1.5),
        )

        # Robô — maior, posicionado no centro da base virado para a entrada
        self.robot = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "robot.obj"), floor_y=True),
            position=(3.0, 0.3, -9.0), rotation=(0, math.radians(240), 0),
            scale=(2.0, 2.0, 2.0),
        )

        # Caixa de armazenamento
        self.storage_box = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "storage_box.obj"), floor_y=True),
            position=(-6, 0.3, 3),
            scale=(1.8, 1.8, 1.8),
        )

        # Letreiro neon (painel) - Na parede esquerda
        self.neon = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "neon.obj")),
            position=(0, 7.7, 0), rotation=(0, math.radians(90), 0),
            scale=(1.5, 1.5, 1.5),
        )

        # Baby Yoda — morador da estação, agora em escala visível
        self.babyyoda = Entity(
            Mesh.from_obj(str(ASSETS / "babyyoda" / "babyyoda.obj"), floor_y=True),
            position=(2, 0.3, 1.0), rotation=(0, math.radians(180), 0),
            scale=(5.0, 5.0, 5.0),
        )

        # Novos objetos internos — assets já presentes em scifi/
        # Garrafa — do lado direito da cama, no chão
        self.bottle = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "bottle.obj"), floor_y=True),
            position=(-5.5, 0.3, -2.5),
            scale=(1.5, 1.5, 1.5),
        )
        # Lixeira — também mais à direita, longe da cama
        self.trash = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "trash.obj"), floor_y=True),
            position=(-2.0, 0.3, -7.0), rotation=(0, math.radians(-45), 0),
            scale=(1.5, 1.5, 1.5),
        )

        # Listas para iteração no draw
        self.outdoor_entities: List[Entity] = [
            self.connector, self.spherebase,
            self.nave, self.rover, self.satelite, self.planet,
            self.cart_ext, self.supply_box,
        ]
        self.indoor_entities: List[Entity] = [
            self.bed, self.matress, self.bed_back, self.bed_front, self.pipes_bed,
            self.desk, self.robot, self.storage_box, self.neon, self.babyyoda,
            self.bottle, self.trash,
        ]

        # ---- Montanhas (externo) ----
        # Posicionadas nos 4 quadrantes afastados da base, com Y base igual
        # ao deslocamento do terreno naquele ponto para assentarem no chão.
        mars_tex = str(ASSETS / "mars" / "mars_colormap.png")
        _mtn_cfg = [
            #  x,   z,  raio, altura
            (-62, -52,    22,     18),
            ( 58, -65,    18,     14),
            ( 68,  42,    25,     20),
            (-48,  62,    16,     12),
        ]
        self.mountains = []
        for mx, mz, mr, mh in _mtn_cfg:
            m = MarsMountain(mars_tex, radius=mr, height=mh)
            by = _height(float(mx), float(mz), BASE_FOOTPRINT_RADIUS)
            m.position = np.array([float(mx), by, float(mz)], dtype=np.float32)
            self.mountains.append(m)

        # Animação da nave
        # parametros da orbita: raio, altura, velocidade e angulo atual
        self._nave_orbit_radius = 55.0
        self._nave_orbit_height = 35.0
        self._nave_speed = 0.25  # rad/s
        self._nave_angle = 0.0

    def update(self, dt: float):
        """Anima a nave em órbita circular."""
        # avanca o angulo proporcional ao tempo passado
        self._nave_angle += dt * self._nave_speed
        # converte angulo em (x,z) usando seno/cosseno
        x = math.cos(self._nave_angle) * self._nave_orbit_radius
        z = math.sin(self._nave_angle) * self._nave_orbit_radius
        self.nave.position = np.array([x, self._nave_orbit_height, z], dtype=np.float32)
        # nariz da nave aponta tangente à órbita
        self.nave.rotation = np.array(
            [0.0, -self._nave_angle - math.pi / 2.0, 0.0], dtype=np.float32,
        )

    def draw(self, shader, wireframe: bool = False):
        """Renderiza tudo. Skybox é desenhado fora desta função (precisa shader próprio)."""
        # piso externo
        self.mars.draw(shader, wireframe=wireframe)
        # piso interno (não é Entity, mas tem método draw compatível)
        self.indoor_floor.draw(shader, wireframe=wireframe)
        # montanhas
        # cada montanha desenha sua geometria propria
        for mtn in self.mountains:
            mtn.draw(shader, wireframe=wireframe)

        # Sciencebase (com cull off pra ver paredes por dentro)
        # desliga culling para podermos ver as paredes tanto por fora quanto por dentro
        glDisable(GL_CULL_FACE)
        self.sciencebase.draw(shader, wireframe=wireframe)
        # restaura o culling para as proximas entidades (otimizacao)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # demais entidades
        # tudo que esta fora da base (rover, nave, planeta, etc.)
        for e in self.outdoor_entities:
            e.draw(shader, wireframe=wireframe)
        # tudo que esta dentro da base (cama, mesa, robo, etc.)
        for e in self.indoor_entities:
            e.draw(shader, wireframe=wireframe)
