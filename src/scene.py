"""Monta a cena: posiciona todas as entidades em escala/local coerente
e tambem expoe as fontes de luz (Projeto 3).

Modelo de iluminacao Phong por fragmento, com luzes pontuais:

  Luz 0 (externa)  — FAROL DO ROVER. Translacao por teclado; afeta
                     APENAS objetos externos (req. 1).
  Luz 1 (interna)  — LAMPADA DE TETO da base. Luz branca-quente; afeta
                     APENAS objetos internos (req. 2).
  Luz 2 (interna)  — SABRE DE LUZ em cima da mesa. Cor verde, contrasta
                     com a lampada de teto (req. 2 — duas fontes
                     internas de CORES DIFERENTES).

Cada Entity carrega seus PROPRIOS parametros de iluminacao difusa e
especular (req. 7 — sem reutilizar valores do .mtl), e cada uma
declara seu "escopo" (outdoor / indoor / shared) — esse escopo vira
mascara de bits no shader, garantindo que cada luz so ilumine os
objetos do seu ambiente.

Layout (vista de cima, X→leste, Z→sul, Y→cima):

  z=-100 ────────────────────────────────── z=+100
   .  mtn(-62,-52)                mtn(58,-65) .
   .                                          .
   .   sphere  connector  SCIENCEBASE         .
   .   (-37)   (-18.7)    (0,0,0)             .
   .              (interior: cama, mesa,      .
   .               robo, lampadas internas)   .
   .                                          .
   .   supply_box(-22,5)  cart_ext(20,10)     .
   .   satelite(-15,25)   rover(15,15) — luz! .
   .                                          .
   .  mtn(-48,62)                mtn(68,42)   .
   .                                          .
   .   nave (orbita raio 55, altura 35)       .
   .   planeta (120, 70, -150) — ao fundo     .
   .   skybox (cubemap nebulosa/espaco)       .
"""
import math
from pathlib import Path
from typing import List

import numpy as np
from OpenGL.GL import GL_BACK, GL_CULL_FACE, glCullFace, glDisable, glEnable

from src.entity import Entity
from src.floor import MarsFloorWithCircularHole, TiledFloorDisk, MarsMountain, _height
from src.mesh import Mesh
from src.procedural import make_light_bulb_mesh, make_cylinder_lamp_mesh
from src.skybox import Skybox

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "objetos"
SKY_DIR = ROOT / "objetos" / "skybox"

# Tamanho do mundo (chao e camera limits)
WORLD_HALF = 100.0
SKY_HEIGHT = 80.0

# Footprint da sciencebase (cupula circular).
BASE_FOOTPRINT_RADIUS = 10.4


class Light:
    """Fonte de luz pontual. Os parametros sao enviados ao fragment
    shader em main.py atraves de arrays de uniforms (vec3[3]/int[3]).
    """
    def __init__(self, position, color, on: bool = True):
        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        # cada luz tem seu proprio "interruptor" (req. 3)
        self.on = on


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

        # =================================================================
        #  LUZES (Projeto 3)
        # =================================================================
        # Luz 0 — FAROL DO ROVER (externa, movel).
        # Posicao calculada em update() a partir do transform do rover; o
        # ponto fica entre os dois farois dianteiros.
        self.light_rover = Light(
            position=(15.0, 2.5, 15.0),
            color=(1.0, 0.95, 0.75),  # branco-quente
        )
        # Luz 1 — LAMPADA DE TETO da base (interna). Posicionada no
        # mesmo lugar do letreiro neon, um pouquinho abaixo das lampadas
        # cilindricas (para iluminar a cena DEbaixo dela).
        self.light_ceiling = Light(
            position=(0.0, 7.5, 0.0),
            color=(1.0, 0.9, 0.7),
        )
        # Luz 2 — SABRE DE LUZ em cima da mesa (interna B). Cor verde,
        # contrastando com a luz quente da lampada de teto (req. 2 do PDF
        # — duas fontes internas de cores diferentes). A posicao exata e
        # calculada mais abaixo, depois que a Entity do sabre define onde
        # fica o centro da lamina em coordenadas de mundo.
        self.light_saber = Light(
            position=(0.0, 0.0, 0.0),  # ajustada abaixo
            color=(0.4, 1.0, 0.5),
        )
        # Lista de luzes ativas (indice 0 = rover, 1 = teto, 2 = sabre).
        self.lights: List[Light] = [
            self.light_rover, self.light_ceiling, self.light_saber,
        ]

        # ---- Chao externo (Marte) com buraco circular sob a base ----
        self.mars = MarsFloorWithCircularHole(
            str(ASSETS / "mars" / "mars_colormap.png"),
            world_half=WORLD_HALF,
            hole_radius=BASE_FOOTPRINT_RADIUS,
            segments=64,
        )
        # Material proprio: Marte e bem fosco (pouco especular).
        self.mars.color = (1.0, 0.95, 0.92)
        self.mars.ks = (0.05, 0.05, 0.05)
        self.mars.shininess = 4.0
        # Marte e externo: so a luz do rover (alem da ambiente) o afeta.
        from src.entity import _scope_to_mask, _scope_to_masks
        self.mars.light_mask, self.mars.light_mask_back = _scope_to_masks("outdoor")

        # ---- Chao interno (disco circular, dentro da base) ----
        self.indoor_floor = TiledFloorDisk(
            radius=BASE_FOOTPRINT_RADIUS,
            segments=64,
            world_tiles_across_diameter=8.0,
            base_color=(0.62, 0.63, 0.66),
            grout_color=(0.18, 0.18, 0.20),
        )
        self.indoor_floor.position = np.array([0.0, 0.3, 0.0], dtype=np.float32)
        # Material: piso de azulejo metalico, brilho moderado.
        self.indoor_floor.ks = (0.4, 0.4, 0.45)
        self.indoor_floor.shininess = 24.0
        self.indoor_floor.light_mask, self.indoor_floor.light_mask_back = _scope_to_masks("indoor")

        # =================================================================
        #  BASE MODULAR
        # =================================================================
        # A sciencebase e vista de fora E de dentro (paredes sem culling),
        # logo declaramos como "shared": todas as luzes a afetam, fazendo
        # sentido sob qualquer ponto de vista.
        self.sciencebase = Entity(
            Mesh.from_obj(str(ASSETS / "modularbase" / "sciencebase.obj"),
                          center_xz=True, floor_y=True),
            position=(0, 0, 0),
            scale=(10, 10, 10),
            disable_culling=True,
            kd=(0.85, 0.85, 0.9),
            ks=(0.4, 0.4, 0.45),
            shininess=48.0,
            scope="shared",
        )
        # Conector e esfera externa: ambos sao vistos de fora.
        self.connector = Entity(
            Mesh.from_obj(str(ASSETS / "modularbase" / "connector.obj"),
                          center_xz=True, floor_y=True),
            position=(-18.7, 0, 0), scale=(10, 10, 10),
            kd=(0.82, 0.82, 0.88), ks=(0.35, 0.35, 0.40),
            shininess=40.0, scope="outdoor",
        )
        self.spherebase = Entity(
            Mesh.from_obj(str(ASSETS / "modularbase" / "spherebase.obj"),
                          center_xz=True, floor_y=True),
            position=(-37.4, 0, 0), scale=(10, 10, 10),
            kd=(0.80, 0.82, 0.90), ks=(0.45, 0.45, 0.50),
            shininess=64.0, scope="outdoor",
        )

        # =================================================================
        #  OBJETOS EXTERNOS
        # =================================================================
        # Nave (animada — posicao definida no update()).
        self.nave = Entity(
            Mesh.from_obj(str(ASSETS / "nave" / "attackship.obj"), center_xz=True),
            position=(0, 25, 50), rotation=(0, 0, 0), scale=(1, 1, 1),
            kd=(0.9, 0.9, 0.95), ks=(0.7, 0.7, 0.75),
            shininess=96.0, scope="outdoor",  # metalica/brilhante
        )

        # Rover — translacao por teclado; carrega o farol (luz 0).
        self.rover = Entity(
            Mesh.from_obj(str(ASSETS / "rover" / "rover.obj"), floor_y=True),
            position=(15, 0, 15), scale=(1.5, 1.5, 1.5),
            kd=(0.95, 0.85, 0.75), ks=(0.55, 0.55, 0.55),
            shininess=64.0, scope="outdoor",
        )

        # Satelite — antena metalica, brilho alto.
        self.satelite = Entity(
            Mesh.from_obj(str(ASSETS / "satelite" / "satelite.obj"), floor_y=True),
            position=(-15, 0, 25),
            rotation=(math.radians(30), 0, math.radians(-15)),
            scale=(1.5, 1.5, 1.5),
            kd=(0.85, 0.85, 0.90), ks=(0.65, 0.65, 0.70),
            shininess=80.0, scope="outdoor",
        )

        # Planeta de fundo — escala fixa (sem controle no projeto 3).
        self.planet = Entity(
            Mesh.from_obj(str(ASSETS / "planet" / "planet.obj")),
            position=(120, 70, -150), rotation=(0, 0, 0), scale=(30, 30, 30),
            disable_culling=True,
            kd=(1.0, 0.9, 0.75), ks=(0.10, 0.10, 0.10),
            shininess=8.0, scope="outdoor",  # gigante fosco
        )

        # Carrinho de carga perto da area do rover.
        self.cart_ext = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "cart.obj"), floor_y=True),
            position=(20, 0, 10), scale=(1.8, 1.8, 1.8),
            rotation=(0, math.radians(-30), 0),
            kd=(0.85, 0.85, 0.90), ks=(0.45, 0.45, 0.50),
            shininess=48.0, scope="outdoor",
        )
        # Caixas de suprimentos perto do conector.
        self.supply_box = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "box_01.obj"), floor_y=True),
            position=(-22, 0, 5), scale=(2.0, 2.0, 2.0),
            kd=(0.85, 0.82, 0.78), ks=(0.20, 0.20, 0.20),
            shininess=16.0, scope="outdoor",
        )

        # =================================================================
        #  OBJETOS INTERNOS (cama composta)
        # =================================================================
        bed_pos = (-2.5, 0.3, -3.5)
        bed_scale = (1.8, 1.8, 1.8)
        bed_rot = (0, math.radians(225), 0)
        bed_offset = (0.13, 0.0, -0.42)
        matress_offset = (0.15, -0.65, -0.055)
        matress_rot = (bed_rot[0], bed_rot[1] + math.radians(90), bed_rot[2])

        # Materiais comuns da cama: estrutura metalica + colchao fosco.
        bed_struct_kd = (0.78, 0.78, 0.85)
        bed_struct_ks = (0.35, 0.35, 0.40)
        bed_struct_sh = 32.0
        matress_kd = (0.92, 0.92, 0.95)
        matress_ks = (0.10, 0.10, 0.10)
        matress_sh = 8.0

        self.bed = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "bed.obj"), offset=bed_offset),
            position=bed_pos, scale=bed_scale, rotation=bed_rot,
            kd=bed_struct_kd, ks=bed_struct_ks, shininess=bed_struct_sh,
            scope="indoor",
        )
        self.matress = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "matress.obj"), offset=matress_offset),
            position=bed_pos, scale=bed_scale, rotation=matress_rot,
            kd=matress_kd, ks=matress_ks, shininess=matress_sh,
            scope="indoor",
        )
        self.bed_back = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "bed_back.obj"), offset=bed_offset),
            position=bed_pos, scale=bed_scale, rotation=bed_rot,
            kd=bed_struct_kd, ks=bed_struct_ks, shininess=bed_struct_sh,
            scope="indoor",
        )
        self.bed_front = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "bed_front.obj"), offset=bed_offset),
            position=bed_pos, scale=bed_scale, rotation=bed_rot,
            kd=bed_struct_kd, ks=bed_struct_ks, shininess=bed_struct_sh,
            scope="indoor",
        )
        self.pipes_bed = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "pipes_bed.obj"), offset=bed_offset),
            position=bed_pos, scale=bed_scale, rotation=bed_rot,
            kd=(0.65, 0.65, 0.70), ks=(0.55, 0.55, 0.60),
            shininess=80.0, scope="indoor",  # tubulacao metalica
        )

        # Mesa de trabalho — mesa metalica brilhante.
        self.desk = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "desk_01.obj"), floor_y=True),
            position=(4.5, 0.3, -7.0), rotation=(0, math.radians(-120), 0),
            scale=(1.5, 1.5, 1.5),
            kd=(0.80, 0.80, 0.85), ks=(0.5, 0.5, 0.55),
            shininess=64.0, scope="indoor",
        )

        # Robo assistente — chapas metalicas com brilho alto.
        self.robot = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "robot.obj"), floor_y=True),
            position=(3.0, 0.3, -9.0), rotation=(0, math.radians(240), 0),
            scale=(2.0, 2.0, 2.0),
            kd=(0.90, 0.90, 0.95), ks=(0.70, 0.70, 0.75),
            shininess=96.0, scope="indoor",
        )

        # Caixa de armazenamento.
        self.storage_box = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "storage_box.obj"), floor_y=True),
            position=(-6, 0.3, 3), scale=(1.8, 1.8, 1.8),
            kd=(0.85, 0.82, 0.78), ks=(0.20, 0.20, 0.20),
            shininess=16.0, scope="indoor",
        )

        # Letreiro neon — painel emissivo na parede esquerda.
        # Marcamos como emissive=True para que brilhe sempre (e um letreiro
        # "ligado"), independente das luzes da sala.
        self.neon = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "neon.obj")),
            position=(0, 7.7, 0), rotation=(0, math.radians(90), 0),
            scale=(1.5, 1.5, 1.5),
            kd=(1.0, 0.4, 0.6), ks=(0.0, 0.0, 0.0),
            shininess=1.0, scope="indoor", emissive=True,
        )

        # Baby Yoda — pele fosca.
        self.babyyoda = Entity(
            Mesh.from_obj(str(ASSETS / "babyyoda" / "babyyoda.obj"), floor_y=True),
            position=(2, 0.3, 1.0), rotation=(0, math.radians(180), 0),
            scale=(5.0, 5.0, 5.0),
            kd=(0.95, 0.95, 0.95), ks=(0.08, 0.08, 0.08),
            shininess=8.0, scope="indoor",
        )

        # Garrafa — vidro: alto brilho especular.
        self.bottle = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "bottle.obj"), floor_y=True),
            position=(-5.5, 0.3, -2.5), scale=(1.5, 1.5, 1.5),
            kd=(0.90, 0.95, 1.0), ks=(0.85, 0.85, 0.90),
            shininess=128.0, scope="indoor",
        )
        # Lixeira — metal fosco.
        self.trash = Entity(
            Mesh.from_obj(str(ASSETS / "scifi" / "trash.obj"), floor_y=True),
            position=(-2.0, 0.3, -7.0), rotation=(0, math.radians(-45), 0),
            scale=(1.5, 1.5, 1.5),
            kd=(0.80, 0.80, 0.85), ks=(0.30, 0.30, 0.35),
            shininess=24.0, scope="indoor",
        )

        # ---- SABRE DE LUZ dentro da storage_box (req. 2: 2a luz interna).
        #
        # ANATOMIA DO MODELO (em coordenadas LOCAIS do .obj):
        #   - O .obj contem APENAS O CABO (handle) do sabre — a lamina
        #     nao existe como geometria. Ela e gerada proceduralmente
        #     mais abaixo, como um cilindro emissivo verde.
        #   - O cabo vai de z_local = -0.41 (pommel) ate +0.19 (emissor).
        #   - A lamina nasce no EMISSOR (extremidade +Z) e estende-se em +Z.
    
        saber_scale_val   = 2.4     
        saber_lean_deg    = -60                                
        saber_pos_xz      = (-5.5, 2.8) 
        saber_floor_y     = 0.3    
        blade_local_len   = 1     # comprimento da LAMINA em unidades locais

        # ---- (geometria interna do modelo — nao mude se nao souber) ----
        HANDLE_POMMEL_Z   = -0.41
        HANDLE_EMITTER_Z  = 0.19

        saber_rot_x = math.radians(saber_lean_deg)
        cos_rx = math.cos(saber_rot_x)
        sin_rx = math.sin(saber_rot_x)

        # Aplica scale + rotacao X (sem translacao) a um ponto (0, 0, z_local).
        # Util para descobrir onde acabam o pommel e a lamina em mundo.
        def _saber_offset_from_z(z_local: float):
            sz = z_local * saber_scale_val
            return (0.0, -sz * sin_rx, sz * cos_rx)

        # Posiciona a origem do sabre de forma que o pommel encoste no chao
        # da caixa, exatamente em (saber_pos_xz[0], saber_floor_y, saber_pos_xz[1]).
        pommel_off = _saber_offset_from_z(HANDLE_POMMEL_Z)
        saber_pos = (
            saber_pos_xz[0],
            saber_floor_y - pommel_off[1],
            saber_pos_xz[1] - pommel_off[2],
        )

        self.lightsaber = Entity(
            Mesh.from_obj(str(ASSETS / "lightsaber" / "luke.obj")),
            position=saber_pos,
            rotation=(saber_rot_x, 0, 0),
            scale=(saber_scale_val, saber_scale_val, saber_scale_val),
            kd=(1.00, 1.00, 1.00), ks=(0.55, 0.55, 0.60),
            shininess=64.0, scope="indoor",
        )

        # ---- LAMINA EMISSIVA (cilindro procedural) saindo do emissor ----
        # Vai de z_local = HANDLE_EMITTER_Z ate HANDLE_EMITTER_Z + blade_local_len.
        # Calcula o CENTRO da lamina em coordenadas de mundo aplicando a
        # mesma transform do sabre (scale + rotacao X + translacao).
        blade_center_z_local = HANDLE_EMITTER_Z + blade_local_len / 2.0
        blade_off = _saber_offset_from_z(blade_center_z_local)
        blade_center_world = (
            saber_pos[0] + blade_off[0],
            saber_pos[1] + blade_off[1],
            saber_pos[2] + blade_off[2],
        )
        blade_len_world = blade_local_len * saber_scale_val

        # A luz pontual fica no MEIO DA LAMINA — assim ela ilumina o que
        # esta proximo da lamina e nao do cabo.
        self.light_saber.position = np.array(blade_center_world, dtype=np.float32)

        # Cilindro procedural nasce alinhado a +Y. Para alinha-lo com a
        # lamina (direcao +Z_local do sabre apos rotacao X), basta rotacao
        # em X por (90° + saber_lean_deg). Vale enquanto o sabre nao tiver
        # rotacao em Y ou Z.
        saber_blade_mesh = make_cylinder_lamp_mesh(
            radius=0.06, height=blade_len_world,
        )
        self.bulb_saber_blade = Entity(
            saber_blade_mesh,
            position=blade_center_world,
            rotation=(math.pi / 2 + saber_rot_x, 0, 0),
            kd=tuple(self.light_saber.color), ks=(0, 0, 0),
            shininess=1.0, scope="indoor", emissive=True,
        )

        # =================================================================
        #  OBJETOS-FONTE VISIVEIS (req. visual do enunciado)
        # =================================================================
        # Esfera pequena (farois do rover) e cilindros (tubos do teto)
        # sao gerados proceduralmente. Cada bulbo e renderizado como
        # Entity emissiva (brilha com a propria cor, sem precisar das
        # luzes da cena).

        # ---- ROVER: dois farois esfericos pequenos no para-choque
        # dianteiro do veiculo. Posicoes em "espaco local" do rover
        # (com Z- = frente) e atualizadas em update() para acompanhar
        # a translacao + rotacao do rover.
        # Os offsets foram escolhidos para ficar no topo dianteiro de
        # um rover padrao (ajustar caso o modelo cresca/encolha).
        bulb_mesh_small = make_light_bulb_mesh(radius=0.13)
        # offsets locais (x_lat, y_alt, z_front com frente = -Z)
        self._rover_headlight_left_local  = (-0.3, 1.20, -0.85)
        self._rover_headlight_right_local = ( 0.3, 1.20, -0.85)
        # Centro entre os 2 farois — onde a fonte de luz pontual fica.
        self._rover_light_local           = ( 0.00, 1.20, -0.85)

        self.bulb_rover_l = Entity(
            bulb_mesh_small,
            position=tuple(self.light_rover.position),
            kd=tuple(self.light_rover.color), ks=(0, 0, 0),
            shininess=1.0, scope="outdoor", emissive=True,
        )
        self.bulb_rover_r = Entity(
            bulb_mesh_small,
            position=tuple(self.light_rover.position),
            kd=tuple(self.light_rover.color), ks=(0, 0, 0),
            shininess=1.0, scope="outdoor", emissive=True,
        )

        # ---- TETO: duas lampadas fluorescentes cilindricas suspensas
        # dentro da cupula, posicionadas no MESMO LUGAR do letreiro neon
        # (y = 7.7), alinhadas com a orientacao do neon (Y = 90°).
        #
        # Como o cilindro nasce alinhado ao eixo Y e a Entity aplica
        # rotacao na ordem T·Ry·Rx·Rz·S, com (rx=0, ry=90°, rz=90°) o
        # eixo longo do tubo termina apontando para +Z. Portanto, para
        # separar os dois tubos LADO-A-LADO, a separacao tem que ser
        # PERPENDICULAR ao eixo Z — ou seja, em X.
        cyl_mesh = make_cylinder_lamp_mesh(radius=0.04, height=1.4)
        lamp_y = 7.7
        lamp_sep = 0.10  # distancia entre os dois tubos (no eixo X)
        self.bulb_ceiling_a = Entity(
            cyl_mesh,
            position=(-lamp_sep, lamp_y, 0.0),
            rotation=(0, math.radians(90), math.radians(90)),
            kd=tuple(self.light_ceiling.color), ks=(0, 0, 0),
            shininess=1.0, scope="indoor", emissive=True,
        )
        self.bulb_ceiling_b = Entity(
            cyl_mesh,
            position=(lamp_sep, lamp_y, 0.0),
            rotation=(0, math.radians(90), math.radians(90)),
            kd=tuple(self.light_ceiling.color), ks=(0, 0, 0),
            shininess=1.0, scope="indoor", emissive=True,
        )

        # (Bulbo da lamina do sabre foi criado junto da Entity do sabre,
        #  mais acima — self.bulb_saber_blade. Apenas registramos no link
        #  de bulbos abaixo para que ele acompanhe o toggle da luz 2.)

        # Mapeia cada bulbo a uma luz (varios bulbos podem compartilhar
        # a mesma luz; ex.: 2 farois -> 1 luz pontual do rover).
        # (entity, light_index)
        self._bulb_links = [
            (self.bulb_rover_l,     0),
            (self.bulb_rover_r,     0),
            (self.bulb_ceiling_a,   1),
            (self.bulb_ceiling_b,   1),
            (self.bulb_saber_blade, 2),
        ]
        # Lista plana para iterar no draw().
        self.bulbs: List[Entity] = [b for b, _ in self._bulb_links]

        # =================================================================
        #  LISTAS DE ITERACAO
        # =================================================================
        # Externos: rover ja inclui o farol (luz 0 acompanha a posicao dele).
        self.outdoor_entities: List[Entity] = [
            self.connector, self.spherebase,
            self.nave, self.rover, self.satelite, self.planet,
            self.cart_ext, self.supply_box,
        ]
        # Internos: cama, mesa, robo, etc.
        self.indoor_entities: List[Entity] = [
            self.bed, self.matress, self.bed_back, self.bed_front, self.pipes_bed,
            self.desk, self.robot, self.storage_box, self.neon, self.babyyoda,
            self.bottle, self.trash, self.lightsaber,
        ]

        # ---- Montanhas (externo) ----
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
            # Materiais proprios das montanhas (req. 7).
            m.color = (1.0, 0.92, 0.88)
            m.ks = (0.08, 0.08, 0.08)
            m.shininess = 6.0
            m.light_mask, m.light_mask_back = _scope_to_masks("outdoor")
            self.mountains.append(m)

        # Animacao da nave (orbita).
        self._nave_orbit_radius = 55.0
        self._nave_orbit_height = 35.0
        self._nave_speed = 0.25
        self._nave_angle = 0.0

    # -----------------------------------------------------------------
    def _rover_local_to_world(self, local_offset):
        """Converte um ponto em coordenadas LOCAIS do rover (x lateral,
        y altura, z frente=-Z) para coordenadas de mundo, aplicando
        escala + rotacao Y + translacao do rover.

        Usado para posicionar os farois nos pontos certos do veiculo,
        seguindo qualquer movimento/giro feito pelo teclado.
        """
        ry = float(self.rover.rotation[1])
        cosr = math.cos(ry)
        sinr = math.sin(ry)
        lx, ly, lz = local_offset
        sx, sy, sz = (float(self.rover.scale[0]),
                      float(self.rover.scale[1]),
                      float(self.rover.scale[2]))
        # escala
        ex, ey, ez = lx * sx, ly * sy, lz * sz
        # rotacao Y: x' = x*cos + z*sin ;  z' = -x*sin + z*cos
        rxw = ex * cosr + ez * sinr
        rzw = -ex * sinr + ez * cosr
        # translacao
        return (
            float(self.rover.position[0]) + rxw,
            float(self.rover.position[1]) + ey,
            float(self.rover.position[2]) + rzw,
        )

    def update(self, dt: float):
        """Atualiza animacoes e posicoes dependentes do tempo.

        - Nave orbita a base em circulo.
        - A luz 0 (farol do rover) e os DOIS bulbos esfericos dianteiros
          do rover sao recalculados a partir do transform atual do veiculo
          (translacao + rotacao Y).
        - Cilindros do teto sao fixos (definidos em __init__).
        - Bulbos somem quando sua luz e desligada (feedback visual).
        """
        # ----- Nave em orbita -----
        self._nave_angle += dt * self._nave_speed
        x = math.cos(self._nave_angle) * self._nave_orbit_radius
        z = math.sin(self._nave_angle) * self._nave_orbit_radius
        self.nave.position = np.array([x, self._nave_orbit_height, z], dtype=np.float32)
        self.nave.rotation = np.array(
            [0.0, -self._nave_angle - math.pi / 2.0, 0.0], dtype=np.float32,
        )

        # ----- Farois do rover -----
        # As duas bolinhas vao para os offsets locais dos farois;
        # a fonte de luz pontual fica no meio dos dois (req. do usuario:
        # uma so luz, mas visualmente representada por duas bolinhas).
        wl = self._rover_local_to_world(self._rover_headlight_left_local)
        wr = self._rover_local_to_world(self._rover_headlight_right_local)
        wc = self._rover_local_to_world(self._rover_light_local)
        self.bulb_rover_l.position = np.array(wl, dtype=np.float32)
        self.bulb_rover_r.position = np.array(wr, dtype=np.float32)
        self.light_rover.position  = np.array(wc, dtype=np.float32)

        # ----- Visibilidade dos bulbos conforme on/off da luz -----
        for bulb, light_idx in self._bulb_links:
            bulb.visible = self.lights[light_idx].on

    # -----------------------------------------------------------------
    def draw(self, shader, wireframe: bool = False):
        """Renderiza tudo. Skybox e desenhado fora desta funcao."""
        # piso externo
        self.mars.draw(shader, wireframe=wireframe)
        # piso interno
        self.indoor_floor.draw(shader, wireframe=wireframe)
        # montanhas
        for mtn in self.mountains:
            mtn.draw(shader, wireframe=wireframe)

        # Sciencebase (com cull off pra ver paredes por dentro)
        glDisable(GL_CULL_FACE)
        self.sciencebase.draw(shader, wireframe=wireframe)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # demais entidades
        for e in self.outdoor_entities:
            e.draw(shader, wireframe=wireframe)
        for e in self.indoor_entities:
            e.draw(shader, wireframe=wireframe)
        # bulbos das luzes (esferas emissivas)
        for b in self.bulbs:
            b.draw(shader, wireframe=wireframe)
