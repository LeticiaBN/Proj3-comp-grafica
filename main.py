"""
Computacao Grafica — Projeto 3
Cenario 3D: Base Cientifica em Marte (interno + externo) com
iluminacao Phong por fragmento (ambiente + difusa + especular).

Pipeline moderno do OpenGL: VAO/VBO + shaders GLSL + matrizes numpy.

Fontes de luz (req. 1 e 2 do enunciado):
  - Luz 0 (externa)  : farol do ROVER. Translada com o rover; afeta
                        APENAS objetos externos.
  - Luz 1 (interna)  : lampada de teto da base — branca-quente.
  (Uma segunda luz interna sera adicionada quando o novo objeto-fonte
   for inserido na cena.)

Controles:
  Camera:
    W A S D            — andar
    Espaco / Shift     — subir / descer
    Mouse              — olhar em volta
    ESC                — sair
  Movimento do rover (a luz externa o acompanha):
    Setas (↑ ↓ ← →)    — translacao do rover no chao de Marte
  Transformacoes herdadas do Projeto 2:
    R / T              — rotacao do satelite (esquerda / direita)
    = / -              — escala do planeta (cresce / diminui)
                          (tambem aceita +/- do numpad)
  Interruptores das luzes (req. 3):
    1                  — liga/desliga FAROL DO ROVER (externa)
    2                  — liga/desliga LAMPADA DE TETO (interna)
    4                  — liga/desliga LUZ AMBIENTE
  Coeficientes globais (req. 4, 5, 6):
    Z / X              — diminui / aumenta intensidade da luz AMBIENTE
    C / V              — diminui / aumenta reflexao DIFUSA
    B / N              — diminui / aumenta reflexao ESPECULAR
  Visualizacao:
    P                  — alterna entre malha poligonal (wireframe) e
                         preenchido (fill)
"""
import math
import sys
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import (
    GL_BACK, GL_COLOR_BUFFER_BIT, GL_CULL_FACE, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_FILL, GL_FRONT_AND_BACK, GL_LESS, GL_LINE,
    glClear, glClearColor, glCullFace, glDepthFunc, glEnable, glPolygonMode,
    glViewport,
)

from src import transforms as T
from src.camera import FpsCamera
from src.floor import _height as terrain_height
from src.scene import Scene, SKY_HEIGHT, WORLD_HALF, BASE_FOOTPRINT_RADIUS
from src.shader import Shader

# Altura dos olhos acima do chao (camera FPS)
EYE_HEIGHT = 3.0

ROOT = Path(__file__).resolve().parent
SHADERS = ROOT / "shaders"

WIDTH, HEIGHT = 1280, 800
TITLE = "Projeto 3 — Base Cientifica em Marte (Iluminacao)"


class InputState:
    """Estado de teclado pensado para detectar "edge" (foi pressionada
    agora, mas nao no frame anterior). Usado para os toggles."""
    def __init__(self):
        self.first_mouse = True
        self.last_x = WIDTH / 2
        self.last_y = HEIGHT / 2
        # estados anteriores das teclas de toggle (para detectar borda)
        self.k1_was = False
        self.k2_was = False
        self.k4_was = False
        # toggle de malha poligonal (wireframe) — tecla P
        self.p_was = False
        self.wireframe = False


def make_window():
    if not glfw.init():
        raise RuntimeError("Falha ao inicializar GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    win = glfw.create_window(WIDTH, HEIGHT, TITLE, None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("Falha ao criar janela GLFW")
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    return win


def main():
    win = make_window()
    inp = InputState()

    print("[init] carregando shaders...")
    basic = Shader.from_files(SHADERS / "basic.vs", SHADERS / "basic.fs")
    sky_shader = Shader.from_files(SHADERS / "skybox.vs", SHADERS / "skybox.fs")

    print("[init] carregando cena...")
    scene = Scene()

    cam = FpsCamera(
        position=(0.0, EYE_HEIGHT, 30.0),
        yaw_deg=-90.0, pitch_deg=-5.0,
        speed=14.0, sensitivity=0.13,
        bounds=((-WORLD_HALF + 5, WORLD_HALF - 5),
                (EYE_HEIGHT, SKY_HEIGHT - 5),
                (-WORLD_HALF + 5, WORLD_HALF - 5)),
    )

    glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_DISABLED)

    def on_mouse(window, xpos, ypos):
        if inp.first_mouse:
            inp.last_x, inp.last_y = xpos, ypos
            inp.first_mouse = False
        dx = xpos - inp.last_x
        dy = ypos - inp.last_y
        inp.last_x, inp.last_y = xpos, ypos
        cam.process_mouse(dx, dy)

    def on_resize(window, w, h):
        glViewport(0, 0, w, h)

    glfw.set_cursor_pos_callback(win, on_mouse)
    glfw.set_framebuffer_size_callback(win, on_resize)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glClearColor(0.02, 0.02, 0.04, 1.0)

    # =====================================================================
    #  ESTADO GLOBAL DE ILUMINACAO (controlado por teclado)
    # =====================================================================
    # Componente ambiente: cor + intensidade + interruptor proprio.
    ambient_color = (0.6, 0.6, 0.7)   # ambiente levemente azulado
    ka_strength = 0.20                 # forca da luz ambiente (req. 4)
    ambient_on = True

    # Multiplicadores globais da reflexao difusa e especular (req. 5, 6).
    # Comecam em 1.0 (intensidade "natural" do material proprio do objeto).
    diffuse_strength = 1.0
    specular_strength = 1.0

    # Velocidade do rover (translacao por teclado; a luz externa o acompanha).
    rover_speed = 6.0
    # Velocidade angular do satelite (rad/s).
    satelite_rot_speed = 1.5
    # Velocidade de escala do planeta e limites (uniformemente nos 3 eixos).
    planet_scale_speed = 5.0
    planet_min_scale, planet_max_scale = 5.0, 80.0

    # Passos por segundo para incremento/decremento das teclas de intensidade.
    AMBIENT_STEP   = 0.6   # /s
    DIFFUSE_STEP   = 0.8
    SPECULAR_STEP  = 0.8

    print("[init] OK. Entrando no loop.")
    print("[ctrls] 1/2/3=toggle luzes  4=toggle ambiente  Z/X=ambiente  C/V=difusa  B/N=especular")
    last_t = glfw.get_time()
    while not glfw.window_should_close(win):
        now = glfw.get_time()
        dt = now - last_t
        last_t = now

        # ---------------------------------------------------------------
        # INPUT
        # ---------------------------------------------------------------
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(win, True)

        # --- camera ---
        fwd = (glfw.get_key(win, glfw.KEY_W) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_S) == glfw.PRESS)
        rgt = (glfw.get_key(win, glfw.KEY_D) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_A) == glfw.PRESS)
        upd = (glfw.get_key(win, glfw.KEY_SPACE) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS)
        cam.process_keyboard(dt, fwd, rgt, upd)

        # --- rover (continua sendo movido por teclado: e o objeto que
        #     carrega a fonte de luz externa, req. 1). ---
        rdx = (glfw.get_key(win, glfw.KEY_RIGHT) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_LEFT) == glfw.PRESS)
        rdz = (glfw.get_key(win, glfw.KEY_DOWN) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_UP) == glfw.PRESS)
        if rdx or rdz:
            scene.rover.position[0] += rdx * rover_speed * dt
            scene.rover.position[2] += rdz * rover_speed * dt
            scene.rover.position[0] = max(-WORLD_HALF + 5, min(WORLD_HALF - 5, scene.rover.position[0]))
            scene.rover.position[2] = max(-WORLD_HALF + 5, min(WORLD_HALF - 5, scene.rover.position[2]))
            scene.rover.position[1] = terrain_height(
                scene.rover.position[0], scene.rover.position[2], BASE_FOOTPRINT_RADIUS,
            )
            if rdx != 0 or rdz != 0:
                # rotate_y(theta) leva a frente local do rover (-Z) para
                # (-sin theta, 0, -cos theta). Para que essa frente coincida
                # com a direcao do movimento (rdx, rdz) precisamos de:
                #   sin theta = -rdx  e  cos theta = -rdz
                # Logo theta = atan2(-rdx, -rdz). O sinal de rdx era o
                # culpado pelo rover andar "de costas" com setas LEFT/RIGHT.
                scene.rover.rotation[1] = math.atan2(-rdx, -rdz)

        # --- satelite (rotacao Y) — R / T ---
        srot = (glfw.get_key(win, glfw.KEY_T) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_R) == glfw.PRESS)
        if srot:
            scene.satelite.rotation[1] += srot * satelite_rot_speed * dt

        # --- planeta (escala uniforme) — = / -  (ou +/- do numpad) ---
        plus = (glfw.get_key(win, glfw.KEY_EQUAL) == glfw.PRESS or
                glfw.get_key(win, glfw.KEY_KP_ADD) == glfw.PRESS)
        minus = (glfw.get_key(win, glfw.KEY_MINUS) == glfw.PRESS or
                 glfw.get_key(win, glfw.KEY_KP_SUBTRACT) == glfw.PRESS)
        if plus or minus:
            d = (1 if plus else 0) - (1 if minus else 0)
            new_scale = float(scene.planet.scale[0]) + d * planet_scale_speed * dt
            new_scale = max(planet_min_scale, min(planet_max_scale, new_scale))
            scene.planet.scale = np.array([new_scale, new_scale, new_scale], dtype=np.float32)

        # --- toggles de luzes (1=rover, 2=teto, 3=abajur, 4=ambiente) ---
        # Borda de subida: liga/desliga apenas no frame em que a tecla
        # passa de "solta" para "pressionada".
        k1_now = glfw.get_key(win, glfw.KEY_1) == glfw.PRESS
        if k1_now and not inp.k1_was:
            scene.lights[0].on = not scene.lights[0].on
            print(f"[luz] farol do rover = {'ON' if scene.lights[0].on else 'OFF'}")
        inp.k1_was = k1_now

        k2_now = glfw.get_key(win, glfw.KEY_2) == glfw.PRESS
        if k2_now and not inp.k2_was:
            scene.lights[1].on = not scene.lights[1].on
            print(f"[luz] lampada de teto = {'ON' if scene.lights[1].on else 'OFF'}")
        inp.k2_was = k2_now

        k4_now = glfw.get_key(win, glfw.KEY_4) == glfw.PRESS
        if k4_now and not inp.k4_was:
            ambient_on = not ambient_on
            print(f"[luz] ambiente = {'ON' if ambient_on else 'OFF'}")
        inp.k4_was = k4_now

        # --- toggle de malha poligonal (wireframe) — tecla P ---
        p_now = glfw.get_key(win, glfw.KEY_P) == glfw.PRESS
        if p_now and not inp.p_was:
            inp.wireframe = not inp.wireframe
            print(f"[wireframe] {'ON' if inp.wireframe else 'OFF'}")
        inp.p_was = p_now

        # --- incremento/decremento dos coeficientes (segura para repetir) ---
        # Luz ambiente Z / X (req. 4).
        if glfw.get_key(win, glfw.KEY_Z) == glfw.PRESS:
            ka_strength = max(0.0, ka_strength - AMBIENT_STEP * dt)
        if glfw.get_key(win, glfw.KEY_X) == glfw.PRESS:
            ka_strength = min(2.0, ka_strength + AMBIENT_STEP * dt)
        # Reflexao difusa C / V (req. 5).
        if glfw.get_key(win, glfw.KEY_C) == glfw.PRESS:
            diffuse_strength = max(0.0, diffuse_strength - DIFFUSE_STEP * dt)
        if glfw.get_key(win, glfw.KEY_V) == glfw.PRESS:
            diffuse_strength = min(4.0, diffuse_strength + DIFFUSE_STEP * dt)
        # Reflexao especular B / N (req. 6).
        if glfw.get_key(win, glfw.KEY_B) == glfw.PRESS:
            specular_strength = max(0.0, specular_strength - SPECULAR_STEP * dt)
        if glfw.get_key(win, glfw.KEY_N) == glfw.PRESS:
            specular_strength = min(4.0, specular_strength + SPECULAR_STEP * dt)

        # ---------------------------------------------------------------
        # UPDATE
        # ---------------------------------------------------------------
        scene.update(dt)

        # Adapta o limite Y minimo da camera ao relevo do terreno.
        th = terrain_height(cam.position[0], cam.position[2], BASE_FOOTPRINT_RADIUS)
        ymin = th + EYE_HEIGHT
        (xb, zb) = (cam.bounds[0], cam.bounds[2])
        cam.bounds = (xb, (ymin, SKY_HEIGHT - 5), zb)
        if cam.position[1] < ymin:
            cam.position[1] = ymin

        # ---------------------------------------------------------------
        # RENDER
        # ---------------------------------------------------------------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        w, h = glfw.get_framebuffer_size(win)
        proj = T.perspective(math.radians(60.0), w / max(h, 1), 0.1, 1500.0)
        view = cam.view_matrix()

        # 1) Skybox primeiro (com view sem translacao)
        view_no_t = view.copy()
        view_no_t[0, 3] = view_no_t[1, 3] = view_no_t[2, 3] = 0.0
        scene.skybox.draw(sky_shader, view_no_t, proj)

        # 2) Resto da cena com shader basico (Phong)
        basic.use()
        basic.set_mat4("u_proj", proj)
        basic.set_mat4("u_view", view)

        # ---- Uniforms de iluminacao globais (atualizados todo frame) ----
        # Posicao do observador (camera) para a reflexao especular.
        basic.set_vec3("u_view_pos", *cam.position)

        # Ambiente.
        basic.set_vec3("u_ambient_color", *ambient_color)
        basic.set_float("u_ka_strength", ka_strength)
        basic.set_int("u_ambient_on", 1 if ambient_on else 0)

        # Multiplicadores globais difusa/especular.
        basic.set_float("u_diffuse_strength", diffuse_strength)
        basic.set_float("u_specular_strength", specular_strength)

        # Arrays de luzes (3 pontuais). Mesmo se uma estiver "off",
        # mandamos a posicao/cor — o shader ignora pelo flag u_light_on[].
        positions = [tuple(l.position) for l in scene.lights]
        colors    = [tuple(l.color)    for l in scene.lights]
        on_flags  = [1 if l.on else 0  for l in scene.lights]
        basic.set_vec3_array("u_light_pos",   positions)
        basic.set_vec3_array("u_light_color", colors)
        basic.set_int_array("u_light_on",     on_flags)

        # ---- Desenha a cena ----
        # Tecla P alterna entre FILL (preenchido) e LINE (malha poligonal).
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if inp.wireframe else GL_FILL)
        scene.draw(basic, wireframe=inp.wireframe)

        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    sys.exit(main() or 0)
