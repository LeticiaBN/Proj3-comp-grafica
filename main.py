"""
Computação Gráfica — Projeto 2
Cenário 3D: Base Científica em Marte (interno + externo).

Pipeline moderno do OpenGL: VAO/VBO + shaders GLSL + matrizes em numpy.
Sem efeitos de iluminação (proibido pelo PDF).

Controles:
  Câmera:
    W A S D            — andar
    Espaço / Shift     — subir / descer
    Mouse              — olhar em volta
    ESC                — sair
  Transformações (regra 7 do PDF):
    Setas (↑↓←→)       — translação do rover
    R / T              — rotação do satélite (esq / dir)
    + / -              — escala do planeta de fundo (cresce / diminui)
  Visualização:
    P                  — toggle wireframe
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

# Altura dos olhos acima do chão (câmera FPS)
EYE_HEIGHT = 3.0

ROOT = Path(__file__).resolve().parent
SHADERS = ROOT / "shaders"

WIDTH, HEIGHT = 1280, 800
TITLE = "Projeto 2 — Base Cientifica em Marte"


class InputState:
    def __init__(self):
        self.first_mouse = True
        self.last_x = WIDTH / 2
        self.last_y = HEIGHT / 2
        self.wireframe = False
        self.p_was_down = False


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

    # parâmetros das transformações por teclado
    rover_speed = 6.0
    satelite_rot_speed = 1.5
    planet_scale_speed = 5.0
    planet_min_scale, planet_max_scale = 5.0, 80.0

    print("[init] OK. Entrando no loop.")
    last_t = glfw.get_time()
    while not glfw.window_should_close(win):
        now = glfw.get_time()
        dt = now - last_t
        last_t = now

        # ---------- INPUT ----------
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(win, True)

        # câmera
        fwd = (glfw.get_key(win, glfw.KEY_W) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_S) == glfw.PRESS)
        rgt = (glfw.get_key(win, glfw.KEY_D) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_A) == glfw.PRESS)
        upd = (glfw.get_key(win, glfw.KEY_SPACE) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS)
        cam.process_keyboard(dt, fwd, rgt, upd)

        # rover (translação) — setas
        rdx = (glfw.get_key(win, glfw.KEY_RIGHT) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_LEFT) == glfw.PRESS)
        rdz = (glfw.get_key(win, glfw.KEY_DOWN) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_UP) == glfw.PRESS)
        if rdx or rdz:
            scene.rover.position[0] += rdx * rover_speed * dt
            scene.rover.position[2] += rdz * rover_speed * dt
            # mantém rover no terreno
            scene.rover.position[0] = max(-WORLD_HALF + 5, min(WORLD_HALF - 5, scene.rover.position[0]))
            scene.rover.position[2] = max(-WORLD_HALF + 5, min(WORLD_HALF - 5, scene.rover.position[2]))
            
            # Ajusta o Y do rover para grudar no terreno
            scene.rover.position[1] = terrain_height(scene.rover.position[0], scene.rover.position[2], BASE_FOOTPRINT_RADIUS)
            
            # vira a "cara" do rover na direção do movimento
            if rdx != 0 or rdz != 0:
                scene.rover.rotation[1] = math.atan2(rdx, -rdz)

        # satélite (rotação) — R / T
        srot = (glfw.get_key(win, glfw.KEY_T) == glfw.PRESS) - (glfw.get_key(win, glfw.KEY_R) == glfw.PRESS)
        if srot:
            scene.satelite.rotation[1] += srot * satelite_rot_speed * dt

        # planeta (escala) — + / -
        plus = (glfw.get_key(win, glfw.KEY_EQUAL) == glfw.PRESS or
                glfw.get_key(win, glfw.KEY_KP_ADD) == glfw.PRESS)
        minus = (glfw.get_key(win, glfw.KEY_MINUS) == glfw.PRESS or
                 glfw.get_key(win, glfw.KEY_KP_SUBTRACT) == glfw.PRESS)
        if plus or minus:
            d = (1 if plus else 0) - (1 if minus else 0)
            new_scale = float(scene.planet.scale[0]) + d * planet_scale_speed * dt
            new_scale = max(planet_min_scale, min(planet_max_scale, new_scale))
            scene.planet.scale = np.array([new_scale, new_scale, new_scale], dtype=np.float32)

        # wireframe toggle
        p_now = glfw.get_key(win, glfw.KEY_P) == glfw.PRESS
        if p_now and not inp.p_was_down:
            inp.wireframe = not inp.wireframe
            print(f"[wireframe] {'ON' if inp.wireframe else 'OFF'}")
        inp.p_was_down = p_now
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if inp.wireframe else GL_FILL)

        # ---------- UPDATE ----------
        scene.update(dt)

        # Ajusta o limite Y mínimo da câmera conforme o relevo do terreno.
        # terrain_height() retorna 0 dentro da base (blend=0) e o
        # deslocamento senoidal fora dela — a câmera nunca entra no chão.
        th = terrain_height(cam.position[0], cam.position[2], BASE_FOOTPRINT_RADIUS)
        ymin = th + EYE_HEIGHT
        (xb, zb) = (cam.bounds[0], cam.bounds[2])
        cam.bounds = (xb, (ymin, SKY_HEIGHT - 5), zb)
        # garante que a câmera não fique presa abaixo do novo mínimo
        if cam.position[1] < ymin:
            cam.position[1] = ymin

        # ---------- RENDER ----------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        w, h = glfw.get_framebuffer_size(win)
        proj = T.perspective(math.radians(60.0), w / max(h, 1), 0.1, 1500.0)
        view = cam.view_matrix()

        # 1) Skybox primeiro (com view sem translação)
        view_no_t = view.copy()
        view_no_t[0, 3] = view_no_t[1, 3] = view_no_t[2, 3] = 0.0
        scene.skybox.draw(sky_shader, view_no_t, proj)

        # 2) Resto da cena
        basic.use()
        basic.set_mat4("u_proj", proj)
        basic.set_mat4("u_view", view)
        scene.draw(basic, wireframe=inp.wireframe)

        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    sys.exit(main() or 0)
