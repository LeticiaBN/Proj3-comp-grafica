"""Helpers de matrizes 4x4 com numpy. Substituem as funções obsoletas
do OpenGL antigo (glTranslate, glRotate, glScale, glLoadIdentity, etc.)."""
import math

import numpy as np


def identity() -> np.ndarray:
    # matriz identidade 4x4: nao altera nada quando multiplicada
    return np.identity(4, dtype=np.float32)


def translate(x: float, y: float, z: float) -> np.ndarray:
    # cria matriz de translacao: move um ponto pelos valores (x,y,z)
    m = identity()
    # ultima coluna recebe o deslocamento em cada eixo
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale(sx: float, sy: float = None, sz: float = None) -> np.ndarray:
    # se passar so um valor, escala uniforme nos 3 eixos
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    m = identity()
    # diagonal principal define quanto multiplicar cada eixo
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotate_x(rad: float) -> np.ndarray:
    # rotacao em torno do eixo x (afeta y e z)
    c, s = math.cos(rad), math.sin(rad)
    m = identity()
    m[1, 1] = c;  m[1, 2] = -s
    m[2, 1] = s;  m[2, 2] = c
    return m


def rotate_y(rad: float) -> np.ndarray:
    # rotacao em torno do eixo y (afeta x e z) — mais usada para "girar" no plano horizontal
    c, s = math.cos(rad), math.sin(rad)
    m = identity()
    m[0, 0] = c;  m[0, 2] = s
    m[2, 0] = -s; m[2, 2] = c
    return m


def rotate_z(rad: float) -> np.ndarray:
    # rotacao em torno do eixo z (afeta x e y)
    c, s = math.cos(rad), math.sin(rad)
    m = identity()
    m[0, 0] = c;  m[0, 1] = -s
    m[1, 0] = s;  m[1, 1] = c
    return m


def perspective(fov_rad: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    # matriz de projecao perspectiva: simula como uma camera real "ve" o mundo (longe = pequeno)
    f = 1.0 / math.tan(fov_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    # f/aspect compensa o aspecto da tela (largura/altura)
    m[0, 0] = f / aspect
    m[1, 1] = f
    # mapeia profundidade [znear, zfar] para o intervalo do clip space
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    # -1 em w faz a divisao perspectiva acontecer
    m[3, 2] = -1.0
    return m


def look_at(eye, center, up) -> np.ndarray:
    # constroi a matriz "view" da camera olhando do ponto eye para center
    eye = np.asarray(eye, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    # f = direcao para frente da camera (do olho para o alvo), normalizada
    f = center - eye
    f /= np.linalg.norm(f)
    # s = direita da camera (perpendicular a f e up)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    # u = up real da camera (recalculado para ser ortogonal)
    u = np.cross(s, f)

    # monta a matriz com as 3 direcoes nas linhas + posicao na ultima coluna
    m = identity()
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    # translada o mundo para que a camera fique na origem
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def to_gl(m: np.ndarray) -> np.ndarray:
    """Converte matriz row-major (numpy) para o formato column-major do GL.
    Usar com glUniformMatrix4fv(loc, 1, GL_TRUE, mat) ou transpor antes."""
    return np.ascontiguousarray(m, dtype=np.float32)
