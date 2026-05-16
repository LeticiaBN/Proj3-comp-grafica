"""Malhas geradas em runtime para visualizar fontes de luz na cena.

Disponiveis:
  - SphereBulb  : pequena esfera UV (usada nos farois do rover).
  - CylinderBulb: cilindro alinhado ao eixo Y (usado nas lampadas
                  fluorescentes do teto da base).

Ambas sao usadas como "marcadores visuais" das fontes de luz pontuais.
Renderizadas como Entity com flag ``emissive=True`` para brilharem
com a propria cor, independente das luzes da cena (caso contrario uma
lampada nao brilharia se sua propria luz fosse a unica afetando-a).
"""
import math
from typing import List, Tuple

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_STATIC_DRAW, GL_TEXTURE0,
    GL_TEXTURE_2D, GL_TRIANGLES, glActiveTexture, glBindBuffer,
    glBindTexture, glBindVertexArray, glBufferData, glDrawArrays,
    glEnableVertexAttribArray, glGenBuffers, glGenVertexArrays,
    glVertexAttribPointer,
)

from src.floor import _white_pixel_texture
from src.mesh import ctypes_void_p


def _generate_uv_sphere(radius: float, stacks: int, sectors: int) -> np.ndarray:
    """Gera vertices intercalados [pos3, uv2, normal3] de uma esfera UV.

    Triangulos em CCW (visto de fora). Suficiente para renderizar o
    pequeno "bulbo" no topo do rover e dentro das luminarias internas.
    """
    verts: List[float] = []
    # gera grade (stacks x sectors) e quebra cada quad em dois triangulos
    for i in range(stacks):
        phi1 = math.pi * (i / stacks) - math.pi / 2.0       # latitude inferior
        phi2 = math.pi * ((i + 1) / stacks) - math.pi / 2.0  # latitude superior
        for j in range(sectors):
            th1 = 2.0 * math.pi * (j / sectors)             # longitude esquerda
            th2 = 2.0 * math.pi * ((j + 1) / sectors)       # longitude direita

            # 4 vertices do quad atual (cantos)
            def vert(phi, theta):
                # posicao na esfera de raio R (coordenadas esfericas -> XYZ)
                x = radius * math.cos(phi) * math.cos(theta)
                y = radius * math.sin(phi)
                z = radius * math.cos(phi) * math.sin(theta)
                # uv simples (cilindrico); nao importa muito porque o
                # objeto e renderizado unlit e a textura e branca 1x1.
                u = theta / (2.0 * math.pi)
                v = (phi + math.pi / 2.0) / math.pi
                # normal na esfera = vetor radial normalizado
                nx, ny, nz = x / radius, y / radius, z / radius
                return [x, y, z, u, v, nx, ny, nz]

            v00 = vert(phi1, th1)
            v01 = vert(phi1, th2)
            v10 = vert(phi2, th1)
            v11 = vert(phi2, th2)
            # Tri 1: v00 -> v01 -> v11
            verts += v00 + v01 + v11
            # Tri 2: v00 -> v11 -> v10
            verts += v00 + v11 + v10
    return np.array(verts, dtype=np.float32)


class _SphereMesh:
    """VAO/VBO + draw para a esferinha gerada. Implementa a mesma
    interface (.draw(shader, wireframe)) que Mesh, para poder ser
    embrulhada em Entity sem mudancas no resto do codigo."""

    def __init__(self, radius: float = 0.4, stacks: int = 12, sectors: int = 18):
        buf = _generate_uv_sphere(radius, stacks, sectors)
        self._vertex_count = len(buf) // 8
        # textura branca 1x1 para o shader ainda ter um sampler valido
        self.texture = _white_pixel_texture()

        # upload classico para a GPU (idem GpuSubMesh)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, buf.nbytes, buf, GL_STATIC_DRAW)
        stride = 8 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes_void_p(20))
        glBindVertexArray(0)

    def draw(self, shader, wireframe: bool = False):
        # textura branca + u_wireframe; u_kd/u_ks/u_unlit vem da Entity
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        shader.set_int("u_tex", 0)
        shader.set_int("u_wireframe", 1 if wireframe else 0)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self._vertex_count)
        glBindVertexArray(0)


def make_light_bulb_mesh(radius: float = 0.4) -> _SphereMesh:
    """Atalho usado pela Scene: cria a esferinha que serve de "lampada"."""
    return _SphereMesh(radius=radius)


# ---------------------------------------------------------------------
#  CILINDRO (para lampadas fluorescentes do teto)
# ---------------------------------------------------------------------
def _generate_cylinder(
    radius: float,
    height: float,
    segments: int = 24,
) -> np.ndarray:
    """Gera um cilindro alinhado ao eixo Y, centrado em (0, height/2, 0).

    Tres partes: lateral + tampa superior + tampa inferior. Layout dos
    vertices identico ao resto do projeto (pos3 + uv2 + normal3).
    Winding CCW visto de fora.
    """
    verts: List[float] = []
    half = height / 2.0

    # ----- Lateral (faces externas com normal radial) -----
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        cos0, sin0 = math.cos(a0), math.sin(a0)
        cos1, sin1 = math.cos(a1), math.sin(a1)
        # 4 cantos do quad lateral
        x0, z0 = radius * cos0, radius * sin0
        x1, z1 = radius * cos1, radius * sin1
        # uv: u em volta do cilindro, v ao longo do comprimento
        u0, u1 = i / segments, (i + 1) / segments
        # normais radiais (apontando para fora) — ja normalizadas pois
        # (cos, 0, sin) tem modulo 1
        n0 = (cos0, 0.0, sin0)
        n1 = (cos1, 0.0, sin1)
        # Triangulos do quad: (top0, bottom0, bottom1) e (top0, bottom1, top1)
        # — CCW visto de fora.
        verts += [x0,  half, z0, u0, 1.0, *n0]
        verts += [x0, -half, z0, u0, 0.0, *n0]
        verts += [x1, -half, z1, u1, 0.0, *n1]
        verts += [x0,  half, z0, u0, 1.0, *n0]
        verts += [x1, -half, z1, u1, 0.0, *n1]
        verts += [x1,  half, z1, u1, 1.0, *n1]

    # ----- Tampas (top com normal +Y, bottom com normal -Y) -----
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        x0, z0 = radius * math.cos(a0), radius * math.sin(a0)
        x1, z1 = radius * math.cos(a1), radius * math.sin(a1)
        # Top: centro -> p0 -> p1 (CCW visto de cima => normal +Y)
        verts += [0.0,  half, 0.0, 0.5, 0.5, 0.0,  1.0, 0.0]
        verts += [x0,   half, z0,  0.5 + 0.5 * math.cos(a0), 0.5 + 0.5 * math.sin(a0), 0.0,  1.0, 0.0]
        verts += [x1,   half, z1,  0.5 + 0.5 * math.cos(a1), 0.5 + 0.5 * math.sin(a1), 0.0,  1.0, 0.0]
        # Bottom: centro -> p1 -> p0 (CCW visto de baixo => normal -Y)
        verts += [0.0, -half, 0.0, 0.5, 0.5, 0.0, -1.0, 0.0]
        verts += [x1,  -half, z1,  0.5 + 0.5 * math.cos(a1), 0.5 + 0.5 * math.sin(a1), 0.0, -1.0, 0.0]
        verts += [x0,  -half, z0,  0.5 + 0.5 * math.cos(a0), 0.5 + 0.5 * math.sin(a0), 0.0, -1.0, 0.0]

    return np.array(verts, dtype=np.float32)


class _CylinderMesh:
    """VAO/VBO + draw para um cilindro procedural. Mesma interface
    de Mesh para poder ser usada como Entity."""

    def __init__(self, radius: float = 0.2, height: float = 4.0, segments: int = 24):
        buf = _generate_cylinder(radius, height, segments)
        self._vertex_count = len(buf) // 8
        self.texture = _white_pixel_texture()

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, buf.nbytes, buf, GL_STATIC_DRAW)
        stride = 8 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes_void_p(20))
        glBindVertexArray(0)

    def draw(self, shader, wireframe: bool = False):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        shader.set_int("u_tex", 0)
        shader.set_int("u_wireframe", 1 if wireframe else 0)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self._vertex_count)
        glBindVertexArray(0)


def make_cylinder_lamp_mesh(radius: float = 0.2, height: float = 4.0) -> _CylinderMesh:
    """Atalho: cria a malha de um "tubo" fluorescente do teto.

    O cilindro nasce alinhado ao eixo Y. Para deixar a lampada DEITADA
    horizontalmente (formato fluorescente), aplique uma rotacao Z=90 ou
    X=90 na Entity que envolve esta malha.
    """
    return _CylinderMesh(radius=radius, height=height)
