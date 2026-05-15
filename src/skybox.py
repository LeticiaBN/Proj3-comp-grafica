"""Skybox renderer (cubemap em cubo unitário)."""
import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_LEQUAL, GL_LESS,
    GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE_CUBE_MAP, GL_TRIANGLES,
    glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray,
    glBufferData, glDepthFunc, glDrawArrays, glEnableVertexAttribArray,
    glGenBuffers, glGenVertexArrays, glVertexAttribPointer,
)

from src.mesh import ctypes_void_p
from src.texture import load_cubemap


# 36 vértices (6 faces x 2 tris x 3 verts), só posição
_SKYBOX_VERTS = np.array([
    # +X (right)
     1, -1, -1,  1,  1,  1,  1,  1, -1,
     1, -1, -1,  1, -1,  1,  1,  1,  1,
    # -X (left)
    -1, -1, -1, -1,  1, -1, -1,  1,  1,
    -1, -1, -1, -1,  1,  1, -1, -1,  1,
    # +Y (top)
    -1,  1, -1,  1,  1, -1,  1,  1,  1,
    -1,  1, -1,  1,  1,  1, -1,  1,  1,
    # -Y (bottom)
    -1, -1, -1, -1, -1,  1,  1, -1,  1,
    -1, -1, -1,  1, -1,  1,  1, -1, -1,
    # +Z (front)
    -1, -1,  1, -1,  1,  1,  1,  1,  1,
    -1, -1,  1,  1,  1,  1,  1, -1,  1,
    # -Z (back)
    -1, -1, -1,  1, -1, -1,  1,  1, -1,
    -1, -1, -1,  1,  1, -1, -1,  1, -1,
], dtype=np.float32)


class Skybox:
    """Cubemap skybox. face_paths em ordem GL: +X,-X,+Y,-Y,+Z,-Z."""

    def __init__(self, face_paths):
        # carrega as 6 imagens do cubemap como uma unica textura
        self.cubemap = load_cubemap(face_paths)
        # cria vao/vbo do cubo unitario que serve de "tela" para o ceu
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # envia os 36 vertices do cubo
        glBufferData(GL_ARRAY_BUFFER, _SKYBOX_VERTS.nbytes, _SKYBOX_VERTS, GL_STATIC_DRAW)
        # so usamos posicao (3 floats por vertice = stride de 12 bytes)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes_void_p(0))
        glBindVertexArray(0)

    def draw(self, shader, view_no_translation: np.ndarray, proj: np.ndarray):
        # GL_LEQUAL deixa o skybox passar no z-test mesmo na profundidade maxima
        glDepthFunc(GL_LEQUAL)
        shader.use()
        # view sem translacao: ceu nunca "se aproxima" quando andamos
        shader.set_mat4("u_view", view_no_translation)
        shader.set_mat4("u_proj", proj)
        # binda o cubemap na unidade de textura 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.cubemap)
        shader.set_int("u_skybox", 0)
        # desenha os 36 vertices (6 faces x 2 triangulos x 3 vertices)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)
        # restaura o teste de profundidade padrao
        glDepthFunc(GL_LESS)
