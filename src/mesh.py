"""Mesh = conjunto de SubMeshes carregados em GPU.
Cada submesh tem seu próprio VAO/VBO + textura diffuse.
"""
from typing import List, Optional

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_STATIC_DRAW, GL_TEXTURE0,
    GL_TEXTURE_2D, GL_TRIANGLES, glActiveTexture, glBindBuffer,
    glBindTexture, glBindVertexArray, glBufferData, glDrawArrays,
    glEnableVertexAttribArray, glGenBuffers, glGenVertexArrays,
    glVertexAttribPointer,
)

from src.obj_loader import SubMesh, load_obj
from src.texture import load_texture_2d


class GpuSubMesh:
    """SubMesh já enviada para GPU (VAO/VBO + textura)."""

    STRIDE = 8 * 4  # 8 floats * 4 bytes = 32 bytes

    def __init__(self, submesh: SubMesh, fallback_texture: Optional[int] = None):
        # guarda nome do material e cor difusa para uso no shader
        self.material = submesh.material
        self.kd = submesh.kd
        # divide por 8 porque cada vertice tem 8 floats (3 pos + 2 uv + 3 normal)
        self.vertex_count = len(submesh.vertices) // 8

        # textura
        if submesh.diffuse_texture:
            # se o material tem textura propria, carrega do disco
            self.texture = load_texture_2d(submesh.diffuse_texture)
        else:
            self.texture = fallback_texture  # pode ser None

        # VAO + VBO
        # vao guarda o "estado" dos atributos, vbo guarda os dados em si
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # envia o array de vertices para a memoria da gpu
        glBufferData(GL_ARRAY_BUFFER, submesh.vertices.nbytes, submesh.vertices, GL_STATIC_DRAW)

        # location 0: position (vec3) offset 0
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.STRIDE, ctypes_void_p(0))
        # location 1: uv (vec2) offset 12
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.STRIDE, ctypes_void_p(12))
        # location 2: normal (vec3) offset 20
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.STRIDE, ctypes_void_p(20))

        # desliga o vao para nao alterar por engano depois
        glBindVertexArray(0)

    def draw(self, shader, wireframe: bool = False):
        # se ha textura, ativa a unidade 0 e binda
        if self.texture is not None:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            shader.set_int("u_tex", 0)
        # passa cor difusa e flag de wireframe para o shader
        shader.set_vec3("u_kd", *self.kd)
        shader.set_int("u_wireframe", 1 if wireframe else 0)
        # binda o vao e desenha como triangulos
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)


class Mesh:
    """Mesh completo (lista de GpuSubMesh)."""

    def __init__(self, submeshes: List[GpuSubMesh]):
        self.submeshes = submeshes

    @classmethod
    def from_obj(
        cls,
        obj_path: str,
        fallback_texture: Optional[int] = None,
        center_xz: bool = False,
        floor_y: bool = False,
        offset: Optional[tuple] = None,
    ) -> "Mesh":
        """Carrega .obj e cria Mesh em GPU.
        center_xz=True: desloca XZ para centralizar a malha em (0,?,0).
        floor_y=True: desloca Y para que ymin=0 (modelo "encostado no chão").
        offset=(dx, dy, dz): aplica um deslocamento fixo (subtraído dos vértices).
        """
        # le o arquivo .obj e quebra em submeshes (uma por material)
        sms = load_obj(obj_path)
        if not sms:
            raise RuntimeError(f"OBJ vazio: {obj_path}")
        
        # deslocamento que sera aplicado a todos os vertices
        cx, cy, cz = 0.0, 0.0, 0.0
        if center_xz or floor_y:
            # junta todos os vertices das submeshes para calcular bounding box
            all_pts = np.vstack([sm.vertices.reshape(-1, 8)[:, :3] for sm in sms])
            if center_xz:
                # acha o centro nos eixos x e z
                xmin, xmax = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
                zmin, zmax = float(all_pts[:, 2].min()), float(all_pts[:, 2].max())
                cx = (xmin + xmax) / 2
                cz = (zmin + zmax) / 2
            if floor_y:
                # encontra o y mais baixo para "encostar" o modelo no chao
                cy = float(all_pts[:, 1].min())
        
        # se o usuario passou um offset manual, ele substitui o calculo automatico
        if offset is not None:
            cx, cy, cz = offset

        # aplica o deslocamento subtraindo dos vertices originais
        if cx != 0.0 or cy != 0.0 or cz != 0.0:
            for sm in sms:
                v = sm.vertices.reshape(-1, 8)
                v[:, 0] -= cx
                v[:, 1] -= cy
                v[:, 2] -= cz
                sm.vertices = v.reshape(-1)
        
        # sobe cada submesh para a gpu
        gpu = [GpuSubMesh(sm, fallback_texture=fallback_texture) for sm in sms]
        return cls(gpu)

    def draw(self, shader, wireframe: bool = False):
        # desenha cada submesh em sequencia (cada uma pode ter textura/cor diferente)
        for sm in self.submeshes:
            sm.draw(shader, wireframe=wireframe)


def ctypes_void_p(offset: int):
    """Helper para passar ponteiros de offset para glVertexAttribPointer."""
    # opengl espera um ponteiro c, mas aqui passamos so um inteiro como deslocamento
    import ctypes
    return ctypes.c_void_p(offset)
