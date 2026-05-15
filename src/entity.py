"""Entity = Mesh + transform (posição, rotação Euler, escala)."""
import numpy as np

from src import transforms as T
from src.mesh import Mesh


class Entity:
    def __init__(
        self,
        mesh: Mesh,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        disable_culling: bool = False,
    ):
        # geometria reutilizada (mesma malha pode aparecer em varias entidades)
        self.mesh = mesh
        # posicao no mundo
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)  # rad em (x,y,z)
        # escala em cada eixo (1,1,1) = tamanho original
        self.scale = np.array(scale, dtype=np.float32)
        # se True, render desabilita backface culling para esta entidade
        # (usado para o interior da base)
        self.disable_culling = disable_culling
        # flag para esconder a entidade sem precisar removela da lista
        self.visible = True

    def model_matrix(self) -> np.ndarray:
        # gera as 5 matrizes basicas do transform
        t = T.translate(*self.position)
        rx = T.rotate_x(self.rotation[0])
        ry = T.rotate_y(self.rotation[1])
        rz = T.rotate_z(self.rotation[2])
        s = T.scale(*self.scale)
        # ordem: T * Ry * Rx * Rz * S
        # (a leitura e da direita pra esquerda: primeiro escala, depois rotaciona, depois translada)
        return t @ ry @ rx @ rz @ s

    def draw(self, shader, wireframe: bool = False):
        # se invisivel, simplesmente nao desenha
        if not self.visible:
            return
        
        from OpenGL.GL import GL_CULL_FACE, glDisable, glEnable
        # desliga culling se a entidade pediu (ex: paredes vistas por dentro)
        if self.disable_culling:
            glDisable(GL_CULL_FACE)
            
        # envia a matriz de modelo para o shader e manda a malha desenhar
        shader.set_mat4("u_model", self.model_matrix())
        self.mesh.draw(shader, wireframe=wireframe)
        
        # restaura o culling para nao afetar as proximas entidades
        if self.disable_culling:
            glEnable(GL_CULL_FACE)
