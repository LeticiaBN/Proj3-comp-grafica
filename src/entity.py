"""Entity = Mesh + transform (posicao, rotacao Euler, escala) + material.

Para o Projeto 3 cada Entity carrega seus PROPRIOS parametros de
iluminacao difusa e especular (req. 7 — sem reutilizar parametros do
.mtl). A Entity tambem declara seu "escopo" (outdoor / indoor / shared),
o que define quais luzes podem afeta-la: a luz externa do rover so
ilumina objetos externos, e as duas luzes internas so iluminam objetos
internos (req. 1 e 2).
"""
import numpy as np

from src import transforms as T
from src.mesh import Mesh


# Mascaras de luz usadas pelo fragment shader.
# bit 0 = luz 0 (farol do rover, externa)
# bit 1 = luz 1 (lampada de teto, interna A)
# (Quando uma segunda luz interna for adicionada, criar bit 2 e refazer
#  LIGHT_MASK_INDOOR = 0b110 / LIGHT_MASK_SHARED = 0b111.)
LIGHT_MASK_OUTDOOR = 0b01  # so o farol do rover
LIGHT_MASK_INDOOR  = 0b10  # so a lampada de teto
LIGHT_MASK_SHARED  = 0b11  # paredes da base (vistas de dentro e de fora)


def _scope_to_mask(scope: str) -> int:
    """Converte o escopo declarativo em mascara de bits."""
    s = scope.lower()
    if s == "outdoor":
        return LIGHT_MASK_OUTDOOR
    if s == "indoor":
        return LIGHT_MASK_INDOOR
    if s == "shared":
        return LIGHT_MASK_SHARED
    raise ValueError(f"scope invalido: {scope!r}")


class Entity:
    def __init__(
        self,
        mesh: Mesh,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        disable_culling: bool = False,
        # ---- material proprio (req. 7) ----
        kd=(1.0, 1.0, 1.0),   # cor difusa (multiplica a textura)
        ks=(0.4, 0.4, 0.4),   # cor especular padrao (cinza medio)
        shininess: float = 32.0,
        # ---- iluminacao ----
        scope: str = "outdoor",  # "outdoor", "indoor" ou "shared"
        emissive: bool = False,  # True = objeto-fonte de luz (renderizado unlit)
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

        # ---- material proprio ----
        # Substitui o Kd lido do .mtl no projeto 2; ks/shininess sao novos.
        self.kd = np.array(kd, dtype=np.float32)
        self.ks = np.array(ks, dtype=np.float32)
        self.shininess = float(shininess)

        # ---- escopo de luz ----
        self.scope = scope
        self.light_mask = _scope_to_mask(scope)

        # objetos-fonte (lampadas) sao renderizados sem iluminacao (unlit)
        # para sempre brilharem com a propria cor — assim e visivel onde
        # cada fonte de luz esta posicionada na cena.
        self.emissive = emissive

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

    def normal_matrix(self) -> np.ndarray:
        """Matriz 3x3 para transformar normais.

        Para escalas uniformes, basta usar a rotacao de mat3(model).
        Como podemos ter escalas diferentes por eixo, calculamos
        transpose(inverse(mat3(model))) — formula correta.
        """
        m = self.model_matrix()[:3, :3]
        try:
            return np.linalg.inv(m).T.astype(np.float32)
        except np.linalg.LinAlgError:
            # fallback raro: matriz singular -> usa identidade
            return np.identity(3, dtype=np.float32)

    def draw(self, shader, wireframe: bool = False):
        # se invisivel, simplesmente nao desenha
        if not self.visible:
            return

        from OpenGL.GL import GL_CULL_FACE, glDisable, glEnable
        # desliga culling se a entidade pediu (ex: paredes vistas por dentro)
        if self.disable_culling:
            glDisable(GL_CULL_FACE)

        # envia as matrizes para o shader
        shader.set_mat4("u_model", self.model_matrix())
        shader.set_mat3("u_normal_mat", self.normal_matrix())

        # envia o material proprio do objeto (req. 7 — nao depende do .mtl)
        shader.set_vec3("u_kd", *self.kd)
        shader.set_vec3("u_ks", *self.ks)
        shader.set_float("u_shininess", self.shininess)

        # mascara de luzes (define se o objeto e externo, interno ou shared)
        shader.set_int("u_light_mask", self.light_mask)
        # objetos-fonte sao renderizados unlit (brilham com a propria cor)
        shader.set_int("u_unlit", 1 if self.emissive else 0)

        # manda a malha desenhar
        self.mesh.draw(shader, wireframe=wireframe)

        # restaura o culling para nao afetar as proximas entidades
        if self.disable_culling:
            glEnable(GL_CULL_FACE)
