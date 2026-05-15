"""Câmera FPS com WASD/setas + mouse, com clamping aos limites do mundo."""
import math

import numpy as np

from src import transforms as T


class FpsCamera:
    def __init__(
        self,
        position=(0.0, 5.0, 30.0),
        yaw_deg: float = -90.0,
        pitch_deg: float = 0.0,
        speed: float = 12.0,
        sensitivity: float = 0.12,
        bounds=None,  # ((xmin, xmax), (ymin, ymax), (zmin, zmax)) ou None
    ):
        # posicao da camera no mundo
        self.position = np.array(position, dtype=np.float32)
        # yaw = giro horizontal (esquerda/direita), pitch = giro vertical (cima/baixo)
        self.yaw = yaw_deg
        self.pitch = pitch_deg
        # velocidade de movimento (unidades por segundo)
        self.speed = speed
        # sensibilidade do mouse (quao rapido o olhar gira)
        self.sensitivity = sensitivity
        # limites opcionais para nao deixar a camera sair do mundo
        self.bounds = bounds
        # eixo "para cima" do mundo, fixo em +y
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # calcula os vetores front/right/up iniciais
        self._update_vectors()

    def _update_vectors(self):
        # converte yaw/pitch em vetores 3d normalizados
        cy = math.cos(math.radians(self.yaw))
        sy = math.sin(math.radians(self.yaw))
        cp = math.cos(math.radians(self.pitch))
        sp = math.sin(math.radians(self.pitch))
        # vetor "para frente" da camera baseado em yaw + pitch
        self.front = np.array([cy * cp, sp, sy * cp], dtype=np.float32)
        self.front /= np.linalg.norm(self.front)
        # "direita" sai do produto vetorial front x world_up
        self.right = np.cross(self.front, self.world_up)
        self.right /= np.linalg.norm(self.right)
        # "cima" da camera (perpendicular a front e right)
        self.up = np.cross(self.right, self.front)

    def view_matrix(self) -> np.ndarray:
        # gera a matriz view olhando da posicao para position+front
        return T.look_at(self.position, self.position + self.front, self.world_up)

    def process_mouse(self, dx: float, dy: float):
        # ajusta o angulo conforme o mouse se movimenta
        self.yaw += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        # limita o pitch para nao "virar de cabeca para baixo"
        self.pitch = max(-89.0, min(89.0, self.pitch))
        self._update_vectors()

    def process_keyboard(self, dt: float, forward: int, right: int, up: int):
        """forward/right/up são -1, 0 ou +1."""
        # movimento "horizontal" (sem componente Y) — movimento estilo FPS
        flat_front = np.array([self.front[0], 0.0, self.front[2]], dtype=np.float32)
        # normaliza para que a velocidade nao dependa do pitch (olhando pra baixo nao anda menos)
        n = np.linalg.norm(flat_front)
        if n > 1e-6:
            flat_front /= n
        # combina movimento para frente e para o lado, escalado pelo tempo (dt)
        delta = (flat_front * forward + self.right * right) * (self.speed * dt)
        # subir/descer e independente da direcao do olhar
        delta[1] += up * self.speed * dt
        self.position += delta
        # garante que a camera nao saia dos limites
        self._clamp()

    def _clamp(self):
        # se nao definimos limites, nao faz nada
        if self.bounds is None:
            return
        # restringe a posicao ao retangulo do mundo em cada eixo
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds
        self.position[0] = max(xmin, min(xmax, self.position[0]))
        self.position[1] = max(ymin, min(ymax, self.position[1]))
        self.position[2] = max(zmin, min(zmax, self.position[2]))
