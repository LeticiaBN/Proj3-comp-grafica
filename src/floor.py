"""Pisos texturizados gerados em runtime (sem .obj).

- TexturedDisk: disco circular horizontal com textura de arquivo.
- SolidColorDisk: disco circular horizontal com cor sólida.
- TiledFloorDisk: disco circular com textura procedural de azulejos
  (tileable). Gera uma "cara" de chão sci-fi sem precisar de asset.
- MarsFloorWithCircularHole: piso de Marte com um buraco circular no
  centro, alinhado com a parede da base. Garante que o chão externo e o
  interno nunca ocupem o mesmo espaço — sem z-fighting ou frestas
  visíveis nos cantos da base.
"""
import math
from typing import List, Tuple

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR,
    GL_NEAREST, GL_REPEAT, GL_RGB, GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T, GL_TRIANGLES, GL_UNSIGNED_BYTE, glActiveTexture,
    glBindBuffer, glBindTexture, glBindVertexArray, glBufferData, glDrawArrays,
    glEnableVertexAttribArray, glGenBuffers, glGenTextures, glGenVertexArrays,
    glGenerateMipmap, glTexImage2D, glTexParameteri, glVertexAttribPointer,
)

from src.mesh import ctypes_void_p
from src.texture import load_texture_2d
from src import transforms as T


_WHITE_TEX_CACHE: int = 0


def _white_pixel_texture() -> int:
    """Retorna (e cacheia) uma textura 1×1 branca. Usada com u_kd para
    pintar superfícies com cor sólida sem precisar mexer no shader."""
    global _WHITE_TEX_CACHE
    # se ja foi criada antes, reusa o mesmo id
    if _WHITE_TEX_CACHE:
        return _WHITE_TEX_CACHE
    # 1 pixel branco rgb (255,255,255)
    pixel = np.array([255, 255, 255], dtype=np.uint8)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    # cria a textura minima possivel (1x1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, pixel)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # nearest no min, linear no mag (suficiente para 1 pixel)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    _WHITE_TEX_CACHE = int(tex)
    return _WHITE_TEX_CACHE


def _generate_tile_floor_pattern(
    size: int = 512,
    tile_count: int = 4,
    base_color: Tuple[float, float, float] = (0.62, 0.63, 0.66),
    grout_color: Tuple[float, float, float] = (0.18, 0.18, 0.20),
    grout_width: int = 6,
    bevel_width: int = 4,
    noise_amp: float = 0.06,
    seed: int = 42,
) -> np.ndarray:
    """Gera uma textura tileable (size×size, RGB uint8) com aparência de
    chão de azulejos sci-fi:
      - tile_count × tile_count azulejos visíveis;
      - rejunte (grout) escuro entre os azulejos;
      - chanfro (bevel) sutil nas bordas, simulando o relevo da junta;
      - ruído leve dentro de cada azulejo, simulando rugosidade;
    Tileable: as bordas dos azulejos atravessam as bordas da imagem na
    metade do rejunte, então repetir lado-a-lado fica contínuo."""
    # gerador aleatorio deterministico (mesmo seed = mesma textura)
    rng = np.random.default_rng(seed)
    # comeca com a imagem inteira preenchida com a cor base
    img = np.empty((size, size, 3), dtype=np.float32)
    img[:] = base_color

    # Ruído fino por pixel (escala de cinza, multiplica o tom base)
    noise = (rng.random((size, size), dtype=np.float32) - 0.5) * 2.0 * noise_amp
    img += noise[..., None]

    # Coordenada dentro do azulejo: [0, tile_size)
    tile_size = size // tile_count
    # cria grades 2d dizendo "em que pixel do azulejo cada pixel da imagem cai"
    yy = np.arange(size).reshape(-1, 1) % tile_size
    xx = np.arange(size).reshape(1, -1) % tile_size

    # Distância (em px) à junta mais próxima — em x e em y separadamente
    half_tile = tile_size // 2
    dx = np.minimum(xx, tile_size - xx)
    dy = np.minimum(yy, tile_size - yy)
    d = np.minimum(dx, dy)  # distância à junta mais próxima

    # Máscara do rejunte (cor escura)
    grout_mask = d < grout_width
    img[grout_mask] = grout_color

    # Máscara do bevel: zona logo após o rejunte → escurece de forma
    # gradual para simular o relevo (escurece quanto mais perto da junta)
    bevel_zone = (~grout_mask) & (d < grout_width + bevel_width)
    if bevel_width > 0:
        # t vai de 0 (junto ao rejunte) a 1 (longe do rejunte)
        bevel_d = d[bevel_zone].astype(np.float32) - grout_width
        t = np.clip(bevel_d / float(bevel_width), 0.0, 1.0)
        # interpola entre escuro (0.7×base) e base
        darken = 0.70 + 0.30 * t
        img[bevel_zone] *= darken[:, None]

    # Variação de tom por azulejo: cada azulejo recebe um leve offset
    # multiplicativo (alguns azulejos um pouco mais claros/escuros).
    tile_idx_y = (np.arange(size) // tile_size).reshape(-1, 1)
    tile_idx_x = (np.arange(size) // tile_size).reshape(1, -1)
    rng2 = np.random.default_rng(seed + 1)
    per_tile = rng2.uniform(0.92, 1.08, size=(tile_count, tile_count)).astype(np.float32)
    tile_factor = per_tile[tile_idx_y, tile_idx_x]
    # Aplica só fora do rejunte (não desbalanceia a cor da junta)
    inside = ~grout_mask
    img[inside] *= tile_factor[inside, None]

    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def _upload_rgb_texture(data: np.ndarray) -> int:
    """Sobe um array (H, W, 3) uint8 como textura 2D (RGB, REPEAT, mipmaps)."""
    # tamanho extraido do shape do array
    h, w, _ = data.shape
    # garante memoria continua (requisito do opengl)
    data = np.ascontiguousarray(data)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    # envia os pixels gerados em python para a gpu
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
    glGenerateMipmap(GL_TEXTURE_2D)
    # repete (azulejos cobrem o piso inteiro), com filtro suave + mipmaps
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return int(tex)


def _make_disk_buffer(radius: float, segments: int, uv_repeat: float) -> np.ndarray:
    """Triangle fan a partir do centro formando um disco em XZ (Y=0).
    UVs mapeiam o quadrado circunscrito [-R, R] em [0, uv_repeat] (origem
    no centro). Winding CCW visto de cima (normal +Y)."""
    verts: List[float] = []
    R = radius
    s = uv_repeat

    # converte coordenadas do plano (x,z) em uv normalizado
    def uv(x: float, z: float) -> tuple:
        return (
            (x / R) * 0.5 * s + 0.5 * s,
            (z / R) * 0.5 * s + 0.5 * s,
        )

    # uv do centro do disco
    uc = (0.5 * s, 0.5 * s)
    # gera segments fatias de pizza ao redor do centro
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        # pontos da borda da fatia
        x0, z0 = R * math.cos(a0), R * math.sin(a0)
        x1, z1 = R * math.cos(a1), R * math.sin(a1)
        # Triângulo (centro, P_{i+1}, P_i) → CCW visto de +Y
        verts += [0.0, 0.0, 0.0,  uc[0], uc[1],         0.0, 1.0, 0.0]
        verts += [x1,  0.0, z1,   *uv(x1, z1),          0.0, 1.0, 0.0]
        verts += [x0,  0.0, z0,   *uv(x0, z0),          0.0, 1.0, 0.0]
    return np.array(verts, dtype=np.float32)


def _height(x: float, z: float, hole_r: float) -> float:
    """Deslocamento vertical Y para o terreno de Marte.

    Usa soma de senóides em diferentes frequências/fases para criar relevo
    orgânico (vento marciano). O blend radial garante que a área ao redor
    da base científica permaneça plana: Y=0 até hole_r+8 unidades do centro,
    crescendo suavemente ao longo de 20 unidades.
    """
    # distancia ao centro (para deixar a area da base plana)
    dist = math.sqrt(x * x + z * z)
    # blend cresce de 0 (perto da base) ate 1 (longe), suavizando o relevo
    blend = max(0.0, min(1.0, (dist - (hole_r + 8.0)) / 20.0))
    # soma de varias senoides com frequencias diferentes => relevo organico
    h = (
        3.5 * math.sin(x * 0.04 + 0.30) * math.cos(z * 0.05 + 1.10) +
        2.0 * math.sin(x * 0.09 - 0.70) * math.sin(z * 0.07 + 0.50) +
        1.2 * math.cos(x * 0.13 + 1.20) * math.cos(z * 0.11 - 0.80)
    )
    return h * blend


def _make_floor_with_circular_hole_buffer(
    world_half: float,
    hole_radius: float,
    segments: int = 64,
    radial_rings: int = 8,
) -> np.ndarray:
    """Gera o piso 2W x 2W com buraco circular de raio R no centro.

    Estratégia: 4 setores angulares de π/2, cada um subdividido em
    ``radial_rings`` anéis concêntricos. Cada anel tem vértices
    interpolados linearmente entre o círculo interno e a borda do mundo.
    Isso dá densidade suficiente para o deslocamento de altura _height()
    produzir um relevo orgânico visível.

    Winding CCW visto de cima (normal +Y).
    """
    W = world_half
    R = hole_radius
    # fator usado para mapear coordenada de mundo em uv normalizado
    inv = 1.0 / (2.0 * W)
    # quantos passos angulares por setor (4 setores totais)
    seg_per_side = max(1, segments // 4)

    # mapeia (x,z) do mundo para uv [0,1]
    def uv(x: float, z: float) -> tuple:
        return ((x + W) * inv, (z + W) * inv)

    def project_to_box(angle: float) -> tuple:
        """Interseção do raio (cos a, sin a) com o quadrado externo |·|=W."""
        cx, cz = math.cos(angle), math.sin(angle)
        ax, az = abs(cx), abs(cz)
        # parametro t para chegar na borda em x ou em z
        tx = W / ax if ax > 1e-9 else float("inf")
        tz = W / az if az > 1e-9 else float("inf")
        # o que estourar primeiro define o ponto na borda do quadrado
        t = min(tx, tz)
        return (t * cx, t * cz)

    def get_point(angle: float, t: float) -> tuple:
        """Ponto interpolado: t=0 → borda interna, t=1 → borda do mundo."""
        # ponto na borda interna do circulo
        xi = R * math.cos(angle)
        zi = R * math.sin(angle)
        # ponto correspondente na borda do mundo
        xo, zo = project_to_box(angle)
        # interpolacao linear entre borda interna e externa
        return (xi + t * (xo - xi), zi + t * (zo - zi))

    verts: List[float] = []
    # 4 setores cobrindo 360 graus, cada um com seu range angular
    for s in range(4):
        a_start = -math.pi / 4.0 + s * math.pi / 2.0
        # subdivide o setor em fatias angulares
        for i in range(seg_per_side):
            a0 = a_start + (i / seg_per_side) * (math.pi / 2.0)
            a1 = a_start + ((i + 1) / seg_per_side) * (math.pi / 2.0)
            # subdivide cada fatia em aneis radiais (quanto mais aneis, mais suave o relevo)
            for r in range(radial_rings):
                t0 = r / radial_rings
                t1 = (r + 1) / radial_rings
                # 4 cantos do quad: A=(a0,t0), B=(a1,t0), C=(a1,t1), D=(a0,t1)
                x_A, z_A = get_point(a0, t0)
                x_B, z_B = get_point(a1, t0)
                x_C, z_C = get_point(a1, t1)
                x_D, z_D = get_point(a0, t1)
                # altura calculada com a funcao senoidal
                y_A = _height(x_A, z_A, R)
                y_B = _height(x_B, z_B, R)
                y_C = _height(x_C, z_C, R)
                y_D = _height(x_D, z_D, R)
                # Tri 1: A → B → C  (CCW visto de +Y)
                verts += [x_A, y_A, z_A, *uv(x_A, z_A), 0.0, 1.0, 0.0]
                verts += [x_B, y_B, z_B, *uv(x_B, z_B), 0.0, 1.0, 0.0]
                verts += [x_C, y_C, z_C, *uv(x_C, z_C), 0.0, 1.0, 0.0]
                # Tri 2: A → C → D
                verts += [x_A, y_A, z_A, *uv(x_A, z_A), 0.0, 1.0, 0.0]
                verts += [x_C, y_C, z_C, *uv(x_C, z_C), 0.0, 1.0, 0.0]
                verts += [x_D, y_D, z_D, *uv(x_D, z_D), 0.0, 1.0, 0.0]
    return np.array(verts, dtype=np.float32)


class _StaticMesh2D:
    """Helper interno: encapsula VAO/VBO + draw para malhas 2D estáticas
    com layout (pos3, uv2, normal3). Pode ser pintada com uma textura
    real ou com cor sólida (textura branca 1x1 + u_kd)."""

    def __init__(
        self,
        buf: np.ndarray,
        texture_id: int,
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.texture = texture_id
        self.color = color
        # cada vertice ocupa 8 floats (pos3 + uv2 + normal3)
        self._vertex_count = len(buf) // 8
        # cria vao + vbo e envia os dados
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, buf.nbytes, buf, GL_STATIC_DRAW)
        # stride = 32 bytes (8 floats x 4 bytes cada)
        stride = 8 * 4
        # configura os 3 atributos: posicao, uv e normal
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes_void_p(20))
        glBindVertexArray(0)

    def _bind_and_draw(self, shader, wireframe: bool):
        # ativa textura na unidade 0 e seta uniforms basicos
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        shader.set_int("u_tex", 0)
        shader.set_vec3("u_kd", *self.color)
        shader.set_int("u_wireframe", 1 if wireframe else 0)
        # binda o vao e dispara o desenho
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self._vertex_count)
        glBindVertexArray(0)


class TexturedDisk(_StaticMesh2D):
    """Disco horizontal texturizado. Posição configurável (translação)."""

    def __init__(
        self,
        texture_path: str,
        radius: float,
        segments: int = 64,
        uv_repeat: float = 1.0,
    ):
        # cria a malha do disco e carrega a textura do arquivo
        super().__init__(
            _make_disk_buffer(radius, segments, uv_repeat),
            load_texture_2d(texture_path),
        )
        # posicao no mundo (pode ser alterada depois pelo usuario)
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def draw(self, shader, wireframe: bool = False):
        # so precisa de translacao (sem rotacao/escala)
        shader.set_mat4("u_model", T.translate(*self.position))
        self._bind_and_draw(shader, wireframe)


class SolidColorDisk(_StaticMesh2D):
    """Disco horizontal pintado com cor sólida (RGB em [0,1]).

    Implementação: bind de uma textura branca 1×1 + cor passada via
    u_kd. Aproveita o mesmo shader/pipeline do resto da cena.
    """

    def __init__(
        self,
        color: Tuple[float, float, float],
        radius: float,
        segments: int = 64,
    ):
        # disco geometrico + textura branca + cor solida via uniform
        super().__init__(
            _make_disk_buffer(radius, segments, uv_repeat=1.0),
            _white_pixel_texture(),
            color=color,
        )
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def draw(self, shader, wireframe: bool = False):
        shader.set_mat4("u_model", T.translate(*self.position))
        self._bind_and_draw(shader, wireframe)


class TiledFloorDisk(_StaticMesh2D):
    """Disco horizontal com textura procedural de azulejos sci-fi.

    A textura é gerada uma única vez no construtor (tileable, com rejunte
    + bevel + ruído + variação leve por azulejo). O parâmetro
    ``world_tiles_across_diameter`` controla quantos azulejos aparecem
    cruzando o disco — quanto maior, menores os azulejos no cenário.
    """

    def __init__(
        self,
        radius: float,
        segments: int = 64,
        tex_size: int = 512,
        tiles_per_texture: int = 4,
        world_tiles_across_diameter: float = 8.0,
        base_color: Tuple[float, float, float] = (0.62, 0.63, 0.66),
        grout_color: Tuple[float, float, float] = (0.18, 0.18, 0.20),
    ):
        # Quantas vezes a textura inteira repete cruzando o disco.
        # Se a textura tem N azulejos visíveis e queremos ver M no mundo,
        # a textura precisa repetir M/N vezes ao longo do diâmetro.
        uv_repeat = world_tiles_across_diameter / float(tiles_per_texture)
        # gera os pixels do azulejo proceduralmente (sem precisar de imagem)
        pattern = _generate_tile_floor_pattern(
            size=tex_size,
            tile_count=tiles_per_texture,
            base_color=base_color,
            grout_color=grout_color,
        )
        # malha do disco + textura procedural recem-gerada
        super().__init__(
            _make_disk_buffer(radius, segments, uv_repeat=uv_repeat),
            _upload_rgb_texture(pattern),
        )
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def draw(self, shader, wireframe: bool = False):
        shader.set_mat4("u_model", T.translate(*self.position))
        self._bind_and_draw(shader, wireframe)


class MarsFloorWithCircularHole(_StaticMesh2D):
    """Piso de Marte 2W x 2W com buraco circular centrado na origem e
    relevo senoidal nos vértices (longe da base o terreno ondula).

    A discretização usa ``radial_rings`` anéis concêntricos por segmento
    angular para que o deslocamento de altura seja suave e orgânico.
    """

    def __init__(
        self,
        texture_path: str,
        world_half: float,
        hole_radius: float,
        segments: int = 64,
        radial_rings: int = 8,
    ):
        # gera o piso com buraco e ja envia para a gpu junto com a textura de marte
        super().__init__(
            _make_floor_with_circular_hole_buffer(
                world_half, hole_radius, segments, radial_rings
            ),
            load_texture_2d(texture_path),
        )

    def draw(self, shader, wireframe: bool = False):
        # piso e fixo no mundo: matriz de modelo = identidade
        shader.set_mat4("u_model", T.identity())
        self._bind_and_draw(shader, wireframe)


def _make_mountain_cone(
    radius: float,
    height: float,
    segments: int = 32,
) -> np.ndarray:
    """Gera um cone (colina) em XZ centrado na origem, com base em Y=0
    e ápice em Y=height. Usado para as montanhas do terreno marciano.

    UVs mapeadas por projeção top-down (mesma lógica do piso), de forma
    que a textura de Marte apareça contínua entre o chão e a montanha.
    Winding CCW visto de fora (face lateral voltada para fora).
    """
    verts: List[float] = []
    R = radius
    H = height
    # UV: projeta o ponto (x, z) no quadrado [-R, R] → [0, 1]
    def uv(x: float, z: float) -> tuple:
        return ((x / R) * 0.5 + 0.5, (z / R) * 0.5 + 0.5)

    # quebra o cone em fatias triangulares ao redor do eixo y
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        # pontos da base do cone
        x0, z0 = R * math.cos(a0), R * math.sin(a0)
        x1, z1 = R * math.cos(a1), R * math.sin(a1)
        # Face lateral: ápice → P_{i+1} → P_i  (CCW visto de fora)
        verts += [0.0, H,  0.0,  *uv(0.0, 0.0),  0.0, 1.0, 0.0]
        verts += [x1,  0.0, z1,  *uv(x1, z1),    0.0, 1.0, 0.0]
        verts += [x0,  0.0, z0,  *uv(x0, z0),    0.0, 1.0, 0.0]
        # Tampa de baixo: (centro, P_i, P_{i+1})  → normal -Y
        verts += [0.0, 0.0, 0.0, *uv(0.0, 0.0),  0.0, -1.0, 0.0]
        verts += [x0,  0.0, z0,  *uv(x0, z0),    0.0, -1.0, 0.0]
        verts += [x1,  0.0, z1,  *uv(x1, z1),    0.0, -1.0, 0.0]
    return np.array(verts, dtype=np.float32)


class MarsMountain(_StaticMesh2D):
    """Colina/montanha cônica gerada proceduralmente com textura de Marte.

    Cada instância pode ser posicionada independentemente via ``position``.
    O parâmetro ``base_y`` deve ser igual ao deslocamento de altura do
    terreno no ponto de posicionamento (obtido via ``_height(x, z, R)``).
    """

    def __init__(
        self,
        texture_path: str,
        radius: float,
        height: float,
        segments: int = 32,
    ):
        # gera o cone e usa a mesma textura do chao para parecer continuo
        super().__init__(
            _make_mountain_cone(radius, height, segments),
            load_texture_2d(texture_path),
        )
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def draw(self, shader, wireframe: bool = False):
        # so translacao: cada montanha vai para sua posicao no mundo
        shader.set_mat4("u_model", T.translate(*self.position))
        self._bind_and_draw(shader, wireframe)
