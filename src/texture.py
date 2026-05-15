"""Carrega texturas 2D e cubemaps."""
from pathlib import Path

from OpenGL.GL import (
    GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT, GL_RGB,
    GL_RGBA, GL_TEXTURE_2D, GL_TEXTURE_CUBE_MAP, GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_R,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_UNSIGNED_BYTE, glBindTexture,
    glGenTextures, glGenerateMipmap, glTexImage2D, glTexParameteri,
)
from PIL import Image


# cache para evitar recarregar a mesma textura repetidamente
_cache_2d = {}


def load_texture_2d(path: str) -> int:
    """Carrega uma textura 2D do disco. Retorna o id GL."""
    # path absoluto para usar como chave do cache
    p = str(Path(path).resolve())
    # se ja foi carregada antes, devolve o id ja existente
    if p in _cache_2d:
        return _cache_2d[p]

    # abre a imagem e garante que tem um modo de cor que o gl entende
    img = Image.open(p)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    # OpenGL espera origem no canto inferior esquerdo
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # converte para bytes brutos para enviar a gpu
    data = img.tobytes()

    # define se tem canal alpha (rgba) ou nao (rgb)
    fmt = GL_RGBA if img.mode == "RGBA" else GL_RGB

    # cria o objeto de textura na gpu e o bindea
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    # envia os pixels para a gpu
    glTexImage2D(GL_TEXTURE_2D, 0, fmt, img.width, img.height, 0, fmt, GL_UNSIGNED_BYTE, data)
    # gera mipmaps (versoes menores para evitar serrilhado a distancia)
    glGenerateMipmap(GL_TEXTURE_2D)

    # repete a textura quando uv sai do intervalo [0,1]
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # filtros lineares + mipmap dao um visual suave
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # guarda no cache para nao recarregar a mesma imagem
    _cache_2d[p] = tex_id
    return tex_id


def load_cubemap(face_paths: list) -> int:
    """Carrega cubemap. face_paths em ordem: +X, -X, +Y, -Y, +Z, -Z
    (right, left, top, bottom, front, back)."""
    # cria a textura tipo cubemap (6 faces de uma so vez)
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_id)

    # carrega cada uma das 6 faces na ordem certa
    for i, p in enumerate(face_paths):
        img = Image.open(p)
        # cubemap so aceita rgb (sem alpha)
        if img.mode != "RGB":
            img = img.convert("RGB")
        # cubemap NÃO precisa flip — convenção do GL é diferente do 2D
        data = img.tobytes()
        # GL_TEXTURE_CUBE_MAP_POSITIVE_X + i seleciona a face correta
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB,
                     img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)

    # filtro linear nas faces
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # clamp_to_edge evita linhas/costuras nas bordas entre as faces
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    return tex_id
