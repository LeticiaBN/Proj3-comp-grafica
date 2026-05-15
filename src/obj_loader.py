"""Parser de Wavefront .obj + .mtl com suporte a multi-material.

Saída: lista de "sub-meshes", cada uma com seu próprio buffer de vértices
intercalado [x,y,z, u,v, nx,ny,nz] e a textura diffuse correspondente.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SubMesh:
    """Faces de um mesmo material."""
    material: str
    vertices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    diffuse_texture: Optional[str] = None  # path absoluto para a textura
    kd: Tuple[float, float, float] = (1.0, 1.0, 1.0)


def _parse_mtl(mtl_path: Path) -> Dict[str, dict]:
    """Retorna {material_name: {'map_Kd': abs_path_or_None, 'Kd': (r,g,b)}}"""
    materials: Dict[str, dict] = {}
    # se o arquivo .mtl nao existe, devolve dict vazio (sem texturas)
    if not mtl_path.exists():
        return materials
    # pasta base usada para resolver caminhos relativos das texturas
    base = mtl_path.parent
    # material que esta sendo lido no momento (acumula propriedades ate o proximo "newmtl")
    current = None
    # le linha por linha o arquivo de materiais
    for line in mtl_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        # ignora linhas vazias e comentarios
        if not line or line.startswith("#"):
            continue
        # separa palavra-chave do resto da linha
        parts = line.split(maxsplit=1)
        kw = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        if kw == "newmtl":
            # comeca a definir um novo material
            current = rest.strip()
            materials[current] = {"map_Kd": None, "Kd": (1.0, 1.0, 1.0)}
        elif current is None:
            # ainda nao apareceu nenhum newmtl, ignora linhas soltas
            continue
        elif kw == "map_Kd":
            # pode ter flags antes do path; pegamos o último token
            tex_rel = rest.split()[-1].replace("\\", "/")
            # resolve para caminho absoluto a partir da pasta do .mtl
            tex_path = (base / tex_rel).resolve()
            # so guarda se o arquivo realmente existir no disco
            materials[current]["map_Kd"] = str(tex_path) if tex_path.exists() else None
        elif kw == "Kd":
            # cor difusa do material (rgb em [0,1])
            try:
                r, g, b = [float(x) for x in rest.split()[:3]]
                materials[current]["Kd"] = (r, g, b)
            except Exception:
                pass
    return materials


def load_obj(obj_path: str) -> List[SubMesh]:
    """Carrega .obj e retorna lista de SubMesh prontas para upload em VBO.
    Cada SubMesh tem vertices intercalados (3 pos + 2 uv + 3 normal = 8 floats)."""
    # caminho absoluto do .obj e a pasta dele
    obj_p = Path(obj_path).resolve()
    base = obj_p.parent

    # listas globais com todos os vertices, uvs e normais lidos do arquivo
    positions: List[Tuple[float, float, float]] = []
    uvs: List[Tuple[float, float]] = []
    normals: List[Tuple[float, float, float]] = []

    # buffers por material; chave None = sem material
    groups: Dict[str, List[float]] = {}
    current_mat = "__default__"
    groups[current_mat] = []

    # dicionario {nome_material: info} preenchido quando achamos um mtllib
    materials: Dict[str, dict] = {}

    # le o .obj linha por linha
    for raw in obj_p.read_text(errors="ignore").splitlines():
        line = raw.strip()
        # pula linhas vazias e comentarios
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        kw = parts[0]
        if kw == "mtllib":
            # referencia a um arquivo .mtl que descreve os materiais
            mtl_name = line.split(maxsplit=1)[1].strip()
            mtl_path = base / mtl_name
            materials.update(_parse_mtl(mtl_path))
        elif kw == "v":
            # vertice geometrico (x, y, z)
            positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif kw == "vt":
            # coordenada de textura (u, v)
            u = float(parts[1])
            v = float(parts[2]) if len(parts) > 2 else 0.0
            uvs.append((u, v))
        elif kw == "vn":
            # vetor normal (nx, ny, nz)
            normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif kw == "usemtl":
            # troca o material ativo: novas faces vao para outro grupo
            current_mat = parts[1] if len(parts) > 1 else "__default__"
            groups.setdefault(current_mat, [])
        elif kw == "f":
            # converte face (n-gon) em triângulos via fan
            verts = parts[1:]
            tri_indices = []
            # divide o poligono em triangulos: (v0, vi, vi+1)
            for i in range(1, len(verts) - 1):
                tri_indices.extend([verts[0], verts[i], verts[i + 1]])
            for vstr in tri_indices:
                # cada vertice de face vem no formato "pos/uv/normal"
                tokens = vstr.split("/")
                vi = int(tokens[0])
                ti = int(tokens[1]) if len(tokens) > 1 and tokens[1] else 0
                ni = int(tokens[2]) if len(tokens) > 2 and tokens[2] else 0

                # OBJ é 1-indexed; suporta índices negativos
                p = positions[vi - 1] if vi > 0 else positions[vi]
                # se nao tem uv, usa (0,0) como default
                if ti != 0:
                    t = uvs[ti - 1] if ti > 0 else uvs[ti]
                else:
                    t = (0.0, 0.0)
                # se nao tem normal, usa "para cima" como default
                if ni != 0:
                    n = normals[ni - 1] if ni > 0 else normals[ni]
                else:
                    n = (0.0, 1.0, 0.0)

                # adiciona o vertice intercalado [pos(3), uv(2), normal(3)] no grupo do material atual
                groups[current_mat].extend([p[0], p[1], p[2],
                                            t[0], t[1],
                                            n[0], n[1], n[2]])

    # gera SubMeshes (descarta grupos vazios)
    submeshes: List[SubMesh] = []
    for mat_name, buf in groups.items():
        # ignora grupos sem nenhuma face usada
        if not buf:
            continue
        # converte a lista python para um array numpy float32 (formato esperado pelo opengl)
        verts = np.asarray(buf, dtype=np.float32)
        info = materials.get(mat_name, {})
        sm = SubMesh(
            material=mat_name,
            vertices=verts,
            diffuse_texture=info.get("map_Kd"),
            kd=info.get("Kd", (1.0, 1.0, 1.0)),
        )
        submeshes.append(sm)

    return submeshes
