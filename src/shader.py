"""Wrapper para programa GLSL: compila VS+FS, linka, expõe uniforms."""
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_COMPILE_STATUS, GL_FRAGMENT_SHADER, GL_LINK_STATUS, GL_TRUE,
    GL_VERTEX_SHADER, glAttachShader, glCompileShader, glCreateProgram,
    glCreateShader, glDeleteShader, glGetProgramInfoLog, glGetProgramiv,
    glGetShaderInfoLog, glGetShaderiv, glGetUniformLocation, glLinkProgram,
    glShaderSource, glUniform1f, glUniform1i, glUniform3f,
    glUniformMatrix4fv, glUseProgram,
)


def _compile(stage, src, label):
    # cria um shader vazio do tipo certo (vertex ou fragment)
    sh = glCreateShader(stage)
    # envia o codigo fonte glsl para o shader
    glShaderSource(sh, src)
    # compila o codigo glsl
    glCompileShader(sh)
    # se a compilacao falhou, mostra a mensagem de erro
    if glGetShaderiv(sh, GL_COMPILE_STATUS) != GL_TRUE:
        log = glGetShaderInfoLog(sh).decode()
        raise RuntimeError(f"Erro ao compilar {label}:\n{log}")
    return sh


class Shader:
    def __init__(self, vs_src: str, fs_src: str):
        # compila os dois estagios separadamente
        vs = _compile(GL_VERTEX_SHADER, vs_src, "VS")
        fs = _compile(GL_FRAGMENT_SHADER, fs_src, "FS")
        # cria o programa (o "executavel" final na gpu)
        self.id = glCreateProgram()
        # junta vertex + fragment em um so programa
        glAttachShader(self.id, vs)
        glAttachShader(self.id, fs)
        # linka tudo (resolve referencias entre vs e fs)
        glLinkProgram(self.id)
        # verifica se a ligacao deu certo
        if glGetProgramiv(self.id, GL_LINK_STATUS) != GL_TRUE:
            log = glGetProgramInfoLog(self.id).decode()
            raise RuntimeError(f"Erro ao linkar shader:\n{log}")
        # apos o link os shaders individuais nao sao mais necessarios
        glDeleteShader(vs)
        glDeleteShader(fs)
        # cache de localizacoes de uniforms (evita chamar opengl varias vezes pelo mesmo nome)
        self._loc_cache = {}

    @classmethod
    def from_files(cls, vs_path, fs_path):
        # le os codigos glsl direto de arquivos no disco
        return cls(Path(vs_path).read_text(), Path(fs_path).read_text())

    def use(self):
        # ativa este shader como o programa atual da gpu
        glUseProgram(self.id)

    def loc(self, name: str) -> int:
        # busca a posicao de um uniform pelo nome, usando cache
        if name not in self._loc_cache:
            self._loc_cache[name] = glGetUniformLocation(self.id, name)
        return self._loc_cache[name]

    def set_mat4(self, name: str, mat: np.ndarray):
        # numpy é row-major; OpenGL espera column-major. transpose=GL_TRUE faz a troca.
        glUniformMatrix4fv(self.loc(name), 1, GL_TRUE, mat.astype(np.float32))

    def set_int(self, name: str, value: int):
        # envia um inteiro para um uniform (ex: indice de textura)
        glUniform1i(self.loc(name), int(value))

    def set_float(self, name: str, value: float):
        # envia um float para um uniform
        glUniform1f(self.loc(name), float(value))

    def set_vec3(self, name: str, x, y=None, z=None):
        # aceita 3 floats separados ou uma tupla/lista de tamanho 3
        if y is None:
            x, y, z = x[0], x[1], x[2]
        glUniform3f(self.loc(name), float(x), float(y), float(z))
