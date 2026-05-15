# Projeto 3 — Computação Gráfica (SCC0250)

João Pedro Alves Notari Godoy - 14582076
Letícia Barbosa Neves - 14588659

Cenário 3D **"Base Científica em Marte"**, em Python + PyOpenGL (pipeline moderno).

## Como executar

```bash
python3 -m pip install --user --index-url https://pypi.org/simple/ -r requirements.txt
python3 main.py
```

## Controles

| Ação | Tecla |
|---|---|
| Mover câmera | `W` `A` `S` `D` |
| Subir / descer | `Espaço` / `Shift` esquerdo |
| Olhar em volta | Mouse |
| **Translação do rover** | Setas `↑ ↓ ← →` |
| **Rotação do satélite** | `R` (esq) / `T` (dir) |
| **Escala do planeta** | `=` / `-` (ou `+` / `-` do numpad) |
| Toggle wireframe | `P` |
| Sair | `ESC` |

## Estrutura

```
main.py                  # entry point (loop GLFW + render + input)
requirements.txt         # dependências Python
src/
  shader.py              # programa GLSL (compilação + uniforms)
  obj_loader.py          # parser .obj/.mtl multi-material
  mesh.py                # VAO/VBO + sub-meshes por material
  texture.py             # carrega texturas 2D + cubemaps
  camera.py              # FPS-cam com clamping e adaptação ao relevo
  skybox.py              # cubemap skybox
  floor.py               # pisos e montanhas gerados em runtime:
                         #   MarsFloorWithCircularHole — chão externo com relevo senoidal
                         #   TiledFloorDisk — piso interno procedural (azulejos sci-fi)
                         #   MarsMountain — colinas cônicas com textura de Marte
  entity.py              # mesh + transform (posição / rotação Euler / escala)
  scene.py               # monta e atualiza a cena completa
  transforms.py          # mat4 helpers numpy (translate, rotate, scale, perspective…)
shaders/
  basic.{vs,fs}          # objetos texturizados
  skybox.{vs,fs}         # skybox
objetos/
  babyyoda/              # modelo Baby Yoda (.obj/.mtl/.png)
  mars/                  # textura e modelo do chão de Marte
  modularbase/           # ScienceBase, Connector e SphereBase (.obj/.mtl/.png)
  nave/                  # nave Attack Ship (.obj/.mtl + texturas)
  planet/                # planeta Júpiter (.obj/.mtl/.jpg)
  rover/                 # rover marciano (.obj/.mtl/.png)
  satelite/              # satélite/antena (.obj/.mtl/.png)
  scifi/                 # mobiliário interno sci-fi (.obj/.mtl/.png)
  skybox/                # faces do cubemap de nebulosa (6 × .png)
```

## Cena

### Externo

Superfície de Marte **200 × 200 m** com relevo senoidal nos vértices (terreno ondulado longe da base). A câmera e o rover adaptam seu Y mínimo ao relevo — nenhum dos dois afunda no chão.

| Objeto | Descrição |
|---|---|
| SkyBox | Cubemap de nebulosa/espaço |
| Chão de Marte | Piso texturizado com buraco circular sob a base e relevo orgânico (soma de senóides) |
| 4 Montanhas | Colinas cônicas processuais nos 4 quadrantes, com textura contínua do chão de Marte |
| ScienceBase | Cúpula circular modular (modelo principal) |
| Connector | Módulo de conexão à esquerda da base |
| SphereBase | Módulo esférico extra ao lado do conector |
| Nave | Orbita circular animada em torno da base |
| Rover | Translação por teclado (setas); gruda automaticamente no relevo |
| Satélite | Rotação por teclado (R/T); inclinado como antena real |
| Planeta | Planeta de fundo (textura Júpiter); escala por teclado |
| Carrinho de carga | Veículo de suprimentos próximo à área do rover |
| Caixas de suprimentos | Caixas empilhadas perto do conector |

### Interno (dentro da ScienceBase)

Piso procedural de **azulejos sci-fi** (tileable, com rejunte + chanfro + variação por azulejo, gerado em runtime sem asset externo).

| Objeto | Descrição |
|---|---|
| Cama composta | 5 partes separadas: estrutura, colchão, cabeceira, rodapé e tubulação |
| Mesa de trabalho | Mesa sci-fi voltada para o centro da base |
| Robô assistente | Virado para a entrada da base |
| Caixa de armazenamento | Storage box encostada na parede |
| Letreiro neon | Painel luminoso na parede lateral |
| Baby Yoda | Morador da estação |
| Garrafa | Ao lado da cama |
| Lixeira | Próxima à mesa de trabalho |

