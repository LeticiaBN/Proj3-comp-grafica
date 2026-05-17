# Projeto 3 — Computação Gráfica (SCC0250)

João Pedro Alves Notari Godoy - 14582076
Letícia Barbosa Neves - 14588659

Cenário 3D **"Base Científica em Marte"**, em Python + PyOpenGL (pipeline moderno), com **iluminação Phong por fragmento** (ambiente + difusa + especular) e múltiplas fontes de luz coloridas.

## Como executar

```bash
python3 -m pip install --user --index-url https://pypi.org/simple/ -r requirements.txt
python3 main.py
```

## Iluminação

Modelo Phong por fragmento, com **3 luzes pontuais** + 1 luz ambiente global. Cada luz tem seu próprio interruptor (incluindo a ambiente).

| Luz | Posição | Cor | Visual | Afeta |
|---|---|---|---|---|
| **0 — Faróis do Rover** (externa, móvel) | acompanha o rover | branco-quente | **2 esferas pequenas** nos faróis dianteiros | apenas objetos externos |
| **1 — Lâmpadas de Teto** (interna A) | dentro da cúpula | branco-quente | **2 cilindros (tubos fluorescentes)** suspensos no teto | apenas objetos internos |
| **2 — Sabre de Luz** (interna B) | lâmina do sabre em cima da mesa | verde | **cilindro emissivo** sobreposto à lâmina do sabre | apenas objetos internos |
| **Ambiente** | global | leve azul | — | todos os objetos |

> Requisito 2 do PDF (duas fontes internas de **cores diferentes**) atendido: lâmpada de teto branca-quente + sabre verde.

O isolamento entre luzes externas/internas é feito por **máscara de bits no fragment shader**: cada objeto declara seu escopo (`outdoor` / `indoor` / `shared`) e o shader só aplica as luzes cujo bit está habilitado para aquele escopo.

Cada objeto tem **seus próprios parâmetros de reflexão difusa (k_d), especular (k_s) e expoente especular (Ns)** definidos em `src/scene.py` — nada vem do `.mtl` (req. 7).

## Controles

| Ação | Tecla |
|---|---|
| Mover câmera | `W` `A` `S` `D` |
| Subir / descer | `Espaço` / `Shift` esquerdo |
| Olhar em volta | Mouse |
| Mover **rover** (a luz externa o acompanha) | Setas `↑ ↓ ← →` |
| Rotacionar **satélite** (esq / dir) | `R` / `T` |
| Escalar **planeta** (diminui / cresce) | `-` / `=` (ou numpad `+`/`-`) |
| Toggle **wireframe** (malha poligonal) | `P` |
| Toggle **faróis do rover** (luz externa) | `1` |
| Toggle **lâmpadas de teto** (interna A) | `2` |
| Toggle **sabre de luz** (interna B) | `3` |
| Toggle **luz ambiente** | `4` |
| Luz ambiente: **diminuir / aumentar** | `Z` / `X` |
| Reflexão **difusa**: diminuir / aumentar | `C` / `V` |
| Reflexão **especular**: diminuir / aumentar | `B` / `N` |
| Sair | `ESC` |

## Estrutura

```
main.py                  # entry point — loop GLFW + input + envio dos uniforms de luz
requirements.txt
src/
  shader.py              # programa GLSL + helpers (mat3/4, vec3, int/vec3 arrays)
  obj_loader.py          # parser .obj/.mtl multi-material
  mesh.py                # VAO/VBO + sub-meshes por material (sem mais usar Kd do .mtl)
  texture.py             # carrega texturas 2D + cubemaps
  camera.py              # FPS-cam com clamping e adaptação ao relevo
  skybox.py              # cubemap skybox
  floor.py               # pisos e montanhas gerados em runtime (com normais corrigidas
                         #   para iluminação correta nas montanhas cônicas)
  entity.py              # mesh + transform + MATERIAL próprio (kd/ks/shininess)
                         #   + escopo de luz (outdoor/indoor/shared)
  procedural.py          # malhas emissivas (esfera + cilindro) usadas como
                         #   bulbos visuais das fontes de luz
  scene.py               # monta a cena + cria as 3 luzes pontuais + materiais por objeto
  transforms.py          # mat4 helpers numpy
shaders/
  basic.{vs,fs}          # Phong por fragmento: ambient + N pontuais + máscara de luz
  skybox.{vs,fs}         # skybox (sem iluminação)
objetos/                  # mesmos assets do Projeto 2 (.obj/.mtl/.png)
```

## Cena

### Externo

Superfície de Marte **200 × 200 m** com relevo senoidal. A câmera e o rover adaptam seu Y mínimo ao relevo.

| Objeto | Descrição |
|---|---|
| SkyBox | Cubemap de nebulosa/espaço (não recebe iluminação) |
| Chão de Marte | Piso texturizado com buraco circular sob a base e relevo orgânico |
| 4 Montanhas | Colinas cônicas processuais com **normais corrigidas** (sombreiam corretamente sob luz móvel) |
| ScienceBase | Cúpula circular modular — escopo "shared" (paredes vistas de dentro E de fora) |
| Connector, SphereBase | Módulos extras conectados à base |
| Nave | Orbita circular animada |
| **Rover** | **Translação por teclado; CARREGA O FAROL (luz externa)** |
| Satélite | Antena inclinada |
| Planeta | Júpiter ao fundo |
| Carrinho de carga, Caixas de suprimentos | Veículos / suprimentos próximos à base |
| **Faróis do rover** | Duas esferas amarelas pequenas nos faróis dianteiros (acompanham o rover) |

### Interno (dentro da ScienceBase)

Piso procedural de azulejos sci-fi.

| Objeto | Descrição |
|---|---|
| Cama composta | 5 partes (estrutura metálica + colchão fosco + tubulações) |
| Mesa de trabalho | Mesa sci-fi |
| Robô assistente | Chapas metálicas (alto brilho especular) |
| Caixa de armazenamento | Caixa fosca |
| Letreiro neon | Painel **emissivo** (brilha sempre) |
| Baby Yoda | Morador da estação |
| Garrafa | Vidro (especular alto + shininess alto) |
| Lixeira | Metal fosco |
| **Sabre de Luz** (Luke) | Em cima da mesa de trabalho — segunda fonte de luz interna (verde) |
| **Lâmpadas de teto** | Dois cilindros (tubos fluorescentes) suspensos dentro da cúpula |

## Notas de implementação

- **Modelo Phong por fragmento** (não por vértice) — saídas `v_world_pos` e `v_normal` interpolados do VS para o FS.
- **Matriz de normais** (`u_normal_mat`) calculada no CPU como `transpose(inverse(mat3(model)))` — necessária quando há escala não-uniforme.
- **Atenuação** quadrática por distância em cada luz pontual, para que a luz do rover destaque visualmente o que está próximo dele.
- **Especular só calculado se a face está virada para a luz** (`dot(N,L) > 0`) — evita brilho "vazando" pelo lado oposto.
- **Bulbos visíveis** somem quando a luz correspondente é desligada (feedback visual imediato).
- A **componente ambiente é independente** das luzes pontuais e tem seu próprio toggle (req. 3, 4).
