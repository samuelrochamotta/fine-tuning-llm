# fine-tuning-llm

Um projeto de estudo para entender fine-tuning de LLMs **por dentro**: do loop de
treino escrito à mão até LoRA, sempre em modelos pequenos o bastante para rodar
num notebook, sem GPU dedicada.

A ideia central é que fine-tuning não é uma técnica nova — é o mesmo loop de treino
de sempre, partindo de pesos já treinados, com learning rate menor. Quem escreve
esse loop uma vez nunca mais o vê como caixa-preta.

## Setup

Requer Python 3.10 ou superior.

```bash
uv venv --python 3.12 .venv          # ou: python3 -m venv .venv
uv pip install -r requirements.txt   # ou: .venv/bin/pip install -r requirements.txt
```

Os scripts detectam sozinhos o acelerador disponível: CUDA, MPS (Apple Silicon) ou CPU.

## Os três passos

### Passo 1 — escrever o loop, do zero

```bash
.venv/bin/python scripts/01_treino_do_zero.py
```

Treina um GPT de ~0,8M de parâmetros, em nível de caractere, sobre o texto de
Shakespeare (baixado na primeira execução). Não há framework escondendo nada:
attention causal, blocos transformer, o cálculo da perda e o loop de otimização
estão todos escritos em `scripts/01_treino_do_zero.py`.

O coração do arquivo são quatro linhas:

```python
_, perda = modelo(x, y)                 # 1. forward: previsão e perda
otimizador.zero_grad(set_to_none=True)  # 2. limpa gradientes do passo anterior
perda.backward()                        # 3. backward: gradiente de cada peso
otimizador.step()                       # 4. anda na direção que reduz a perda
```

Isso é todo o aprendizado profundo. Tudo nos passos 2 e 3 é esse mesmo loop, com
os pesos vindo prontos e o `Trainer` do Hugging Face escondendo as quatro linhas.

**O que observar:** a geração antes do treino é ruído puro; depois de 2000 passos
sai algo com estrutura de peça de teatro. E a perplexidade, impressa a cada
avaliação, é a leitura intuitiva da perda — "entre quantos caracteres o modelo
está hesitando a cada posição".

Resultado real (MPS, 76s para 2000 passos):

| Passo | Treino | Validação | Perplexidade |
|------:|-------:|----------:|-------------:|
| 0     | 4,32   | 4,32      | 75,3         |
| 250   | 2,49   | 2,51      | 12,3         |
| 2000  | 1,81   | 1,93      | 6,9          |

No passo 0 a perplexidade é ~75: com 65 símbolos no vocabulário, o modelo hesita
entre praticamente todos eles, que é exatamente o esperado de pesos aleatórios. No
fim, hesita entre ~7. Repare também que treino e validação são idênticos no passo
250 e já diferem em 0,12 no passo 2000 — é o overfitting nascendo, ainda inofensivo
aqui e bem mais agressivo no passo 2.

### Passo 2 — fine-tuning completo de um modelo pré-treinado

```bash
.venv/bin/python scripts/preparar_dados.py        # gera treino.jsonl e validacao.jsonl
.venv/bin/python scripts/02_finetune_completo.py
```

Pega o [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) e continua
o treino dele sobre 128 pares instrução/resposta de um glossário de machine learning
(`data/exemplo/glossario.json`). Todos os 135M de pesos são atualizados.

O script imprime as mesmas gerações **antes e depois** do treino — é essa comparação
que mostra o aprendizado acontecendo.

**O que observar:** o modelo passa a responder na estrutura de três campos do dataset
(`**Definição:** / **Analogia:** / **Erro comum:**`), inclusive para termos que nunca
viu. Ele aprendeu o **formato**, não fatos novos: 135M de parâmetros e 128 exemplos
ensinam comportamento, e o conteúdo sobre um termo desconhecido será inventado com
toda a confiança do mundo. Essa é a lição mais importante do passo 2 — fine-tuning é
uma forma cara e ruim de injetar conhecimento factual, que pertence a um sistema de
recuperação (RAG).

**Resultados reais** (Apple M-series, MPS, 113s de treino). Com os defaults do
script — 3 épocas, `lr=1e-4`:

| Época | Treino | Validação |
|------:|-------:|----------:|
| 1     | 2,7837 | 2,8363    |
| 2     | 2,3716 | 2,7732    |
| 3     | 2,2695 | **2,7712** |

Modelo base, para referência: 3,3485 de perda de validação.

E o que acontece esticando o treino para 10 épocas (`--epocas 10 --lr 1e-4`):

| Época | Treino | Validação |
|------:|-------:|----------:|
| 1     | 3,1494 | 2,9585    |
| 2     | 2,3726 | **2,7604** |
| 3     | 1,9502 | 2,8311    |
| 5     | 1,4041 | 3,1126    |
| 10    | 1,1252 | 3,2771    |

Esse é o gráfico de overfitting inteiro. A perda de treino cai de 3,15 para 1,13 e
parece ótimo; a de validação toca o fundo na época 2 e sobe daí em diante. Na época
10 o modelo é **pior em dados novos do que era na época 1**. Sem a coluna da direita,
você declararia vitória.

Vale registrar também o que NÃO funcionou: com `lr=2e-5` (o valor clássico para
datasets grandes) e 3 épocas, a validação parava em 3,2277 — praticamente o modelo
base intocado. Não é bug: 128 exemplos em batch de 4 dão só 32 passos por época, e
learning rate se calibra junto com o número de passos, nunca em abstrato.

**Um trade-off que aparece na prática:** o modelo de 10 épocas gera respostas mais
bem formadas, que sabem parar, apesar da perda de validação pior. A perda mede
predição token a token; ela não é métrica de qualidade de geração. Com 3 épocas o
modelo às vezes entra em loop porque ainda não aprendeu direito a emitir o EOS. Os
dois números medem coisas diferentes, e é preciso olhar os dois.

### Passo 3 — o mesmo resultado com LoRA

```bash
.venv/bin/python scripts/03_lora.py
```

Mesmo modelo, mesmo dado, mesmo loop. A diferença: os 135M de pesos ficam congelados
e treinam-se apenas matrizes de baixo posto acopladas às projeções `q_proj` e `v_proj`
— algo em torno de 0,5% dos parâmetros. O script imprime a contagem exata.

O artefato salvo é um **adaptador** de poucos MB, não um modelo inteiro. É a técnica
que se usa de verdade ao subir para modelos de 1B a 8B numa GPU só.

**Resultados reais** (8 épocas, `lr=2e-4`, rank 8, 85s de treino):

| | Passo 2 (completo) | Passo 3 (LoRA) |
|---|---:|---:|
| Parâmetros treinados | 134.515.008 (100%) | 460.800 (**0,34%**) |
| Melhor perda de validação | 2,7712 | 2,8420 |
| Arquivo de pesos salvo | 538 MB | **1,86 MB** |
| Tempo de treino | 113s | 85s |

É o argumento inteiro da técnica em quatro linhas: 0,34% dos parâmetros e 0,35% do
tamanho em disco, por 2,5% a mais de perda de validação. Em modelos de bilhões de
parâmetros essa troca é o que separa o que cabe numa GPU do que não cabe.

Note que o LoRA precisa de mais épocas e learning rate maior (8 e 2e-4, contra 3 e
1e-4): com menos parâmetros livres, cada passo muda menos, e a curva de validação
leva até a época 5 para achar o fundo.

### Comparar os três

```bash
.venv/bin/python scripts/comparar.py
.venv/bin/python scripts/comparar.py --prompt "O que é warmup?"
```

Roda os mesmos prompts no modelo base, no do passo 2 e no do passo 3, lado a lado,
com geração determinística (`do_sample=False`) — comparação com amostragem ligada
é loteria, não medição.

## Os dois conceitos que merecem atenção

**Máscara de loss.** O modelo deve ser cobrado pela resposta, não por reproduzir a
pergunta que já lhe foi entregue. Em `scripts/comum.py`, prompt e resposta são
tokenizados separadamente e os tokens do prompt recebem o rótulo `-100` — o valor
que a cross-entropy do PyTorch ignora por padrão. A biblioteca `trl`, com o
`SFTTrainer`, faz isso automaticamente; aqui está explícito de propósito.

**Validação.** 15% dos dados ficam de fora do treino, e o split é feito **por termo**,
não por exemplo: as quatro variações da mesma pergunta precisam cair todas do mesmo
lado, senão a validação vira um espelho do treino e mente para você. Sem esse número,
não há como distinguir "o modelo aprendeu" de "o modelo decorou".

## Experimentos que valem a pena

Cada um destes mostra um conceito quebrando na prática:

```bash
# overfitting: a perda de treino cai, a de validação sobe
.venv/bin/python scripts/02_finetune_completo.py --epocas 15

# catastrophic forgetting: learning rate alto demais apaga o pré-treino
.venv/bin/python scripts/02_finetune_completo.py --lr 5e-4

# capacidade do adaptador: mais rank, mais parâmetros treináveis
.venv/bin/python scripts/03_lora.py --rank 32
```

## Estrutura

```
data/exemplo/glossario.json    38 termos de ML com definição, analogia e erro comum
data/exemplo/*.jsonl           treino e validação, gerados por preparar_dados.py
scripts/01_treino_do_zero.py   GPT char-level do zero, loop escrito à mão
scripts/02_finetune_completo.py  SFT completo do SmolLM2-135M
scripts/03_lora.py             o mesmo, com LoRA/PEFT
scripts/comum.py               máscara de loss, collator, geração (compartilhados)
scripts/comparar.py            base vs. passo 2 vs. passo 3, lado a lado
scripts/preparar_dados.py      glossário -> treino/validação, split por termo
outputs/                       modelos treinados (fora do git)
```

## Trocar o dataset pelo seu

Substitua `data/exemplo/glossario.json` pelo seu domínio ou gere direto os arquivos
`data/exemplo/treino.jsonl` e `validacao.jsonl`, uma linha JSON por exemplo:

```json
{"instrucao": "...", "resposta": "...", "termo": "chave para o split"}
```

Nada mais precisa mudar. Se o formato de resposta for consistente, dá para ver o
efeito do fine-tuning com poucas centenas de exemplos.

## Créditos

O passo 1 segue de perto o [nanoGPT](https://github.com/karpathy/nanoGPT) e a aula
[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY), de Andrej Karpathy.
