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

### Passo 3 — o mesmo resultado com LoRA

```bash
.venv/bin/python scripts/03_lora.py
```

Mesmo modelo, mesmo dado, mesmo loop. A diferença: os 135M de pesos ficam congelados
e treinam-se apenas matrizes de baixo posto acopladas às projeções `q_proj` e `v_proj`
— algo em torno de 0,5% dos parâmetros. O script imprime a contagem exata.

O artefato salvo é um **adaptador** de poucos MB, não um modelo inteiro. É a técnica
que se usa de verdade ao subir para modelos de 1B a 8B numa GPU só.

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
