"""PASSO 1 — Treinar um GPT minúsculo do zero, em nível de caractere.

Objetivo: escrever o loop de treino com as próprias mãos, uma vez, para que
fine-tuning deixe de ser caixa-preta. O loop aqui é EXATAMENTE o mesmo dos passos
2 e 3; o que muda lá é só o ponto de partida (pesos treinados em vez de aleatórios),
o learning rate (menor) e o fato de a biblioteca esconder o loop de você.

O modelo tem ~0,8M de parâmetros e prevê o próximo CARACTERE do texto de
Shakespeare. Roda em CPU; em MPS/CUDA leva poucos minutos.

Uso:
    python scripts/01_treino_do_zero.py                # treino padrão (2000 passos)
    python scripts/01_treino_do_zero.py --passos 200   # só para ver rodando
"""

import argparse
import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

RAIZ = Path(__file__).resolve().parent.parent
CORPUS = RAIZ / "data" / "tinyshakespeare.txt"
URL_CORPUS = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# ---------------------------------------------------------------- hiperparâmetros
BLOCO = 128          # quantos caracteres de contexto o modelo enxerga
BATCH = 32           # quantas sequências por passo de treino
N_EMBD = 128         # dimensão dos embeddings
N_HEAD = 4           # cabeças de atenção (cada uma com N_EMBD // N_HEAD = 32 dims)
N_CAMADAS = 4        # blocos transformer empilhados
DROPOUT = 0.1
LEARNING_RATE = 3e-4
PASSOS_PADRAO = 2000
INTERVALO_AVALIACAO = 250
ITERS_AVALIACAO = 50  # batches usados para estimar cada loss (mais = menos ruído)


def escolher_dispositivo() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def baixar_corpus() -> str:
    if not CORPUS.exists():
        CORPUS.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando corpus de {URL_CORPUS} ...")
        urllib.request.urlretrieve(URL_CORPUS, CORPUS)
    return CORPUS.read_text(encoding="utf-8")


# --------------------------------------------------------------------- o modelo
class CabecaAtencao(nn.Module):
    """Uma cabeça de self-attention causal.

    Cada posição produz uma consulta (query), uma chave (key) e um valor (value).
    O peso que a posição i dá à posição j é o produto escalar entre a query de i e
    a key de j. A máscara triangular zera j > i: sem ela o modelo leria a resposta
    adiante, o treino convergiria lindamente e a geração seria lixo.
    """

    def __init__(self, tamanho_cabeca: int):
        super().__init__()
        self.key = nn.Linear(N_EMBD, tamanho_cabeca, bias=False)
        self.query = nn.Linear(N_EMBD, tamanho_cabeca, bias=False)
        self.value = nn.Linear(N_EMBD, tamanho_cabeca, bias=False)
        # buffer = tensor que acompanha o modelo (device, save) mas não é parâmetro
        self.register_buffer("mascara", torch.tril(torch.ones(BLOCO, BLOCO)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, cabeca)
        q = self.query(x)
        # A divisão por sqrt(dim) mantém a variância dos scores em ~1; sem ela o
        # softmax satura e os gradientes somem.
        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5     # (B, T, T)
        scores = scores.masked_fill(self.mascara[:T, :T] == 0, float("-inf"))
        pesos = self.dropout(F.softmax(scores, dim=-1))
        return pesos @ self.value(x)                                # (B, T, cabeca)


class MultiHead(nn.Module):
    """Várias cabeças em paralelo: cada uma aprende um tipo de relação diferente."""

    def __init__(self, n_cabecas: int, tamanho_cabeca: int):
        super().__init__()
        self.cabecas = nn.ModuleList(CabecaAtencao(tamanho_cabeca) for _ in range(n_cabecas))
        self.proj = nn.Linear(n_cabecas * tamanho_cabeca, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        saida = torch.cat([c(x) for c in self.cabecas], dim=-1)
        return self.dropout(self.proj(saida))


class FeedForward(nn.Module):
    """Processamento posição a posição. O fator 4 é a convenção desde o paper original."""

    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.rede(x)


class Bloco(nn.Module):
    """Comunicação (attention) seguida de processamento (feed-forward).

    Note o padrão `x = x + sublayer(norm(x))`: pré-normalização com conexão
    residual. O `+` dá ao gradiente um caminho direto até as camadas iniciais —
    é o que torna possível empilhar dezenas de blocos sem o treino travar.
    """

    def __init__(self):
        super().__init__()
        self.atencao = MultiHead(N_HEAD, N_EMBD // N_HEAD)
        self.ff = FeedForward()
        self.norm1 = nn.LayerNorm(N_EMBD)
        self.norm2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.atencao(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class GPTMinusculo(nn.Module):
    def __init__(self, tamanho_vocab: int):
        super().__init__()
        self.emb_token = nn.Embedding(tamanho_vocab, N_EMBD)
        # O transformer não tem noção de ordem por si só: a posição entra somada
        # ao embedding do token.
        self.emb_posicao = nn.Embedding(BLOCO, N_EMBD)
        self.blocos = nn.Sequential(*[Bloco() for _ in range(N_CAMADAS)])
        self.norm_final = nn.LayerNorm(N_EMBD)
        self.cabeca_lm = nn.Linear(N_EMBD, tamanho_vocab)

    def forward(self, idx, alvos=None):
        B, T = idx.shape
        x = self.emb_token(idx) + self.emb_posicao(torch.arange(T, device=idx.device))
        x = self.norm_final(self.blocos(x))
        logits = self.cabeca_lm(x)                       # (B, T, vocab)

        if alvos is None:
            return logits, None

        # cross_entropy espera (N, classes) e (N,), então achatamos batch e tempo.
        perda = F.cross_entropy(
            logits.view(B * T, -1),
            alvos.reshape(B * T),
        )
        return logits, perda

    @torch.no_grad()
    def gerar(self, idx, novos_tokens: int, temperatura: float = 1.0):
        for _ in range(novos_tokens):
            # o contexto é truncado ao bloco: o modelo não sabe olhar mais longe
            logits, _ = self(idx[:, -BLOCO:])
            logits = logits[:, -1, :] / temperatura   # só a última posição importa
            probs = F.softmax(logits, dim=-1)
            proximo = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, proximo), dim=1)
        return idx


# ---------------------------------------------------------------------- o treino
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passos", type=int, default=PASSOS_PADRAO)
    parser.add_argument("--semente", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.semente)
    dispositivo = escolher_dispositivo()
    print(f"dispositivo: {dispositivo}")

    texto = baixar_corpus()

    # Tokenizer em nível de caractere: o mais simples possível, para deixar claro
    # que tokenizer é só um mapa de ida e volta entre texto e inteiros.
    caracteres = sorted(set(texto))
    stoi = {c: i for i, c in enumerate(caracteres)}
    itos = dict(enumerate(caracteres))
    codificar = lambda s: [stoi[c] for c in s]
    decodificar = lambda ids: "".join(itos[i] for i in ids)
    print(f"corpus: {len(texto):,} caracteres | vocabulário: {len(caracteres)} símbolos")

    dados = torch.tensor(codificar(texto), dtype=torch.long)
    n = int(0.9 * len(dados))
    divisao = {"treino": dados[:n], "validacao": dados[n:]}

    def pegar_batch(nome: str):
        d = divisao[nome]
        # posições iniciais aleatórias; o alvo é a mesma janela deslocada em 1,
        # porque a tarefa é literalmente "preveja o próximo caractere"
        ix = torch.randint(len(d) - BLOCO - 1, (BATCH,))
        x = torch.stack([d[i:i + BLOCO] for i in ix])
        y = torch.stack([d[i + 1:i + BLOCO + 1] for i in ix])
        return x.to(dispositivo), y.to(dispositivo)

    modelo = GPTMinusculo(len(caracteres)).to(dispositivo)
    n_params = sum(p.numel() for p in modelo.parameters())
    print(f"parâmetros: {n_params/1e6:.2f}M\n")

    otimizador = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE)

    @torch.no_grad()
    def estimar_perdas():
        """Média sobre vários batches: uma medida de um batch só é ruído puro."""
        modelo.eval()   # desliga o dropout — sem isso a medida oscila sem razão
        resultado = {}
        for nome in divisao:
            perdas = torch.zeros(ITERS_AVALIACAO)
            for k in range(ITERS_AVALIACAO):
                x, y = pegar_batch(nome)
                _, perda = modelo(x, y)
                perdas[k] = perda.item()
            resultado[nome] = perdas.mean().item()
        modelo.train()
        return resultado

    print("Antes do treino (pesos aleatórios), o modelo gera isto:")
    contexto = torch.zeros((1, 1), dtype=torch.long, device=dispositivo)
    print("-" * 60)
    print(decodificar(modelo.gerar(contexto, 200)[0].tolist()))
    print("-" * 60, "\n")

    inicio = time.time()
    for passo in range(args.passos + 1):
        if passo % INTERVALO_AVALIACAO == 0 or passo == args.passos:
            p = estimar_perdas()
            # perplexidade = exp(loss): "entre quantos caracteres o modelo hesita"
            print(f"passo {passo:>5} | treino {p['treino']:.4f} | "
                  f"validação {p['validacao']:.4f} | ppl {math.exp(p['validacao']):.1f}")

        x, y = pegar_batch("treino")

        # ---- O LOOP. Estes quatro passos são todo o aprendizado profundo. ----
        _, perda = modelo(x, y)              # 1. forward: previsão e perda
        otimizador.zero_grad(set_to_none=True)  # 2. limpa gradientes do passo anterior
        perda.backward()                     # 3. backward: gradiente de cada peso
        otimizador.step()                    # 4. o otimizador anda na direção oposta
        # ---------------------------------------------------------------------

    print(f"\ntreino concluído em {time.time() - inicio:.1f}s")
    print("\nDepois do treino, com os mesmos pesos que começaram aleatórios:")
    print("-" * 60)
    print(decodificar(modelo.gerar(contexto, 500)[0].tolist()))
    print("-" * 60)
    print("\nNão é Shakespeare, mas virou inglês com estrutura de peça de teatro —")
    print("nomes em maiúsculas, dois pontos, falas. Isso saiu de ruído puro.")


if __name__ == "__main__":
    main()
