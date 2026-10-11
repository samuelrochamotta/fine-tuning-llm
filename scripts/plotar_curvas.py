"""Desenha as curvas de treino e validação a partir de outputs/*/historico.json.

As tabelas do README são uma amostra; o gráfico mostra a curva inteira. É nele
que o overfitting fica óbvio: a perda de treino descendo sem parar enquanto a
de validação faz um "U".

Uso:
    python scripts/plotar_curvas.py                           # todos os runs
    python scripts/plotar_curvas.py sft-completo sft-ep15     # só estes
    python scripts/plotar_curvas.py --arquivo curvas.png
"""

import argparse
import json

from comum import DIR_SAIDA


def carregar_curvas(nome: str) -> dict:
    """Separa o log do Trainer em duas séries: treino (a cada N passos) e validação."""
    log = json.loads((DIR_SAIDA / nome / "historico.json").read_text(encoding="utf-8"))
    hist = log["log_history"]
    return {
        "nome": nome,
        "treino": [(r["epoch"], r["loss"]) for r in hist if "loss" in r],
        "validacao": [(r["epoch"], r["eval_loss"]) for r in hist if "eval_loss" in r],
        "args": log.get("args", {}),
    }


def runs_disponiveis() -> list[str]:
    return sorted(p.parent.name for p in DIR_SAIDA.glob("*/historico.json"))


def plotar(nomes: list[str], ax=None, base: float | None = None):
    """Uma cor por run; linha tracejada = treino, linha cheia com pontos = validação."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    for i, nome in enumerate(nomes):
        c = carregar_curvas(nome)
        cor = f"C{i}"
        if c["treino"]:
            ax.plot(*zip(*c["treino"]), "--", color=cor, alpha=0.6, label=f"{nome} — treino")
        if c["validacao"]:
            ax.plot(*zip(*c["validacao"]), "o-", color=cor, label=f"{nome} — validação")
            ep, melhor = min(c["validacao"], key=lambda t: t[1])
            ax.annotate(f"{melhor:.3f}", (ep, melhor), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8, color=cor)
    if base is not None:
        ax.axhline(base, color="gray", lw=1, ls=":", label=f"modelo base ({base:.2f})")
    ax.set_xlabel("época")
    ax.set_ylabel("perda (cross-entropy)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="pastas de outputs/ (padrão: todas)")
    parser.add_argument("--arquivo", default="curvas.png", help="PNG salvo em outputs/")
    parser.add_argument("--base", type=float, default=None,
                        help="perda de validação do modelo base, desenhada como referência")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")  # sem janela: só grava o arquivo
    import matplotlib.pyplot as plt

    nomes = args.runs or runs_disponiveis()
    if not nomes:
        raise SystemExit("nenhum outputs/*/historico.json encontrado: rode antes os passos 2 ou 3")
    ax = plotar(nomes, base=args.base)
    ax.set_title("Perda por época")
    destino = DIR_SAIDA / args.arquivo
    plt.tight_layout()
    plt.savefig(destino, dpi=120)
    print(f"gráfico salvo em {destino}")


if __name__ == "__main__":
    main()
