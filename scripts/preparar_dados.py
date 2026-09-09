"""Gera os arquivos de treino e validação a partir do glossário de exemplo.

Cada termo do glossário vira alguns exemplos de instrução -> resposta, variando a
forma da pergunta. Isso é augmentation simples: o conteúdo da resposta é o mesmo,
mas o modelo vê várias maneiras de pedir a mesma coisa e não decora uma frase só.

A resposta tem sempre a mesma estrutura de três campos. Essa rigidez é proposital:
é o que um modelo de 135M consegue de fato aprender com pouco dado, e é o que torna
o efeito do fine-tuning visível a olho nu na comparação antes/depois.

Uso:
    python scripts/preparar_dados.py
"""

import json
import random
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
GLOSSARIO = RAIZ / "data" / "exemplo" / "glossario.json"
SAIDA = RAIZ / "data" / "exemplo"

# Proporção separada para validação. Sem isso não há como distinguir
# "o modelo aprendeu" de "o modelo decorou".
FRACAO_VALIDACAO = 0.15
SEMENTE = 42

MOLDES_PERGUNTA = [
    "O que é {termo}?",
    "Explique o conceito de {termo}.",
    "O que significa {termo} em machine learning?",
    "Me explique {termo} e qual erro as pessoas costumam cometer.",
]


def formatar_resposta(item: dict) -> str:
    return (
        f"**Definição:** {item['definicao']}\n"
        f"**Analogia:** {item['analogia']}\n"
        f"**Erro comum:** {item['erro_comum']}"
    )


def main() -> None:
    glossario = json.loads(GLOSSARIO.read_text(encoding="utf-8"))

    exemplos = [
        {
            "instrucao": molde.format(termo=item["termo"]),
            "resposta": formatar_resposta(item),
            "termo": item["termo"],
        }
        for item in glossario
        for molde in MOLDES_PERGUNTA
    ]

    # O split é por TERMO, não por exemplo: as quatro variações de um mesmo termo
    # precisam cair todas do mesmo lado. Se um termo aparecesse nos dois conjuntos,
    # a validação estaria medindo memorização e mentiria para você.
    termos = sorted({item["termo"] for item in glossario})
    random.Random(SEMENTE).shuffle(termos)
    n_val = max(1, round(len(termos) * FRACAO_VALIDACAO))
    termos_validacao = set(termos[:n_val])

    treino = [e for e in exemplos if e["termo"] not in termos_validacao]
    validacao = [e for e in exemplos if e["termo"] in termos_validacao]

    for nome, dados in (("treino", treino), ("validacao", validacao)):
        caminho = SAIDA / f"{nome}.jsonl"
        with caminho.open("w", encoding="utf-8") as f:
            for exemplo in dados:
                f.write(json.dumps(exemplo, ensure_ascii=False) + "\n")
        print(f"{caminho.relative_to(RAIZ)}: {len(dados)} exemplos")

    print(f"\nTermos reservados para validação ({len(termos_validacao)}): "
          f"{', '.join(sorted(termos_validacao))}")


if __name__ == "__main__":
    main()
