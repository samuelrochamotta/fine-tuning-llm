"""Usa o categorizador treinado no passo 4.

Recebe nomes de estabelecimento como aparecem na fatura, pela linha de comando
ou um por linha na entrada padrão, e devolve a categoria de cada um.

Uso:
    python scripts/categorizar.py "UBER *TRIP" "DROGARIA SAO PAULO"
    cat nomes.txt | python scripts/categorizar.py
    python scripts/categorizar.py --adaptador cat-360m "PADARIA REAL"
"""

import argparse
import json
import sys

from comum import DIR_SAIDA, carregar_modelo, escolher_dispositivo, gerar
from preparar_fatura import MOLDE_INSTRUCAO, limpar_estabelecimento


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nomes", nargs="*", help="se vazio, lê da entrada padrão")
    parser.add_argument("--adaptador", default="categorizador", help="pasta em outputs/")
    args = parser.parse_args()

    origem = DIR_SAIDA / args.adaptador
    if not (origem / "categorias.json").exists():
        raise SystemExit(f"{origem} não tem categorizador. Rode antes: "
                         f"python scripts/04_categorizar.py")
    categorias = json.loads((origem / "categorias.json").read_text(encoding="utf-8"))
    nomes = args.nomes or [linha.strip() for linha in sys.stdin if linha.strip()]

    modelo, tokenizer = carregar_modelo(origem, escolher_dispositivo())
    for nome in nomes:
        instrucao = MOLDE_INSTRUCAO.format(estabelecimento=limpar_estabelecimento(nome),
                                           categorias=", ".join(categorias))
        resposta = gerar(modelo, tokenizer, instrucao, max_novos=12).splitlines()
        categoria = resposta[0].strip().strip(".").lower() if resposta else ""
        # O modelo gera texto livre: nada garante que a resposta esteja na lista.
        aviso = "" if categoria in categorias else "  (fora da lista!)"
        print(f"{nome:<30} {categoria}{aviso}")


if __name__ == "__main__":
    main()
