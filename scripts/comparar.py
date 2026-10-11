"""Compara lado a lado o modelo base, o do passo 2 e o do passo 3.

Roda os mesmos prompts em cada modelo disponível em outputs/. É a forma honesta
de responder "o fine-tuning funcionou?": mesma pergunta, mesma configuração de
geração, saídas empilhadas.

Uso:
    python scripts/comparar.py
    python scripts/comparar.py --prompt "O que é warmup?"
    python scripts/comparar.py --modelo sft-lr5e-4 --modelo lora-todos
"""

import argparse

from comum import (
    PROMPTS_DE_TESTE,
    carregar_modelo,
    escolher_dispositivo,
    gerar,
    modelos_disponiveis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--modelo", action="append", dest="modelos", default=[],
                        help="pasta extra em outputs/ (ou id do Hub) para incluir")
    args = parser.parse_args()

    dispositivo = escolher_dispositivo()
    # Cada modelo vem com o PRÓPRIO tokenizer: o do instruct tem chat template e
    # outro EOS, e usar o tokenizer errado estraga a geração sem erro nenhum.
    modelos = [(nome, *carregar_modelo(origem, dispositivo))
               for nome, origem in modelos_disponiveis(args.modelos)]

    if len(modelos) == 1:
        print("Só o modelo base foi encontrado. Rode antes os passos 2 e/ou 3 "
              "para ter o que comparar.\n")

    for prompt in args.prompts or PROMPTS_DE_TESTE:
        print("=" * 74)
        print(f"PERGUNTA: {prompt}")
        print("=" * 74)
        for nome, modelo, tokenizer in modelos:
            print(f"\n--- {nome} ---")
            print(gerar(modelo, tokenizer, prompt))
        print()


if __name__ == "__main__":
    main()
