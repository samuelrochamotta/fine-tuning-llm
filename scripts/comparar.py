"""Compara lado a lado o modelo base, o do passo 2 e o do passo 3.

Roda os mesmos prompts em cada modelo disponível em outputs/. É a forma honesta
de responder "o fine-tuning funcionou?": mesma pergunta, mesma configuração de
geração, saídas empilhadas.

Uso:
    python scripts/comparar.py
    python scripts/comparar.py --prompt "O que é warmup?"
"""

import argparse

from transformers import AutoModelForCausalLM

from comum import (
    DIR_SAIDA,
    MODELO_BASE,
    PROMPTS_DE_TESTE,
    carregar_tokenizer,
    escolher_dispositivo,
    gerar,
)


def carregar_modelos(dispositivo: str) -> list[tuple[str, object]]:
    modelos = [("base (sem treino)", AutoModelForCausalLM.from_pretrained(MODELO_BASE))]

    caminho_sft = DIR_SAIDA / "sft-completo"
    if (caminho_sft / "config.json").exists():
        modelos.append(
            ("passo 2 — fine-tuning completo",
             AutoModelForCausalLM.from_pretrained(caminho_sft))
        )

    caminho_lora = DIR_SAIDA / "lora"
    if (caminho_lora / "adapter_config.json").exists():
        from peft import PeftModel

        # O adaptador é carregado POR CIMA do modelo base — os pesos originais
        # continuam intactos no disco e na memória.
        base = AutoModelForCausalLM.from_pretrained(MODELO_BASE)
        modelos.append(("passo 3 — LoRA", PeftModel.from_pretrained(base, caminho_lora)))

    return [(nome, m.to(dispositivo)) for nome, m in modelos]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", action="append", dest="prompts")
    args = parser.parse_args()

    dispositivo = escolher_dispositivo()
    tokenizer = carregar_tokenizer()
    modelos = carregar_modelos(dispositivo)

    if len(modelos) == 1:
        print("Só o modelo base foi encontrado. Rode antes os passos 2 e/ou 3 "
              "para ter o que comparar.\n")

    for prompt in args.prompts or PROMPTS_DE_TESTE:
        print("=" * 74)
        print(f"PERGUNTA: {prompt}")
        print("=" * 74)
        for nome, modelo in modelos:
            print(f"\n--- {nome} ---")
            print(gerar(modelo, tokenizer, prompt))
        print()


if __name__ == "__main__":
    main()
