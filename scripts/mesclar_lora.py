"""Funde o adaptador LoRA nos pesos do modelo e prova que nada mudou.

LoRA aprende W + (alpha/r)·B·A sem tocar em W. Na inferência, dá para fazer essa
soma uma vez e jogar o adaptador fora: o resultado é um modelo comum, do mesmo
tamanho do original, sem a camada extra de cálculo por token.

O script compara os logits do modelo com adaptador e do modelo fundido para os
mesmos prompts. Se a técnica é o que diz ser, a diferença é só arredondamento
de ponto flutuante — e a geração greedy sai idêntica.

Uso:
    python scripts/mesclar_lora.py                     # funde outputs/lora
    python scripts/mesclar_lora.py --adaptador lora-todos
"""

import argparse

import torch
from transformers import AutoModelForCausalLM

from comum import (
    DIR_SAIDA,
    PROMPTS_DE_TESTE,
    carregar_modelo,
    escolher_dispositivo,
    gerar,
    montar_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptador", default="lora", help="pasta em outputs/")
    args = parser.parse_args()

    dispositivo = escolher_dispositivo()
    origem = DIR_SAIDA / args.adaptador
    if not (origem / "adapter_config.json").exists():
        raise SystemExit(f"{origem} não tem adaptador. Rode antes: python scripts/03_lora.py")

    com_adaptador, tokenizer = carregar_modelo(origem, dispositivo)
    com_adaptador.eval()

    # Logits ANTES do merge, guardados para comparar depois. merge_and_unload
    # altera o modelo no lugar, então não dá para calcular os dois em paralelo.
    entradas = [
        tokenizer(montar_prompt(p, tokenizer), return_tensors="pt",
                  add_special_tokens=False).to(dispositivo)
        for p in PROMPTS_DE_TESTE
    ]
    with torch.no_grad():
        logits_antes = [com_adaptador(**e).logits for e in entradas]
    textos_antes = [gerar(com_adaptador, tokenizer, p) for p in PROMPTS_DE_TESTE]

    fundido = com_adaptador.merge_and_unload()
    destino = DIR_SAIDA / f"{args.adaptador}-fundido"
    fundido.save_pretrained(destino)
    tokenizer.save_pretrained(destino)

    # Recarrega do disco como um modelo comum, sem peft: é assim que ele seria
    # usado em produção.
    recarregado = AutoModelForCausalLM.from_pretrained(destino).to(dispositivo).eval()
    with torch.no_grad():
        logits_depois = [recarregado(**e).logits for e in entradas]
    textos_depois = [gerar(recarregado, tokenizer, p) for p in PROMPTS_DE_TESTE]

    maior_dif = max((a - d).abs().max().item() for a, d in zip(logits_antes, logits_depois))
    escala = max(a.abs().max().item() for a in logits_antes)
    iguais = sum(a == d for a, d in zip(textos_antes, textos_depois))

    tam_adaptador = (origem / "adapter_model.safetensors").stat().st_size / 1e6
    tam_fundido = (destino / "model.safetensors").stat().st_size / 1e6

    print(f"\nmodelo fundido salvo em {destino}")
    print(f"  adaptador: {tam_adaptador:.2f} MB  ->  modelo fundido: {tam_fundido:.1f} MB")
    print(f"maior diferença absoluta entre logits: {maior_dif:.2e} "
          f"(logits chegam a {escala:.1f}, diferença relativa {maior_dif / escala:.1e})")
    print(f"gerações greedy idênticas: {iguais} de {len(PROMPTS_DE_TESTE)}")
    if iguais < len(PROMPTS_DE_TESTE):
        for p, a, d in zip(PROMPTS_DE_TESTE, textos_antes, textos_depois):
            if a != d:
                print(f"\n>>> {p}\n--- com adaptador ---\n{a}\n--- fundido ---\n{d}")
    print("\nO adaptador é o formato para guardar e trocar (vários por modelo base);")
    print("o modelo fundido é o formato para servir (um só, sem custo extra por token).")


if __name__ == "__main__":
    main()
