"""PASSO 2 — Fine-tuning COMPLETO de um modelo pré-treinado (SmolLM2-135M).

Todos os pesos são atualizados. O loop de treino é o mesmo do passo 1, só que
escondido dentro do `Trainer` e partindo de pesos que já sabem português em vez
de ruído aleatório.

O script imprime as MESMAS gerações antes e depois do treino. É essa comparação
que mostra o aprendizado acontecendo.

Uso:
    python scripts/02_finetune_completo.py
    python scripts/02_finetune_completo.py --epocas 6 --lr 5e-4   # veja quebrar
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from comum import (
    DIR_SAIDA,
    MODELO_BASE,
    ColetorSFT,
    DatasetSFT,
    carregar_exemplos,
    carregar_tokenizer,
    escolher_dispositivo,
    passos_de_warmup,
    mostrar_amostras,
    resumir_parametros,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epocas", type=float, default=3)
    # 2e-5 é a ordem de grandeza certa para fine-tuning completo. O learning rate
    # do pré-treino (1e-3 ou mais) apagaria o que o modelo já sabe.
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    dispositivo = escolher_dispositivo()
    print(f"dispositivo: {dispositivo}")

    tokenizer = carregar_tokenizer()
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE).to(dispositivo)
    resumir_parametros(modelo)

    ds_treino = DatasetSFT(carregar_exemplos("treino"), tokenizer)
    ds_validacao = DatasetSFT(carregar_exemplos("validacao"), tokenizer)
    print(f"treino: {len(ds_treino)} exemplos | validação: {len(ds_validacao)} exemplos")

    # Vale olhar uma vez o que de fato entra no modelo: os tokens com label -100
    # são o prompt, ignorados na perda.
    amostra = ds_treino[0]
    n_mascarados = sum(1 for l in amostra["labels"] if l == -100)
    print(f"exemplo 0: {len(amostra['input_ids'])} tokens, "
          f"{n_mascarados} mascarados (o prompt), "
          f"{len(amostra['labels']) - n_mascarados} cobrados na loss")

    mostrar_amostras(modelo, tokenizer, "ANTES do fine-tuning (modelo base)")

    saida = DIR_SAIDA / "sft-completo"
    treinador = Trainer(
        model=modelo,
        args=TrainingArguments(
            output_dir=str(saida),
            num_train_epochs=args.epocas,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            learning_rate=args.lr,
            # Warmup: os primeiros passos, com estado do otimizador ainda ruim,
            # não podem sacudir pesos que custaram milhares de GPU-horas.
            warmup_steps=passos_de_warmup(len(ds_treino), args.batch, args.epocas),
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="epoch",   # a perda de validação é o que denuncia overfitting
            save_strategy="no",
            report_to=[],
            seed=42,
        ),
        train_dataset=ds_treino,
        eval_dataset=ds_validacao,
        data_collator=ColetorSFT(tokenizer),
    )

    resultado = treinador.train()

    print("\nHistórico de perdas por época:")
    print(f"{'época':>6} | {'treino':>8} | {'validação':>9}")
    perdas_treino = {}
    for reg in treinador.state.log_history:
        if "loss" in reg and "epoch" in reg:
            perdas_treino[round(reg["epoch"], 2)] = reg["loss"]
        if "eval_loss" in reg:
            ep = round(reg["epoch"], 2)
            proxima = min(perdas_treino, key=lambda e: abs(e - ep), default=None)
            tr = f"{perdas_treino[proxima]:.4f}" if proxima is not None else "-"
            print(f"{ep:>6.2f} | {tr:>8} | {reg['eval_loss']:>9.4f}")

    print("\nSe a coluna de validação parar de cair enquanto a de treino continua")
    print("caindo, o modelo passou de aprender para decorar: isso é overfitting.")

    modelo.save_pretrained(saida)
    tokenizer.save_pretrained(saida)
    print(f"\nmodelo salvo em {saida}")
    print(f"tempo de treino: {resultado.metrics['train_runtime']:.1f}s")

    mostrar_amostras(modelo, tokenizer, "DEPOIS do fine-tuning")
    print("Repare que o modelo passou a responder na estrutura de três campos do")
    print("dataset — inclusive para o termo que ele nunca viu. Ele aprendeu o FORMATO.")
    print("O conteúdo sobre um termo desconhecido será inventado: 135M de parâmetros")
    print("e 128 exemplos ensinam comportamento, não conhecimento novo.")


if __name__ == "__main__":
    main()
