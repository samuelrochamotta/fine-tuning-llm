"""PASSO 3 — O mesmo fine-tuning, agora com LoRA.

Mesmo modelo, mesmo dado, mesmo loop. A única diferença: em vez de atualizar os
135M de pesos, congelamos todos eles e treinamos pequenas matrizes de baixo posto
acopladas às projeções de atenção — algo em torno de 0,5% dos parâmetros.

É a técnica que se usa de verdade ao subir para modelos de 1B a 8B numa GPU só.
O resultado salvo é um ADAPTADOR de poucos MB, não um modelo inteiro.

Uso:
    python scripts/03_lora.py
    python scripts/03_lora.py --rank 32          # mais capacidade de ajuste
"""

import argparse

from peft import LoraConfig, TaskType, get_peft_model
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
    parser.add_argument("--epocas", type=float, default=8)
    # LoRA treina pouquíssimos parâmetros, então aguenta (e precisa de) um
    # learning rate uma ou duas ordens de grandeza maior que o do passo 2.
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    dispositivo = escolher_dispositivo()
    print(f"dispositivo: {dispositivo}")

    tokenizer = carregar_tokenizer()
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE).to(dispositivo)

    print("\nmodelo base, antes do LoRA:")
    resumir_parametros(modelo)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,                 # dimensão interna das matrizes adaptadoras
        # alpha/r é o que escala de fato o ajuste. Mantendo alpha = 2*r, dobrar o
        # rank não muda a escala do efeito, só a capacidade.
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        # As projeções de query e value são o alvo clássico: dão quase todo o
        # ganho pelo menor custo. Incluir k_proj/o_proj/MLP rende um pouco mais,
        # por mais parâmetros.
        target_modules=["q_proj", "v_proj"],
    )
    modelo = get_peft_model(modelo, config)

    print(f"\ncom LoRA (rank={args.rank}):")
    resumir_parametros(modelo)

    ds_treino = DatasetSFT(carregar_exemplos("treino"), tokenizer)
    ds_validacao = DatasetSFT(carregar_exemplos("validacao"), tokenizer)

    mostrar_amostras(modelo, tokenizer, "ANTES do treino LoRA (adaptador zerado = modelo base)")

    saida = DIR_SAIDA / "lora"
    treinador = Trainer(
        model=modelo,
        args=TrainingArguments(
            output_dir=str(saida),
            num_train_epochs=args.epocas,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            learning_rate=args.lr,
            warmup_steps=passos_de_warmup(len(ds_treino), args.batch, args.epocas),
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            seed=42,
        ),
        train_dataset=ds_treino,
        eval_dataset=ds_validacao,
        data_collator=ColetorSFT(tokenizer),
    )

    resultado = treinador.train()

    print("\nPerda de validação por época:")
    for reg in treinador.state.log_history:
        if "eval_loss" in reg:
            print(f"  época {reg['epoch']:>4.1f} | {reg['eval_loss']:.4f}")

    # Salva SÓ o adaptador, não o modelo inteiro.
    modelo.save_pretrained(saida)
    tokenizer.save_pretrained(saida)
    print(f"\nadaptador salvo em {saida}")

    peso_lora = saida / "adapter_model.safetensors"
    peso_sft = DIR_SAIDA / "sft-completo" / "model.safetensors"
    if peso_lora.exists():
        print(f"  adapter_model.safetensors: {peso_lora.stat().st_size / 1e6:.2f} MB")
    if peso_sft.exists():
        print(f"  contra o modelo completo do passo 2: "
              f"{peso_sft.stat().st_size / 1e6:.1f} MB")
    print(f"tempo de treino: {resultado.metrics['train_runtime']:.1f}s")

    mostrar_amostras(modelo, tokenizer, "DEPOIS do treino LoRA")
    print("Resultado parecido com o do passo 2, treinando uma fração dos pesos e")
    print("guardando um arquivo minúsculo. É por isso que LoRA é o padrão na prática.")


if __name__ == "__main__":
    main()
