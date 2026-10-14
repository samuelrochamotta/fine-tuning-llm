"""PASSO 4 — Uma tarefa de verdade: categorizar transações do cartão com LoRA.

Os rótulos são os de data/fatura/rotulos.csv, não os do banco: veja
preparar_fatura.py.

O glossário dos passos 2 e 3 ensinava um FORMATO de texto longo. Aqui a saída é
uma palavra ("alimentação", "saúde"...), e isso muda tudo na avaliação: dá para
contar acertos. Acurácia é a métrica, não a perda.

E acurácia sozinha não diz nada sem uma régua. O script mede, na mesma validação:

  maioria   chuta sempre a categoria mais comum do treino
  vizinho   copia a categoria do estabelecimento de treino com nome mais parecido
            (trigramas de caracteres) — o "modelo" mais simples que aprende algo
  zero-shot o modelo de linguagem antes do treino, com a lista de categorias no prompt
  LoRA      o mesmo modelo depois do fine-tuning

O fine-tuning só se justifica se ganhar do vizinho. Se empatar, um dicionário
resolve o problema e o LLM é enfeite.

O treino roda um número FIXO de épocas, sem escolher o melhor checkpoint pela
validação: com ~45 exemplos de validação, usá-los para escolher o modelo e
também para medi-lo inflaria o número reportado.

Uso:
    python scripts/preparar_fatura.py ~/Downloads/Fatura_Itau_*.pdf
    python scripts/04_categorizar.py
    python scripts/04_categorizar.py --modelo HuggingFaceTB/SmolLM2-360M-Instruct
    python scripts/04_categorizar.py --dataset ficticios --saida categorizador-ficticios
"""

import argparse
import json
import math
from collections import Counter

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments

from comum import (
    DIR_SAIDA,
    MODELOS,
    RAIZ,
    ColetorSFT,
    DatasetSFT,
    carregar_exemplos,
    carregar_tokenizer,
    criar_treinador,
    escolher_dispositivo,
    gerar,
    passos_de_warmup,
    resumir_parametros,
    salvar_historico,
)
from preparar_fatura import CATEGORIAS

DIR_FATURA = RAIZ / "data" / "fatura"

# A categoria mais longa ("turismo e entretenimento") tem ~6 tokens. Folga
# suficiente para ver se o modelo para sozinho, sem deixá-lo divagar.
MAX_NOVOS = 12


def estabelecimento(exemplo: dict) -> str:
    """Recupera o nome da transação de dentro da instrução."""
    return exemplo["instrucao"].split(": ", 1)[1].split("\n", 1)[0]


def normalizar(texto: str) -> str:
    """Primeira linha, minúscula, sem pontuação no fim: 'Saúde.' conta como 'saúde'."""
    linhas = texto.strip().splitlines() or [""]
    return linhas[0].strip().strip(".").lower()


# ---------------------------------------------------------------- baselines

def trigramas(texto: str) -> Counter:
    t = f"  {texto.lower()} "
    return Counter(t[i:i + 3] for i in range(len(t) - 2))


def cosseno(a: Counter, b: Counter) -> float:
    produto = sum(a[k] * b[k] for k in a.keys() & b.keys())
    return produto / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def prever_vizinho(treino: list[dict], alvo: str) -> str:
    vetor = trigramas(alvo)
    melhor = max(treino, key=lambda e: cosseno(vetor, trigramas(estabelecimento(e))))
    return melhor["resposta"]


# ---------------------------------------------------------------- avaliação

def acuracia(previstos: list[str], exemplos: list[dict]) -> float:
    return sum(p == e["resposta"] for p, e in zip(previstos, exemplos)) / len(exemplos)


def prever_modelo(modelo, tokenizer, exemplos: list[dict]) -> list[str]:
    return [normalizar(gerar(modelo, tokenizer, e["instrucao"], MAX_NOVOS))
            for e in exemplos]


def relatorio_por_categoria(previstos: list[str], exemplos: list[dict]) -> None:
    total, certos = Counter(), Counter()
    for p, e in zip(previstos, exemplos):
        total[e["resposta"]] += 1
        certos[e["resposta"]] += p == e["resposta"]
    print(f"\n  {'categoria':<26} acertos")
    for cat, n in total.most_common():
        print(f"  {cat:<26} {certos[cat]:>2}/{n:<2}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default="instruct",
                        help="'base', 'instruct' (SmolLM2-135M) ou um id do Hub")
    parser.add_argument("--epocas", type=float, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alvos", choices=["qv", "todos"], default="todos")
    parser.add_argument("--dataset", default="",
                        help="'ficticios' soma estabelecimentos inventados ao treino; "
                             "a validação continua só com os reais")
    parser.add_argument("--saida", default="categorizador",
                        help="pasta dentro de outputs/")
    args = parser.parse_args()

    treino = carregar_exemplos("treino", args.dataset, pasta=DIR_FATURA)
    validacao = carregar_exemplos("validacao", args.dataset, pasta=DIR_FATURA)
    print(f"{len(treino)} estabelecimentos no treino, {len(validacao)} na validação")
    print(f"cada acerto na validação vale {1 / len(validacao):.1%}\n")

    resultados = {}
    maioria = Counter(e["resposta"] for e in treino).most_common(1)[0][0]
    resultados["maioria"] = acuracia([maioria] * len(validacao), validacao)
    previstos_vizinho = [prever_vizinho(treino, estabelecimento(e)) for e in validacao]
    resultados["vizinho"] = acuracia(previstos_vizinho, validacao)
    print(f"maioria ('{maioria}'): {resultados['maioria']:.1%}")
    print(f"vizinho mais parecido: {resultados['vizinho']:.1%}")

    dispositivo = escolher_dispositivo()
    nome_modelo = MODELOS.get(args.modelo, args.modelo)
    print(f"\ndispositivo: {dispositivo} | modelo: {nome_modelo}")
    tokenizer = carregar_tokenizer(nome_modelo)
    modelo = AutoModelForCausalLM.from_pretrained(nome_modelo).to(dispositivo)

    previstos_antes = prever_modelo(modelo, tokenizer, validacao)
    resultados["zero-shot"] = acuracia(previstos_antes, validacao)
    print(f"zero-shot, antes do treino: {resultados['zero-shot']:.1%}")
    print("  primeiras respostas:", [p[:30] for p in previstos_antes[:5]])

    # Mesma receita do passo 3, mas em todas as projeções: a tarefa pede que o
    # modelo associe nomes a categorias, e o MLP é onde fica essa "memória".
    modelo = get_peft_model(modelo, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=(["q_proj", "v_proj"] if args.alvos == "qv" else
                        ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]),
    ))
    resumir_parametros(modelo)

    ds_treino = DatasetSFT(treino, tokenizer)
    saida = DIR_SAIDA / args.saida
    treinador = criar_treinador(
        model=modelo,
        args=TrainingArguments(
            output_dir=str(saida),
            num_train_epochs=args.epocas,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            learning_rate=args.lr,
            warmup_steps=passos_de_warmup(len(ds_treino), args.batch, args.epocas),
            lr_scheduler_type="cosine",
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="no",
            report_to=[],
            seed=42,
        ),
        train_dataset=ds_treino,
        eval_dataset=DatasetSFT(validacao, tokenizer),
        data_collator=ColetorSFT(tokenizer),
        processing_class=tokenizer,
    )
    resultado = treinador.train()

    previstos_depois = prever_modelo(modelo, tokenizer, validacao)
    resultados["LoRA"] = acuracia(previstos_depois, validacao)

    print(f"\n{'=' * 50}\nACURÁCIA NA VALIDAÇÃO ({len(validacao)} estabelecimentos "
          f"nunca vistos)\n{'=' * 50}")
    for nome, valor in resultados.items():
        print(f"  {nome:<12} {valor:>6.1%}  {'#' * round(valor * 40)}")
    relatorio_por_categoria(previstos_depois, validacao)

    print("\nerros do LoRA (estabelecimento -> previsto | correto):")
    for p, e in zip(previstos_depois, validacao):
        if p != e["resposta"]:
            print(f"  {estabelecimento(e):<24} -> {p:<26} | {e['resposta']}")

    modelo.save_pretrained(saida)
    tokenizer.save_pretrained(saida)
    (saida / "categorias.json").write_text(
        json.dumps(CATEGORIAS, ensure_ascii=False), encoding="utf-8")
    salvar_historico(treinador, saida, args, {
        "tecnica": "lora",
        "modelo": nome_modelo,
        "acuracia": resultados,
        "tempo_treino_s": resultado.metrics["train_runtime"],
        "previsoes": [{"estabelecimento": estabelecimento(e), "correto": e["resposta"],
                       "antes": a, "depois": d, "vizinho": v}
                      for e, a, d, v in zip(validacao, previstos_antes,
                                            previstos_depois, previstos_vizinho)],
    })
    print(f"\nadaptador salvo em {saida} ({resultado.metrics['train_runtime']:.0f}s de treino)")
    print(f"teste: python scripts/categorizar.py \"UBER *TRIP\" \"DROGARIA SAO PAULO\"")


if __name__ == "__main__":
    main()
