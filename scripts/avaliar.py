"""Mede a qualidade da GERAÇÃO, não só a perda.

O README do passo 2 mostra o problema: o modelo de 10 épocas tem perda de
validação pior e, ainda assim, gera respostas melhores. A perda mede previsão
token a token; ela não diz se a resposta tem o formato certo nem se o modelo
sabe parar. Este script gera uma resposta para cada pergunta da validação e
conta o que dá para contar:

  formato   resposta com os três campos, cada um uma única vez, na ordem certa
  parou     o modelo emitiu o EOS antes do limite de tokens (não entrou em loop)
  tokens    tamanho médio da resposta gerada
  F1        sobreposição de palavras com a resposta de referência — um proxy
            grosseiro de conteúdo, útil para comparar modelos entre si

E mais uma medida que nada tem a ver com o glossário:

  ppl geral perplexidade em data/exemplo/texto_geral.txt, um texto genérico em
            inglês. Se ela sobe muito depois do fine-tuning, o modelo esqueceu
            parte do pré-treino: é o catastrophic forgetting, em número.

Uso:
    python scripts/avaliar.py                          # base + passos 2 e 3
    python scripts/avaliar.py --modelo sft-lr5e-4      # inclui outra pasta de outputs/
    python scripts/avaliar.py --so-ppl                 # só a perplexidade, rápido
"""

import argparse
import json
import math
import re
from collections import Counter

import torch

from comum import (
    DIR_DADOS,
    DIR_SAIDA,
    ColetorSFT,
    DatasetSFT,
    carregar_exemplos,
    carregar_modelo,
    escolher_dispositivo,
    gerar,
    modelos_disponiveis,
)

CAMPOS = ["**Definição:**", "**Analogia:**", "**Erro comum:**"]
MAX_NOVOS = 256  # bem acima da maior resposta do dataset (~200 tokens)


def tem_formato(texto: str) -> bool:
    """Os três campos, cada um exatamente uma vez, na ordem do dataset."""
    if any(texto.count(c) != 1 for c in CAMPOS):
        return False
    posicoes = [texto.index(c) for c in CAMPOS]
    return posicoes == sorted(posicoes)


def palavras(texto: str) -> list[str]:
    texto = texto.lower()
    for c in CAMPOS:
        texto = texto.replace(c.lower(), " ")
    return re.findall(r"\w+", texto)


def f1_palavras(gerado: str, referencia: str) -> float:
    """F1 de sobreposição de palavras, como no SQuAD. Cego para ordem e sentido."""
    g, r = Counter(palavras(gerado)), Counter(palavras(referencia))
    comum = sum((g & r).values())
    if comum == 0:
        return 0.0
    precisao, revocacao = comum / sum(g.values()), comum / sum(r.values())
    return 2 * precisao * revocacao / (precisao + revocacao)


@torch.no_grad()
def perplexidade_geral(modelo, tokenizer) -> float:
    texto = (DIR_DADOS / "texto_geral.txt").read_text(encoding="utf-8")
    dispositivo = next(modelo.parameters()).device
    ids = tokenizer(texto, return_tensors="pt", add_special_tokens=False)["input_ids"]
    modelo.eval()
    perda = modelo(input_ids=ids.to(dispositivo), labels=ids.to(dispositivo)).loss
    return math.exp(perda.item())


@torch.no_grad()
def perda_validacao(modelo, tokenizer, exemplos: list[dict], batch: int = 4) -> float:
    """A mesma conta do eval_loss do Trainer: média das perdas de cada batch.

    Serve para o modelo base, que não tem historico.json, entrar na tabela com
    um número comparável aos dos modelos treinados.
    """
    ds, coletor = DatasetSFT(exemplos, tokenizer), ColetorSFT(tokenizer)
    dispositivo = next(modelo.parameters()).device
    modelo.eval()
    perdas = []
    for i in range(0, len(ds), batch):
        lote = {k: v.to(dispositivo) for k, v in coletor(ds.itens[i:i + batch]).items()}
        perdas.append(modelo(**lote).loss.item())
    return sum(perdas) / len(perdas)


def avaliar(modelo, tokenizer, exemplos: list[dict]) -> dict:
    linhas = []
    for e in exemplos:
        texto, info = gerar(modelo, tokenizer, e["instrucao"], MAX_NOVOS, detalhes=True)
        linhas.append({
            "instrucao": e["instrucao"],
            "gerado": texto,
            "formato": tem_formato(texto),
            "parou": info["parou"],
            "tokens": info["tokens"],
            "f1": f1_palavras(texto, e["resposta"]),
        })
    n = len(linhas)
    return {
        "formato": sum(l["formato"] for l in linhas) / n,
        "parou": sum(l["parou"] for l in linhas) / n,
        "tokens": sum(l["tokens"] for l in linhas) / n,
        "f1": sum(l["f1"] for l in linhas) / n,
        "respostas": linhas,
    }


def melhor_eval_loss(nome: str) -> float | None:
    historico = DIR_SAIDA / nome / "historico.json"
    if not historico.exists():
        return None
    log = json.loads(historico.read_text(encoding="utf-8"))["log_history"]
    perdas = [r["eval_loss"] for r in log if "eval_loss" in r]
    return min(perdas) if perdas else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", action="append", dest="modelos", default=[],
                        help="pasta extra em outputs/ (ou id do Hub) para incluir")
    parser.add_argument("--so-extras", action="store_true",
                        help="avalia só os --modelo passados, sem base/passo 2/passo 3")
    parser.add_argument("--so-ppl", action="store_true",
                        help="pula a geração e mede só a perplexidade geral")
    parser.add_argument("--saida", default="avaliacao.json",
                        help="arquivo em outputs/ onde os resultados são acumulados")
    args = parser.parse_args()

    dispositivo = escolher_dispositivo()
    exemplos = carregar_exemplos("validacao")
    alvos = modelos_disponiveis(args.modelos)
    if args.so_extras:
        alvos = alvos[-len(args.modelos):]

    # Acumula entre execuções: rodar de novo com --modelo X só atualiza X.
    arquivo = DIR_SAIDA / args.saida
    resultados = json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else {}

    for nome, origem in alvos:
        print(f"avaliando {nome}...", flush=True)
        modelo, tokenizer = carregar_modelo(origem, dispositivo)
        r = {} if args.so_ppl else avaliar(modelo, tokenizer, exemplos)
        r["ppl_geral"] = perplexidade_geral(modelo, tokenizer)
        r["eval_loss"] = melhor_eval_loss(nome) or perda_validacao(modelo, tokenizer, exemplos)
        resultados[nome] = {**resultados.get(nome, {}), **r}
        del modelo
        if dispositivo == "mps":
            torch.mps.empty_cache()

    arquivo.parent.mkdir(parents=True, exist_ok=True)
    arquivo.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(exemplos)} perguntas da validação, geração greedy, até {MAX_NOVOS} tokens\n")
    print(f"{'modelo':<16} | {'val loss':>8} | {'formato':>7} | {'parou':>6} | "
          f"{'tokens':>6} | {'F1':>5} | {'ppl geral':>9}")
    print("-" * 78)
    for nome, _ in alvos:
        r = resultados[nome]
        vl = f"{r['eval_loss']:.4f}" if r.get("eval_loss") else "-"
        if "formato" in r:
            print(f"{nome:<16} | {vl:>8} | {r['formato']:>7.0%} | {r['parou']:>6.0%} | "
                  f"{r['tokens']:>6.0f} | {r['f1']:>5.2f} | {r['ppl_geral']:>9.2f}")
        else:
            print(f"{nome:<16} | {vl:>8} | {'':>7} | {'':>6} | {'':>6} | {'':>5} | "
                  f"{r['ppl_geral']:>9.2f}")
    print(f"\nrespostas completas em {arquivo}")


if __name__ == "__main__":
    main()
