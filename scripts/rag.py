"""RAG contra fine-tuning: onde deve morar o conhecimento factual.

A conclusão do passo 2 é que o fine-tuning ensinou o FORMATO, não os fatos: para
os termos da validação, que ele nunca viu, o modelo inventa o conteúdo. Este
script testa a alternativa: deixar os fatos num índice e entregá-los no prompt.

  recuperação  BM25 escrito à mão sobre os 38 verbetes do glossário (inclusive os
               6 da validação — o índice não é treino, é consulta).
  geração      o modelo base e o do passo 2, cada um com e sem o verbete
               recuperado no prompt.

A pergunta é o que cada ingrediente traz: o fine-tuning traz o formato de três
campos, o contexto traz o conteúdo. O recuperador é de brinquedo de propósito —
com 38 documentos e o termo escrito na própria pergunta, qualquer busca acerta.
O que interessa aqui é o lado do gerador.

Uso:
    python scripts/rag.py
    python scripts/rag.py --modelo lora      # avalia também outro modelo
"""

import argparse
import json
import math
import re
from collections import Counter

from avaliar import MAX_NOVOS, f1_palavras, tem_formato
from comum import (
    DIR_DADOS,
    DIR_SAIDA,
    MODELO_BASE,
    carregar_exemplos,
    carregar_modelo,
    escolher_dispositivo,
    gerar,
)


def tokens_busca(texto: str) -> list[str]:
    return re.findall(r"\w+", texto.lower())


class BM25:
    """O ranqueador clássico de busca textual, antes dos embeddings.

    Pontua um documento pela soma, sobre as palavras da consulta, de:
      IDF (palavras raras valem mais) × frequência no documento, saturada por k1
      e normalizada pelo tamanho do documento (b), para documentos longos não
      ganharem só por serem longos.
    """

    def __init__(self, documentos: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(tokens_busca(d)) for d in documentos]
        self.tamanhos = [sum(d.values()) for d in self.docs]
        self.media = sum(self.tamanhos) / len(self.docs)
        self.k1, self.b = k1, b
        n = len(self.docs)
        df = Counter(p for d in self.docs for p in d)
        self.idf = {p: math.log(1 + (n - f + 0.5) / (f + 0.5)) for p, f in df.items()}

    def buscar(self, consulta: str) -> int:
        def nota(i: int) -> float:
            d, tam = self.docs[i], self.tamanhos[i]
            return sum(
                self.idf.get(p, 0) * d[p] * (self.k1 + 1)
                / (d[p] + self.k1 * (1 - self.b + self.b * tam / self.media))
                for p in tokens_busca(consulta)
            )
        return max(range(len(self.docs)), key=nota)


def verbete(item: dict) -> str:
    # Texto corrido, SEM os marcadores **Definição:** etc. Se o documento já viesse
    # no formato da resposta, o modelo só precisaria copiar e o teste não diria
    # nada sobre formato.
    return (f"{item['termo']}: {item['definicao']} {item['analogia']} "
            f"Um erro comum: {item['erro_comum']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", action="append", dest="modelos", default=[],
                        help="pasta extra em outputs/ para avaliar com e sem RAG")
    args = parser.parse_args()

    glossario = json.loads((DIR_DADOS / "glossario.json").read_text(encoding="utf-8"))
    indice = BM25([verbete(i) for i in glossario])
    exemplos = carregar_exemplos("validacao")

    recuperados = [glossario[indice.buscar(e["instrucao"])] for e in exemplos]
    acertos = sum(r["termo"] == e["termo"] for r, e in zip(recuperados, exemplos))
    print(f"recuperação: {acertos} de {len(exemplos)} perguntas trouxeram o verbete certo")

    dispositivo = escolher_dispositivo()
    nomes = ["base", "sft-completo", *args.modelos]
    resultados = {}
    for nome in nomes:
        origem = DIR_SAIDA / nome if nome != "base" else MODELO_BASE
        if nome != "base" and not origem.is_dir():
            print(f"{origem} não existe, pulando")
            continue
        modelo, tokenizer = carregar_modelo(origem, dispositivo)
        for com_rag in (False, True):
            rotulo = f"{nome} + RAG" if com_rag else nome
            print(f"gerando: {rotulo}...", flush=True)
            linhas = []
            for e, r in zip(exemplos, recuperados):
                texto, info = gerar(modelo, tokenizer, e["instrucao"], MAX_NOVOS,
                                    detalhes=True,
                                    contexto=verbete(r) if com_rag else None)
                linhas.append({"instrucao": e["instrucao"], "gerado": texto,
                               "formato": tem_formato(texto), "parou": info["parou"],
                               "f1": f1_palavras(texto, e["resposta"])})
            n = len(linhas)
            resultados[rotulo] = {
                "formato": sum(l["formato"] for l in linhas) / n,
                "parou": sum(l["parou"] for l in linhas) / n,
                "f1": sum(l["f1"] for l in linhas) / n,
                "respostas": linhas,
            }

    arquivo = DIR_SAIDA / "rag.json"
    arquivo.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(exemplos)} perguntas sobre os {len({e['termo'] for e in exemplos})} "
          "termos da validação (nunca vistos no treino)\n")
    print(f"{'configuração':<22} | {'formato':>7} | {'parou':>6} | {'F1':>5}")
    print("-" * 50)
    for rotulo, r in resultados.items():
        print(f"{rotulo:<22} | {r['formato']:>7.0%} | {r['parou']:>6.0%} | {r['f1']:>5.2f}")

    exemplo = exemplos[0]["instrucao"]
    print(f"\nExemplo — {exemplo}")
    for rotulo, r in resultados.items():
        print(f"\n--- {rotulo} ---\n{r['respostas'][0]['gerado']}")
    print(f"\nrespostas completas em {arquivo}")


if __name__ == "__main__":
    main()
