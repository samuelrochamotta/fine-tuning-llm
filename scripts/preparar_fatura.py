"""Transforma faturas do cartão Itaú (PDF) em dataset de categorização.

A fatura do Itaú traz uma categoria em cada lançamento, mas ela é IGNORADA de
propósito: o modelo deve aprender a SUA taxonomia, não a do banco. Os rótulos
vêm de data/fatura/rotulos.csv, uma linha por estabelecimento:

    estabelecimento,categoria,confianca
    IFD*PADARIA CENTRAL,restaurante,alta
    MP *FULANO,,          <- sem categoria: fica fora do dataset

O fluxo é:
  1. este script lê os PDFs e acrescenta ao CSV, com categoria em branco, todo
     estabelecimento que ainda não estiver lá;
  2. alguém preenche as categorias (à mão, ou um modelo grande como professor,
     com revisão humana — linhas de confiança "baixa" primeiro);
  3. roda de novo: só as linhas com categoria entram no dataset.

Uma fatura só rende poucas dezenas de exemplos por categoria. Por isso também é
gerada a versão _ficticios: o treino real somado aos estabelecimentos inventados
de data/exemplo/estabelecimentos_ficticios.csv, com a mesma validação real.

A página tem duas colunas lado a lado. Extrair o texto da página inteira mistura
as duas linha a linha, então cada coluna é recortada e lida separadamente.

PRIVACIDADE: só o nome do estabelecimento é extraído, para data/fatura/, pasta
que o .gitignore deixa fora do repositório. Nome do titular, endereço, número do
cartão, datas e valores não saem do PDF.

Uso:
    python scripts/preparar_fatura.py ~/Downloads/Fatura_Itau_*.pdf
    python scripts/preparar_fatura.py      # só refaz o split a partir do CSV
"""

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "data" / "fatura"
ROTULOS = SAIDA / "rotulos.csv"
# Nomes inventados, sem dado pessoal nenhum: por isso ficam versionados.
FICTICIOS = RAIZ / "data" / "exemplo" / "estabelecimentos_ficticios.csv"

CATEGORIAS = [
    "mercado",       # supermercado, mercadinho, padaria, açougue, peixaria
    "restaurante",   # comer fora e delivery: restaurante, lanchonete, café, iFood
    "saúde",         # farmácia, clínica, laboratório, ótica
    "transporte",    # Uber, combustível, estacionamento, pedágio, seguro do carro
    "roupas",        # roupa, calçado, lingerie, acessórios
    "beleza",        # salão, cosméticos, perfumaria
    "casa",          # móveis, utilidades, decoração, condomínio, eletrônicos
    "assinaturas",   # apps e serviços recorrentes: streaming, nuvem, software
    "viagem",        # hotel, passagem, locadora, duty free
    "lazer",         # esporte, hobby, ingressos, brinquedos, pet
    "educação",      # cursos, escola, livros
]
# Não há "outros" de propósito: pessoa física, marketplace genérico e nomes
# opacos ficam SEM rótulo no CSV e fora do dataset. Uma gaveta de "não sei"
# seria a classe mais frequente e o modelo aprenderia a responder isso.

# Coordenada x (em pontos) que separa as duas colunas de lançamentos. Medida na
# fatura: a coluna da esquerda termina em x≈340 (valores) e a da direita começa
# em x≈367 (datas). Se o layout do Itaú mudar, é aqui que quebra.
DIVISA_COLUNAS = 355

# Mais validação que no glossário (15%): aqui cada estabelecimento vira um único
# exemplo, então 15% de ~180 deixaria pouco mais de 25 casos para medir.
FRACAO_VALIDACAO = 0.25
SEMENTE = 42

# Estornos vêm com o sinal separado do número: "LOJAS RIACHUELO SA - 0,01".
RE_LANCAMENTO = re.compile(r"^(\d{2}/\d{2}) (.+?) (-\s?)?[\d.]+,\d{2}$")
# Parcela no fim do nome: "LOJAS RIACHUELO SA01/03", "ITAUSHOP 06/10".
RE_PARCELA = re.compile(r"\s*\d{2}/\d{2}$")
# Prefixos de adquirente/marketplace: "IFD*" é iFood, "MP*" Mercado Pago etc.
RE_PREFIXO = re.compile(r"^[A-Z0-9 ]{1,6}\s?\*")

MOLDE_INSTRUCAO = "Categorize a transação do cartão: {estabelecimento}\nCategorias: {categorias}"


def limpar_estabelecimento(texto: str) -> str:
    """O que o modelo vê: o nome como está na fatura, sem o número da parcela.

    O hífen solto do fim também sai: "108-RIACHUELO-01/02" vira "108 - RIACHUELO".
    """
    return re.sub(r"\s+", " ", RE_PARCELA.sub("", texto)).strip(" -")


def chave(estabelecimento: str) -> str:
    """Identidade do estabelecimento, para o split.

    Variações do mesmo lugar precisam cair do mesmo lado do split, senão a
    validação mede memorização: "DROGASIL 3375", "DROGASIL277-CT" e
    "IFD*RAIA DROGASIL S/A" / "RAIA DROGASIL SA" viram a mesma chave.
    """
    s = unicodedata.normalize("NFKD", estabelecimento.upper())
    s = RE_PREFIXO.sub("", s)
    s = re.sub(r"-CT\b.*$", "", s)
    return re.sub(r"[^A-Z]", "", s)


def extrair_estabelecimentos(linhas: list[str]) -> list[str]:
    """Nomes de todas as linhas de lançamento de uma coluna."""
    return [limpar_estabelecimento(m.group(2))
            for m in map(RE_LANCAMENTO.match, linhas) if m]


def extrair_pdf(caminho: Path) -> list[str]:
    import pdfplumber

    nomes = []
    with pdfplumber.open(caminho) as pdf:
        for pagina in pdf.pages:
            for x0, x1 in ((0, DIVISA_COLUNAS), (DIVISA_COLUNAS, pagina.width)):
                # x_tolerance=1: com o padrão (3) o pdfplumber cola as palavras,
                # e "RAIA DROGASIL" vira "RAIADROGASIL".
                texto = pagina.crop((x0, 0, x1, pagina.height)).extract_text(x_tolerance=1)
                nomes += extrair_estabelecimentos((texto or "").splitlines())
    return nomes


def ler_rotulos() -> dict[str, dict]:
    if not ROTULOS.exists():
        return {}
    with ROTULOS.open(encoding="utf-8", newline="") as f:
        return {linha["estabelecimento"]: linha for linha in csv.DictReader(f)}


def gravar_rotulos(rotulos: dict[str, dict]) -> None:
    with ROTULOS.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=["estabelecimento", "categoria", "confianca"])
        escritor.writeheader()
        for nome in sorted(rotulos, key=str.lower):
            escritor.writerow(rotulos[nome])


def montar_exemplo(estabelecimento: str, categoria: str) -> dict:
    return {
        "instrucao": MOLDE_INSTRUCAO.format(estabelecimento=estabelecimento,
                                            categorias=", ".join(CATEGORIAS)),
        "resposta": categoria,
        "termo": chave(estabelecimento),
    }


def gravar(nome: str, dados: list[dict]) -> None:
    caminho = SAIDA / f"{nome}.jsonl"
    with caminho.open("w", encoding="utf-8") as f:
        for item in dados:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{caminho.relative_to(RAIZ)}: {len(dados)} exemplos")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdfs", nargs="*", type=Path, help="faturas do Itaú em PDF")
    args = parser.parse_args()

    SAIDA.mkdir(parents=True, exist_ok=True)
    rotulos = ler_rotulos()

    novos = set()
    for pdf in args.pdfs:
        nomes = extrair_pdf(pdf)
        print(f"{pdf.name}: {len(nomes)} lançamentos, {len(set(nomes))} nomes distintos")
        novos |= set(nomes) - rotulos.keys()
    for nome in novos:
        rotulos[nome] = {"estabelecimento": nome, "categoria": "", "confianca": ""}
    if novos:
        gravar_rotulos(rotulos)
        print(f"{len(novos)} estabelecimentos novos acrescentados a "
              f"{ROTULOS.relative_to(RAIZ)}, sem categoria")

    invalidos = {n: r["categoria"] for n, r in rotulos.items()
                 if r["categoria"] and r["categoria"] not in CATEGORIAS}
    if invalidos:
        raise SystemExit(f"categorias fora da lista em {ROTULOS.name}: {invalidos}")

    rotulados = {n: r["categoria"] for n, r in rotulos.items() if r["categoria"]}
    pendentes = len(rotulos) - len(rotulados)
    if pendentes:
        print(f"\n{pendentes} estabelecimentos sem categoria em {ROTULOS.name}: "
              f"ficam fora do dataset até serem rotulados")
    if not rotulados:
        raise SystemExit("nenhum estabelecimento rotulado ainda")

    exemplos = [montar_exemplo(n, c) for n, c in sorted(rotulados.items())]

    # Split por ESTABELECIMENTO, como o do glossário é por termo — e por GRUPO de
    # chaves parecidas, para "DROGASIL 2771" e "RAIA DROGASIL SA" não caírem
    # cada um de um lado.
    chaves = sorted({e["termo"] for e in exemplos})
    grupos = agrupar(chaves)
    random.Random(SEMENTE).shuffle(grupos)
    chaves_validacao: set[str] = set()
    for grupo in grupos:
        if len(chaves_validacao) >= round(len(chaves) * FRACAO_VALIDACAO):
            break
        chaves_validacao |= grupo

    treino = [e for e in exemplos if e["termo"] not in chaves_validacao]
    validacao = [e for e in exemplos if e["termo"] in chaves_validacao]
    gravar("treino", treino)
    gravar("validacao", validacao)
    print(f"{len(exemplos)} nomes de {len(chaves)} estabelecimentos "
          f"({len(chaves_validacao)} na validação)")

    print("\nexemplos reais por categoria:")
    for cat, n in Counter(rotulados.values()).most_common():
        print(f"  {cat:<13} {n:>4}  ({n / len(rotulados):.0%})")

    gravar_com_ficticios(treino, validacao, chaves_validacao)


def parecidas(a: str, b: str) -> bool:
    """Duas chaves do mesmo estabelecimento (ou da mesma marca), por heurística.

    Iguais; uma contida na outra ("DROGASIL" em "RAIADROGASILSA"); ou o mesmo
    começo ("GIRAFFASPIER" x "GIRAFFASSPAEROPO"). Erra para o lado de juntar
    demais, que só custa um split um pouco menos aleatório.
    """
    if a == b:
        return True
    if min(len(a), len(b)) >= 5 and (a in b or b in a):
        return True
    return len(a) >= 7 and len(b) >= 7 and a[:7] == b[:7]


def agrupar(chaves: list[str]) -> list[set[str]]:
    """Componentes conexos de "parecidas": se A~B e B~C, os três ficam juntos."""
    grupos: list[set[str]] = []
    for c in chaves:
        ligados = [g for g in grupos if any(parecidas(c, outra) for outra in g)]
        novo = {c}.union(*ligados)
        grupos = [g for g in grupos if g not in ligados] + [novo]
    return sorted(grupos, key=lambda g: sorted(g))


def gravar_com_ficticios(treino: list[dict], validacao: list[dict],
                         chaves_validacao: set[str]) -> None:
    """Versão _ficticios: treino real + estabelecimentos inventados; validação idêntica.

    Os fictícios entram SÓ no treino. Se entrassem na validação, o número mediria
    o quanto o modelo imita nomes inventados, não o quanto acerta a sua fatura.
    E qualquer fictício parecido com um estabelecimento da validação é descartado:
    seria a resposta da prova vazando para o treino.
    """
    with FICTICIOS.open(encoding="utf-8", newline="") as f:
        linhas = list(csv.DictReader(f))
    invalidos = [l for l in linhas if l["categoria"] not in CATEGORIAS]
    if invalidos:
        raise SystemExit(f"categorias fora da lista em {FICTICIOS.name}: {invalidos}")

    # O Itaú corta o nome em 22 caracteres; os fictícios seguem a mesma regra.
    ficticios, descartados = [], []
    for l in linhas:
        exemplo = montar_exemplo(l["estabelecimento"][:22].strip(), l["categoria"])
        if any(parecidas(exemplo["termo"], c) for c in chaves_validacao):
            descartados.append(l["estabelecimento"])
        else:
            ficticios.append(exemplo)

    print(f"\n{len(ficticios)} estabelecimentos fictícios de {FICTICIOS.name}")
    if descartados:
        print(f"  descartados por lembrar um da validação: {', '.join(descartados)}")
    gravar("treino_ficticios", treino + ficticios)
    gravar("validacao_ficticios", validacao)


if __name__ == "__main__":
    main()
