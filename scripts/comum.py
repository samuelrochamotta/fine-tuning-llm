"""Peças compartilhadas pelos passos 2 (fine-tuning completo) e 3 (LoRA).

O ponto central deste arquivo é `tokenizar_exemplo`: é ali que a MÁSCARA DE LOSS
acontece. Os dois scripts usam exatamente o mesmo dado e o mesmo pipeline, para que
a única variável entre eles seja a técnica de treino.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

RAIZ = Path(__file__).resolve().parent.parent
DIR_DADOS = RAIZ / "data" / "exemplo"
DIR_SAIDA = RAIZ / "outputs"

# 135M de parâmetros: pequeno o bastante para treinar em CPU/MPS em minutos,
# grande o bastante para já ter gramática e algum conhecimento do pré-treino.
MODELO_BASE = "HuggingFaceTB/SmolLM2-135M"

MAX_TOKENS = 320

# SmolLM2-135M é um modelo BASE: não tem chat template, só sabe continuar texto.
# Então inventamos um formato e o ensinamos por repetição. Os marcadores precisam
# ser idênticos no treino e na inferência — se divergirem, o modelo não reconhece
# o pedido e a qualidade despenca sem erro nenhum aparecer.
MOLDE_PROMPT = "### Pergunta:\n{instrucao}\n\n### Resposta:\n"

PROMPTS_DE_TESTE = [
    "O que é tokenizer?",              # termo do conjunto de VALIDAÇÃO
    "O que é gradient accumulation?",  # termo do conjunto de TREINO
    "O que é uma cadeia de blocos?",   # termo que não está em lugar nenhum
]


def escolher_dispositivo() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def carregar_exemplos(nome: str) -> list[dict]:
    caminho = DIR_DADOS / f"{nome}.jsonl"
    if not caminho.exists():
        raise SystemExit(
            f"{caminho} não existe. Rode antes: python scripts/preparar_dados.py"
        )
    with caminho.open(encoding="utf-8") as f:
        return [json.loads(linha) for linha in f]


def tokenizar_exemplo(exemplo: dict, tokenizer) -> dict:
    """Converte um par instrução/resposta em input_ids e labels mascarados.

    Prompt e resposta são tokenizados SEPARADAMENTE e concatenados. Assim sabemos
    exatamente onde termina um e começa o outro, sem depender de procurar o
    marcador no texto já tokenizado (que quebra quando o tokenizer funde
    caracteres na fronteira).

    Em `labels`, todo token do prompt vira -100. Esse é o valor que a cross-entropy
    do PyTorch ignora por padrão: o modelo é cobrado apenas pela resposta, não por
    reproduzir a pergunta que já lhe foi entregue.
    """
    ids_prompt = tokenizer(
        MOLDE_PROMPT.format(instrucao=exemplo["instrucao"]),
        add_special_tokens=False,
    )["input_ids"]

    # O EOS ensina o modelo a PARAR. Sem ele, a geração emenda uma resposta na
    # outra até bater o limite de tokens.
    ids_resposta = tokenizer(
        exemplo["resposta"], add_special_tokens=False
    )["input_ids"] + [tokenizer.eos_token_id]

    input_ids = (ids_prompt + ids_resposta)[:MAX_TOKENS]
    labels = ([-100] * len(ids_prompt) + ids_resposta)[:MAX_TOKENS]

    return {"input_ids": input_ids, "labels": labels}


class DatasetSFT(Dataset):
    """Lista de exemplos já tokenizados.

    Em projetos maiores isto seria um `datasets.Dataset` com `.map()`. Aqui é uma
    lista Python de propósito: dá para imprimir um item e ver exatamente o que
    entra no modelo.
    """

    def __init__(self, exemplos: list[dict], tokenizer):
        self.itens = [tokenizar_exemplo(e, tokenizer) for e in exemplos]

    def __len__(self) -> int:
        return len(self.itens)

    def __getitem__(self, i: int) -> dict:
        return self.itens[i]


@dataclass
class ColetorSFT:
    """Junta exemplos de comprimentos diferentes em um batch retangular.

    Três detalhes que costumam passar batido:
      - input_ids recebem padding com o pad token;
      - labels recebem padding com -100, NUNCA com o pad token (senão o modelo é
        treinado para prever preenchimento);
      - attention_mask marca o que é conteúdo real, para o modelo não prestar
        atenção no padding.
    """

    tokenizer: object

    def __call__(self, exemplos: list[dict]) -> dict:
        maior = max(len(e["input_ids"]) for e in exemplos)
        pad_id = self.tokenizer.pad_token_id

        lote = {"input_ids": [], "attention_mask": [], "labels": []}
        for e in exemplos:
            falta = maior - len(e["input_ids"])
            lote["input_ids"].append(e["input_ids"] + [pad_id] * falta)
            lote["attention_mask"].append([1] * len(e["input_ids"]) + [0] * falta)
            lote["labels"].append(e["labels"] + [-100] * falta)

        return {k: torch.tensor(v, dtype=torch.long) for k, v in lote.items()}


def carregar_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    if tokenizer.pad_token is None:
        # Modelos base costumam não ter pad token. Reaproveitar o EOS é seguro
        # aqui porque a attention_mask garante que o padding será ignorado.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@torch.no_grad()
def gerar(modelo, tokenizer, instrucao: str, max_novos: int = 160) -> str:
    """Gera uma resposta. Sempre com o mesmo formato de prompt usado no treino."""
    modelo.eval()  # desliga dropout: sem isto a saída varia entre chamadas iguais
    dispositivo = next(modelo.parameters()).device
    entrada = tokenizer(
        MOLDE_PROMPT.format(instrucao=instrucao), return_tensors="pt"
    ).to(dispositivo)

    saida = modelo.generate(
        **entrada,
        max_new_tokens=max_novos,
        do_sample=False,          # greedy: comparações antes/depois viram loteria com amostragem
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # devolve só o que foi gerado, sem repetir o prompt
    return tokenizer.decode(
        saida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def mostrar_amostras(modelo, tokenizer, titulo: str) -> None:
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")
    for prompt in PROMPTS_DE_TESTE:
        print(f"\n>>> {prompt}")
        print(gerar(modelo, tokenizer, prompt))
    print()


def passos_de_warmup(n_exemplos: int, batch: int, epocas: float, fracao: float = 0.1) -> int:
    """10% dos passos totais em warmup.

    `TrainingArguments` só aceita o número absoluto de passos (o antigo
    `warmup_ratio` foi removido no transformers 5), então a conta fica explícita:
    passos por época = exemplos / batch; total = isso vezes as épocas.
    """
    import math

    total = math.ceil(n_exemplos / batch) * epocas
    return max(1, int(total * fracao))


def resumir_parametros(modelo) -> None:
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    print(f"parâmetros treináveis: {treinaveis:,} de {total:,} "
          f"({100 * treinaveis / total:.4f}%)")
