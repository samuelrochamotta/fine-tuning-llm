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

# A mesma arquitetura e o mesmo tamanho, mas já ajustada pelo próprio Hugging Face
# para seguir instruções em formato de chat. É o ponto de partida usual na prática.
MODELO_INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODELOS = {"base": MODELO_BASE, "instruct": MODELO_INSTRUCT}

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


def montar_prompt(instrucao: str, tokenizer) -> str:
    """Texto que antecede a resposta, no formato que o modelo espera.

    Modelo base não tem chat template: usamos o MOLDE_PROMPT inventado. Modelo
    instruct já vem com um (ChatML, no caso do SmolLM2), e é ele que deve ser
    usado — foi nesse formato que o modelo aprendeu a separar os turnos. Quem
    decide é o próprio tokenizer, que carrega o template junto.
    """
    if tokenizer.chat_template is None:
        return MOLDE_PROMPT.format(instrucao=instrucao)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instrucao}],
        tokenize=False,
        add_generation_prompt=True,  # termina em "<|im_start|>assistant\n"
    )


def carregar_exemplos(nome: str, dataset: str = "") -> list[dict]:
    """Lê data/exemplo/<nome>.jsonl, ou <nome>_<dataset>.jsonl (ex.: treino_grande)."""
    caminho = DIR_DADOS / (f"{nome}_{dataset}.jsonl" if dataset else f"{nome}.jsonl")
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
        montar_prompt(exemplo["instrucao"], tokenizer),
        add_special_tokens=False,
    )["input_ids"]

    # O EOS ensina o modelo a PARAR. Sem ele, a geração emenda uma resposta na
    # outra até bater o limite de tokens. No modelo instruct o EOS é o
    # "<|im_end|>" que fecha o turno, não o "<|endoftext|>" do modelo base.
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


def carregar_modelo(origem: str | Path, dispositivo: str):
    """Carrega (modelo, tokenizer) de um id do Hub ou de uma pasta em outputs/.

    Uma pasta com `adapter_config.json` é um adaptador LoRA: o modelo de que ele
    partiu vem do Hub e o adaptador é aplicado por cima — os pesos originais
    continuam intactos no disco e na memória.
    """
    from transformers import AutoModelForCausalLM

    origem = Path(origem) if Path(origem).exists() else origem
    if isinstance(origem, Path) and (origem / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel

        nome_base = PeftConfig.from_pretrained(origem).base_model_name_or_path
        base = AutoModelForCausalLM.from_pretrained(nome_base)
        modelo = PeftModel.from_pretrained(base, origem)
    else:
        modelo = AutoModelForCausalLM.from_pretrained(origem)
    return modelo.to(dispositivo), carregar_tokenizer(origem)


def modelos_disponiveis(extras: list[str] | None = None) -> list[tuple[str, object]]:
    """Base + passos 2 e 3, se já treinados, + pastas extras de outputs/."""
    lista = [("base", MODELO_BASE)]
    for nome in ("sft-completo", "lora"):
        if (DIR_SAIDA / nome).is_dir():
            lista.append((nome, DIR_SAIDA / nome))
    for nome in extras or []:
        caminho = DIR_SAIDA / nome
        lista.append((nome, caminho if caminho.is_dir() else nome))
    return lista


def criar_treinador(*args, peso_eos: float = 1.0, **kwargs):
    """Um `Trainer` comum, ou um que dá peso extra ao token EOS na perda.

    Cada resposta tem ~150 tokens e um único EOS, então o sinal para "pare aqui"
    é menos de 1% da perda. `peso_eos` > 1 multiplica esse sinal — uma forma
    direta de reforçar o aprendizado da parada sem mais dados nem mais épocas.

    O peso só vale no TREINO: na avaliação a perda volta a ser a média simples,
    para que eval_loss continue comparável com os outros runs. As importações
    ficam aqui dentro para que `comum.py` continue leve para quem só tokeniza.
    """
    import torch.nn.functional as F
    from transformers import Trainer

    if peso_eos == 1.0:
        return Trainer(*args, **kwargs)

    class TreinadorPesoEOS(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            if not model.training:
                return super().compute_loss(model, inputs, return_outputs,
                                            num_items_in_batch)
            saida = model(input_ids=inputs["input_ids"],
                          attention_mask=inputs["attention_mask"])
            # o token t prevê o token t+1: desloca logits e labels uma posição
            logits = saida.logits[:, :-1].float()
            alvo = inputs["labels"][:, 1:]
            perda = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), alvo.reshape(-1),
                ignore_index=-100, reduction="none",
            ).view_as(alvo)
            pesos = (alvo != -100).float()
            pesos[alvo == self.processing_class.eos_token_id] *= peso_eos
            loss = (perda * pesos).sum() / pesos.sum()
            return (loss, saida) if return_outputs else loss

    return TreinadorPesoEOS(*args, **kwargs)


def carregar_tokenizer(origem: str | Path = MODELO_BASE):
    """Carrega o tokenizer de um id do Hub ou de uma pasta em outputs/.

    Cada modelo salvo leva o próprio tokenizer junto: o do instruct tem chat
    template e outro EOS, então nunca compartilhe um tokenizer entre modelos.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(origem)
    if tokenizer.pad_token is None:
        # Modelos base costumam não ter pad token. Reaproveitar o EOS é seguro
        # aqui porque a attention_mask garante que o padding será ignorado.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@torch.no_grad()
def gerar(modelo, tokenizer, instrucao: str, max_novos: int = 160,
          detalhes: bool = False):
    """Gera uma resposta. Sempre com o mesmo formato de prompt usado no treino.

    Com `detalhes=True` devolve também quantos tokens foram gerados e se o modelo
    emitiu o EOS por conta própria — ou seja, se soube parar. É o que
    `avaliar.py` mede.
    """
    modelo.eval()  # desliga dropout: sem isto a saída varia entre chamadas iguais
    dispositivo = next(modelo.parameters()).device
    entrada = tokenizer(
        montar_prompt(instrucao, tokenizer), return_tensors="pt", add_special_tokens=False
    ).to(dispositivo)

    saida = modelo.generate(
        **entrada,
        max_new_tokens=max_novos,
        do_sample=False,          # greedy: comparações antes/depois viram loteria com amostragem
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # devolve só o que foi gerado, sem repetir o prompt
    novos = saida[0][entrada["input_ids"].shape[1]:]
    texto = tokenizer.decode(novos, skip_special_tokens=True).strip()
    if not detalhes:
        return texto
    parou = bool((novos == tokenizer.eos_token_id).any())
    n_tokens = int((novos != tokenizer.eos_token_id).sum())
    return texto, {"tokens": n_tokens, "parou": parou}


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


def salvar_historico(treinador, saida: Path, args, extra: dict | None = None) -> None:
    """Grava o log de treino em outputs/<nome>/historico.json.

    As tabelas do README e os gráficos de `plotar_curvas.py` saem daqui, não de
    números copiados à mão do terminal.
    """
    saida.mkdir(parents=True, exist_ok=True)
    registro = {
        "args": vars(args),
        "log_history": treinador.state.log_history,
        **(extra or {}),
    }
    (saida / "historico.json").write_text(
        json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def resumir_parametros(modelo) -> None:
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    print(f"parâmetros treináveis: {treinaveis:,} de {total:,} "
          f"({100 * treinaveis / total:.4f}%)")
