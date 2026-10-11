"""Testes da máscara de loss e do collator — o ponto mais fácil de errar sem perceber.

Um erro aqui não quebra nada: o treino roda, a perda cai, e o modelo aprende a
coisa errada (repetir a pergunta, prever padding, nunca parar). Por isso vale
um teste.

Uso:
    .venv/bin/python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from comum import (  # noqa: E402
    MODELO_INSTRUCT,
    ColetorSFT,
    carregar_tokenizer,
    montar_prompt,
    tokenizar_exemplo,
)

EXEMPLO = {"instrucao": "O que é LoRA?", "resposta": "**Definição:** Um adaptador."}


class TestMascara(unittest.TestCase):
    tokenizers = {}

    @classmethod
    def setUpClass(cls):
        cls.tokenizers = {"base": carregar_tokenizer(),
                          "instruct": carregar_tokenizer(MODELO_INSTRUCT)}

    def test_prompt_mascarado_resposta_cobrada(self):
        for nome, tok in self.tokenizers.items():
            with self.subTest(tokenizer=nome):
                item = tokenizar_exemplo(EXEMPLO, tok)
                n_prompt = len(tok(montar_prompt(EXEMPLO["instrucao"], tok),
                                   add_special_tokens=False)["input_ids"])
                ids, labels = item["input_ids"], item["labels"]

                self.assertEqual(len(ids), len(labels))
                # todo o prompt é ignorado pela cross-entropy...
                self.assertEqual(labels[:n_prompt], [-100] * n_prompt)
                # ...e toda a resposta é cobrada, token a token
                self.assertEqual(labels[n_prompt:], ids[n_prompt:])
                self.assertNotIn(-100, labels[n_prompt:])

    def test_resposta_termina_no_eos(self):
        # Sem o EOS nos labels o modelo aprende o formato, mas não aprende a parar.
        for nome, tok in self.tokenizers.items():
            with self.subTest(tokenizer=nome):
                item = tokenizar_exemplo(EXEMPLO, tok)
                self.assertEqual(item["labels"][-1], tok.eos_token_id)
                self.assertEqual(item["input_ids"][-1], tok.eos_token_id)

    def test_instruct_usa_chat_template(self):
        self.assertIsNone(self.tokenizers["base"].chat_template)
        prompt = montar_prompt("oi", self.tokenizers["instruct"])
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertEqual(self.tokenizers["instruct"].eos_token, "<|im_end|>")

    def test_collator_preenche_labels_com_menos_100(self):
        tok = self.tokenizers["base"]
        curto = tokenizar_exemplo(EXEMPLO, tok)
        longo = tokenizar_exemplo({**EXEMPLO, "resposta": EXEMPLO["resposta"] * 5}, tok)
        lote = ColetorSFT(tok)([curto, longo])
        falta = len(longo["input_ids"]) - len(curto["input_ids"])
        self.assertGreater(falta, 0)

        linha = lote["labels"][0].tolist()
        # padding nos labels é -100, NUNCA o pad token — senão o modelo é
        # treinado para prever preenchimento (e aqui pad == eos!)
        self.assertEqual(linha[-falta:], [-100] * falta)
        self.assertEqual(lote["input_ids"][0, -falta:].tolist(), [tok.pad_token_id] * falta)
        self.assertEqual(lote["attention_mask"][0, -falta:].tolist(), [0] * falta)
        self.assertEqual(lote["attention_mask"][1].tolist(), [1] * len(longo["input_ids"]))


if __name__ == "__main__":
    unittest.main()
