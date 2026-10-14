"""Testes do parser da fatura — com linhas inventadas, nunca com uma fatura real.

O que se protege aqui é o split: se duas grafias do mesmo estabelecimento caem
em lados diferentes, a validação passa a medir memorização e ninguém percebe.

Uso:
    .venv/bin/python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from preparar_fatura import (  # noqa: E402
    agrupar,
    chave,
    extrair_estabelecimentos,
    limpar_estabelecimento,
    parecidas,
)


class TestFatura(unittest.TestCase):
    def test_extrai_nomes_e_ignora_categoria_do_banco(self):
        linhas = [
            "Lançamentos: compras e saques",
            "DATA ESTABELECIMENTO",
            "18/08 IFD*PADARIA CENTRAL 13,47",
            "ALIMENTAÇÃO .FORTALEZA",
            "24/07 LOJA DE ROUPAS SA - 0,01",
            "VESTUÁRIO .",
            "21/08 SITE.DEV 758,80",
            "HOUSTON 717,55 BRL 138,72",
            "Lançamentos no cartão (final 0000) 5.225,23",
        ]
        self.assertEqual(extrair_estabelecimentos(linhas),
                         ["IFD*PADARIA CENTRAL", "LOJA DE ROUPAS SA", "SITE.DEV"])

    def test_remove_numero_da_parcela(self):
        self.assertEqual(limpar_estabelecimento("LOJAS EXEMPLO SA01/03"), "LOJAS EXEMPLO SA")
        self.assertEqual(limpar_estabelecimento("ITAUSHOP 06/10"), "ITAUSHOP")
        self.assertEqual(limpar_estabelecimento("108 - LOJA - 01/02"), "108 - LOJA")
        self.assertEqual(limpar_estabelecimento("APPLE.COM/BILL"), "APPLE.COM/BILL")

    def test_grafias_do_mesmo_lugar_tem_a_mesma_chave(self):
        self.assertEqual(chave("IFD*RAIA DROGASIL S/A"), chave("RAIA DROGASIL SA"))
        self.assertEqual(chave("DROGASIL 3375"), chave("DROGASIL277-CT"))
        self.assertEqual(chave("APP *LOJA"), chave("LOJA"))
        self.assertNotEqual(chave("SHOPEE*LOJA A"), chave("SHOPEE*LOJA B"))

    def test_marcas_parecidas_caem_no_mesmo_grupo(self):
        chaves = [chave(n) for n in ("DROGASIL 2771", "RAIA DROGASIL SA",
                                     "GIRAFFAS PIER 21", "GIRAFFAS - SP - AEROPO",
                                     "PADARIA CENTRAL")]
        grupos = agrupar(chaves)
        self.assertEqual(len(grupos), 3)
        self.assertIn({chave("DROGASIL 2771"), chave("RAIA DROGASIL SA")}, grupos)
        self.assertIn({chave("GIRAFFAS PIER 21"), chave("GIRAFFAS - SP - AEROPO")}, grupos)
        self.assertFalse(parecidas(chave("UBER *TRIP"), chave("UVLINE")))


if __name__ == "__main__":
    unittest.main()
