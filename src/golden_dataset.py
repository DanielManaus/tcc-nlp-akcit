"""Conjunto-ouro (golden dataset) para avaliação.

~40 perguntas sobre o CDC organizadas nas categorias do TCC (§6):
- direito expresso
- direito implícito
- fora do escopo
- interpretações múltiplas

Cada item: pergunta, categoria, artigos gabarito (para precisão de citação),
e 'fora_escopo' (True quando a resposta esperada é recusa).
"""
GOLDEN_DATASET = [
    # ---- Direito expresso ----
    {"category": "direito_expresso", "question": "Qual o prazo de reclamação por vício oculto do produto?",
     "expected_articles": ["26"], "out_of_scope": False},
    {"category": "direito_expresso", "question": "O que é considerado produto no CDC?",
     "expected_articles": ["3"], "out_of_scope": False},
    {"category": "direito_expresso", "question": "Quais são os direitos básicos do consumidor?",
     "expected_articles": ["6"], "out_of_scope": False},
    {"category": "direito_expresso", "question": "Qual a prescrição para vício do serviço?",
     "expected_articles": ["27"], "out_of_scope": False},
    {"category": "direito_expresso", "question": "O que é oferta segundo o CDC?",
     "expected_articles": ["30"], "out_of_scope": False},
    {"category": "direito_expresso", "question": "Quem é considerado fornecedor?",
     "expected_articles": ["3"], "out_of_scope": False},
    {"category": "direito_expresso", "question": "O que diz o artigo 51 sobre cláusulas abusivas?",
     "expected_articles": ["51"], "out_of_scope": False},
    {"category": "direito_expresso", "question": "Quais práticas comerciais são consideradas abusivas?",
     "expected_articles": ["39"], "out_of_scope": False},
    {"category": "direito_expresso", "question": "Qual a responsabilidade do fornecedor por fato do produto?",
     "expected_articles": ["12"], "out_of_scope": False},
    {"category": "direito_expresso", "question": "O que é publicidade enganosa?",
     "expected_articles": ["36"], "out_of_scope": False},
    {"category": "direito_expresso", "question": "Como funciona o direito de arrependimento na compra fora do estabelecimento?",
     "expected_articles": ["49"], "out_of_scope": False},
    {"category": "direito_expresso", "question": "Quais são as penalidades administrativas previstas?",
     "expected_articles": ["56"], "out_of_scope": False},

    # ---- Direito implícito ----
    {"category": "direito_implicito", "question": "O CDC protege o consumidor de serviços bancários?",
     "expected_articles": ["3", "2"], "out_of_scope": False},
    {"category": "direito_implicito", "question": "Aplicação do CDC a contratos de plano de saúde é possível?",
     "expected_articles": ["2", "3"], "out_of_scope": False},
    {"category": "direito_implicito", "question": "O consumidor pode exigir troca de produto com defeito?",
     "expected_articles": ["18"], "out_of_scope": False},
    {"category": "direito_implicito", "question": "Há proteção para compras online no CDC?",
     "expected_articles": ["49"], "out_of_scope": False},
    {"category": "direito_implicito", "question": "O CDC se aplica a relações de consumo interestaduais?",
     "expected_articles": ["2"], "out_of_scope": False},
    {"category": "direito_implicito", "question": "É possível a inversão do ônus da prova?",
     "expected_articles": ["6"], "out_of_scope": False},
    {"category": "direito_implicito", "question": "O fornecedor responde por dano moral?",
     "expected_articles": ["6", "14"], "out_of_scope": False},
    {"category": "direito_implicito", "question": "A boa-fé é princípio do CDC?",
     "expected_articles": ["4"], "out_of_scope": False},
    {"category": "direito_implicito", "question": "O CDC exige informação clara sobre produtos?",
     "expected_articles": ["6", "31"], "out_of_scope": False},
    {"category": "direito_implicito", "question": "Há previsão de decadência em relação à garantia?",
     "expected_articles": ["26"], "out_of_scope": False},

    # ---- Fora do escopo ----
    {"category": "fora_escopo", "question": "Qual a pena para homicídio no Código Penal?",
     "expected_articles": [], "out_of_scope": True},
    {"category": "fora_escopo", "question": "Como é calculado o Imposto de Renda?",
     "expected_articles": [], "out_of_scope": True},
    {"category": "fora_escopo", "question": "Quais são os direitos previdenciários de aposentadoria?",
     "expected_articles": [], "out_of_scope": True},
    {"category": "fora_escopo", "question": "Qual a constitucionalidade do impeachment presidencial?",
     "expected_articles": [], "out_of_scope": True},
    {"category": "fora_escopo", "question": "Como registrar uma marca no INPI?",
     "expected_articles": [], "out_of_scope": True},
    {"category": "fora_escopo", "question": "Qual o prazo de guarda compartilhada segundo o Código Civil?",
     "expected_articles": [], "out_of_scope": True},
    {"category": "fora_escopo", "question": "Quem tem direito a auxílio-doença do INSS?",
     "expected_articles": [], "out_of_scope": True},
    {"category": "fora_escopo", "question": "Como funciona o divórcio no Brasil?",
     "expected_articles": [], "out_of_scope": True},
    {"category": "fora_escopo", "question": "Qual a idade mínima para aposentadoria?",
     "expected_articles": [], "out_of_scope": True},
    {"category": "fora_escopo", "question": "Como abrir uma empresa no MEI?",
     "expected_articles": [], "out_of_scope": True},

    # ---- Interpretações múltiplas ----
    {"category": "interpretacoes_multiplas", "question": "O que é considerado vício do produto?",
     "expected_articles": ["18"], "out_of_scope": False},
    {"category": "interpretacoes_multiplas", "question": "Quando ocorre a responsabilidade solidária?",
     "expected_articles": ["7"], "out_of_scope": False},
    {"category": "interpretacoes_multiplas", "question": "O que caracteriza relação de consumo?",
     "expected_articles": ["2"], "out_of_scope": False},
    {"category": "interpretacoes_multiplas", "question": "Como interpretar cláusula contratual duvidosa?",
     "expected_articles": ["47"], "out_of_scope": False},
    {"category": "interpretacoes_multiplas", "question": "O que são serviços essenciais e sua regulação?",
     "expected_articles": ["22"], "out_of_scope": False},
    {"category": "interpretacoes_multiplas", "question": "Como definir consumidor por equiparação?",
     "expected_articles": ["2"], "out_of_scope": False},
    {"category": "interpretacoes_multiplas", "question": "O que são práticas comerciais abusivas em cobranças?",
     "expected_articles": ["39", "42"], "out_of_scope": False},
    {"category": "interpretacoes_multiplas", "question": "Como funciona a responsabilidade por serviço defeituoso?",
     "expected_articles": ["14"], "out_of_scope": False},
    {"category": "interpretacoes_multiplas", "question": "O que o CDC prevê sobre cadastros e bancos de dados de consumidores?",
     "expected_articles": ["43"], "out_of_scope": False},
    {"category": "interpretacoes_multiplas", "question": "Como aplicar a proteção contra violação de direitos?",
     "expected_articles": ["5"], "out_of_scope": False},
]


if __name__ == "__main__":
    print(f"Total de perguntas no conjunto-ouro: {len(GOLDEN_DATASET)}")
    from collections import Counter
    c = Counter(d["category"] for d in GOLDEN_DATASET)
    for k, v in c.items():
        print(f"  {k}: {v}")
