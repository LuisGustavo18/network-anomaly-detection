from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ARQUIVO = Path(
    "data/CIC-IDS2017/cic_ids2017_preparado.parquet"
)

PASTA_SAIDA = Path("data/CIC-IDS2017")

print("Carregando dataset preparado...")

df = pd.read_parquet(ARQUIVO)

y = df["Label"].astype(str)
grupos = df["grupo_duplicata"]

# Divisão em cinco partes: uma para teste e quatro para treino
divisor = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

indice_treino, indice_teste = next(
    divisor.split(df, y, groups=grupos)
)

treino = df.iloc[indice_treino].copy()
teste = df.iloc[indice_teste].copy()


def mostrar_distribuicao(nome, dados):
    print(f"\n{nome}")
    print("-" * 45)

    contagem = dados["Label"].value_counts()
    percentual = dados["Label"].value_counts(normalize=True) * 100

    resultado = pd.DataFrame({
        "Registros": contagem,
        "Percentual": percentual.round(2),
    })

    print(resultado.to_string())
    print("Total:", len(dados))
    print("Grupos:", dados["grupo_duplicata"].nunique())


mostrar_distribuicao("TREINAMENTO", treino)
mostrar_distribuicao("TESTE", teste)


# Verifica vazamento de grupos entre as divisões
grupos_treino = set(treino["grupo_duplicata"])
grupos_teste = set(teste["grupo_duplicata"])

intersecao = grupos_treino.intersection(grupos_teste)

print("\n" + "=" * 45)
print("AUDITORIA DE VAZAMENTO")
print("=" * 45)

print("Grupos compartilhados:", len(intersecao))

if intersecao:
    raise RuntimeError("Encontrado vazamento entre treino e teste.")

print("Nenhum grupo compartilhado.")


# Salva as divisões
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

arquivo_treino = PASTA_SAIDA / "cic_ids2017_treino.parquet"
arquivo_teste = PASTA_SAIDA / "cic_ids2017_teste.parquet"

treino.to_parquet(arquivo_treino, index=False)
teste.to_parquet(arquivo_teste, index=False)

print("\nArquivos salvos:")
print(arquivo_treino)
print(arquivo_teste)

print("\nSEPARAÇÃO CONCLUÍDA.")