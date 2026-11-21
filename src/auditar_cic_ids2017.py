from pathlib import Path

import numpy as np
import pandas as pd


ARQUIVO = Path(
    "data/CIC-IDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"
)

print("CARREGANDO DATASET...")

df = pd.read_csv(ARQUIVO, low_memory=False)

# Remove espaços desnecessários dos nomes das colunas
df.columns = df.columns.astype(str).str.strip()

print("\n" + "=" * 55)
print("DIMENSÕES DO DATASET")
print("=" * 55)

print("Registros:", len(df))
print("Colunas:", len(df.columns))

print("\nCOLUNAS:")
for coluna in df.columns:
    print(repr(coluna))


print("\n" + "=" * 55)
print("DISTRIBUIÇÃO DAS CLASSES")
print("=" * 55)

if "Label" not in df.columns:
    raise ValueError("Coluna Label não encontrada.")

df["Label"] = df["Label"].astype("string").str.strip()

contagem = df["Label"].value_counts(dropna=False)

print(contagem.to_string())

print("\nPERCENTUAL:")
print(
    (df["Label"].value_counts(normalize=True, dropna=False) * 100)
    .round(2)
    .to_string()
)


print("\n" + "=" * 55)
print("TIPOS DE DADOS")
print("=" * 55)

print(df.dtypes.value_counts())


print("\n" + "=" * 55)
print("VALORES AUSENTES")
print("=" * 55)

nulos = df.isna().sum().sort_values(ascending=False)

print(nulos[nulos > 0].head(20).to_string())


print("\n" + "=" * 55)
print("VALORES INFINITOS")
print("=" * 55)

numericas = df.select_dtypes(include="number").columns

infinitos = {}

for coluna in numericas:
    infinitos[coluna] = np.isinf(df[coluna]).sum()

resultado_inf = pd.Series(infinitos)
resultado_inf = resultado_inf[resultado_inf > 0]

print(
    resultado_inf.to_string()
    if not resultado_inf.empty
    else "Nenhum valor infinito encontrado."
)


print("\nAUDITORIA CONCLUÍDA.")