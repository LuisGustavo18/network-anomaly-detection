from pathlib import Path

import numpy as np
import pandas as pd


ARQUIVO = Path(
    "data/CIC-IDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"
)

SAIDA = Path(
    "data/CIC-IDS2017/cic_ids2017_preparado.parquet"
)

COLUNAS_CONSTANTES = [
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

COLUNAS_REDUNDANTES = [
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Subflow Fwd Bytes",
    "Avg Fwd Segment Size",
    "Fwd Header Length.1",
    "SYN Flag Count",
    "ECE Flag Count",
]

print("Carregando dataset...")

df = pd.read_csv(ARQUIVO, low_memory=False)
df.columns = df.columns.astype(str).str.strip()

if "Label" not in df.columns:
    raise ValueError("Coluna Label não encontrada.")

# Verifica se todas as colunas previstas existem
remover = COLUNAS_CONSTANTES + COLUNAS_REDUNDANTES

ausentes = [coluna for coluna in remover if coluna not in df.columns]

if ausentes:
    raise ValueError(f"Colunas não encontradas: {ausentes}")

# Remove constantes e redundâncias
df = df.drop(columns=remover)

# Normaliza os rótulos
df["Label"] = df["Label"].astype("string").str.strip()

# Substitui infinitos por valores ausentes
df = df.replace([np.inf, -np.inf], np.nan)

# Cria identificador de grupo para evitar vazamento de duplicatas
features = [col for col in df.columns if col != "Label"]

df.insert(
    0,
    "grupo_duplicata",
    pd.util.hash_pandas_object(
        df[features],
        index=False,
    ).astype("uint64"),
)

# Resumo
print("\n" + "=" * 50)
print("DATASET PREPARADO")
print("=" * 50)

print("Registros:", len(df))
print("Colunas totais:", len(df.columns))
print("Características:", len(features))
print("Colunas removidas:", len(remover))

print("\nDistribuição das classes:")
print(df["Label"].value_counts().to_string())

print("\nValores ausentes por característica:")
nulos = df[features].isna().sum()
print(nulos[nulos > 0].to_string())

print("\nGrupos de registros duplicados:")
print("Grupos distintos:", df["grupo_duplicata"].nunique())
print(
    "Registros repetidos:",
    df["grupo_duplicata"].duplicated().sum(),
)

# Salva sem imputar valores ausentes
SAIDA.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(SAIDA, index=False)

print("\nArquivo salvo:", SAIDA)
print("\nPREPARAÇÃO CONCLUÍDA.")