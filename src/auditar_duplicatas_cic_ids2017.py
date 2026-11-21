from pathlib import Path

import pandas as pd


ARQUIVO = Path(
    "data/CIC-IDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"
)

print("CARREGANDO DATASET...")

df = pd.read_csv(ARQUIVO, low_memory=False)
df.columns = df.columns.astype(str).str.strip()
df["Label"] = df["Label"].astype("string").str.strip()

features = [col for col in df.columns if col != "Label"]

# 1. Linhas totalmente duplicadas, incluindo o rótulo
duplicatas_completas = df.duplicated().sum()

# 2. Assinatura das características, sem considerar o rótulo
assinaturas = pd.util.hash_pandas_object(
    df[features],
    index=False,
)

auditoria = pd.DataFrame({
    "assinatura": assinaturas,
    "Label": df["Label"],
})

frequencias = auditoria["assinatura"].value_counts()

assinaturas_repetidas = frequencias[frequencias > 1]

# 3. Assinaturas que aparecem associadas a rótulos diferentes
rotulos_por_assinatura = (
    auditoria.groupby("assinatura")["Label"].nunique()
)

conflitos = rotulos_por_assinatura[
    rotulos_por_assinatura > 1
]

linhas_com_conflito = (
    auditoria["assinatura"].isin(conflitos.index).sum()
)

print("\n" + "=" * 55)
print("DUPLICATAS")
print("=" * 55)

print("Linhas totalmente duplicadas:", duplicatas_completas)
print(
    "Linhas com características já existentes:",
    assinaturas.duplicated().sum(),
)
print(
    "Assinaturas repetidas:",
    len(assinaturas_repetidas),
)
print(
    "Assinaturas com rótulos conflitantes:",
    len(conflitos),
)
print(
    "Linhas envolvidas em conflitos:",
    linhas_com_conflito,
)

print("\n" + "=" * 55)
print("DUPLICATAS POR CLASSE")
print("=" * 55)

for classe in df["Label"].dropna().unique():
    mascara = df["Label"] == classe

    repetidas = assinaturas[mascara].duplicated().sum()

    print(
        f"{classe}: {repetidas} linhas repetidas "
        f"de {mascara.sum()} registros"
    )

print("\n" + "=" * 55)
print("COLUNAS CONSTANTES")
print("=" * 55)

variacoes = df[features].nunique(dropna=False)

constantes = variacoes[variacoes <= 1]

print(
    constantes.to_string()
    if not constantes.empty
    else "Nenhuma coluna constante."
)

print("\n" + "=" * 55)
print("COLUNAS EXATAMENTE DUPLICADAS")
print("=" * 55)

duplicadas = df[features].T.duplicated(keep=False)

colunas_duplicadas = df[features].columns[duplicadas]

print(
    colunas_duplicadas.tolist()
    if len(colunas_duplicadas)
    else "Nenhuma coluna duplicada."
)

print("\nAUDITORIA CONCLUÍDA.")