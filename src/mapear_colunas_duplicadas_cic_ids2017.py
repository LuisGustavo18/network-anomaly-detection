from pathlib import Path

import pandas as pd


ARQUIVO = Path(
    "data/CIC-IDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"
)

df = pd.read_csv(ARQUIVO, low_memory=False)
df.columns = df.columns.astype(str).str.strip()

features = [col for col in df.columns if col != "Label"]

print("MAPEANDO COLUNAS DUPLICADAS\n")

pares = []

for i, coluna_a in enumerate(features):
    a = pd.to_numeric(df[coluna_a], errors="coerce")

    for coluna_b in features[i + 1:]:
        b = pd.to_numeric(df[coluna_b], errors="coerce")

        iguais = a.eq(b) | (a.isna() & b.isna())

        if iguais.all():
            pares.append({
                "Coluna_A": coluna_a,
                "Coluna_B": coluna_b,
                "Valores_distintos": a.nunique(dropna=False),
            })

if pares:
    resultado = pd.DataFrame(pares)
    print(resultado.to_string(index=False))
else:
    print("Nenhum par encontrado.")

print("\nTOTAL DE PARES:", len(pares))
print("\nANÁLISE CONCLUÍDA.")