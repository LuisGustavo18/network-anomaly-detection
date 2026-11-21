from pathlib import Path

import pandas as pd


PASTA = Path("data/CIC-IDS2017")

ARQUIVOS = {
    "TREINAMENTO": PASTA / "cic_ids2017_treino.parquet",
    "TESTE": PASTA / "cic_ids2017_teste.parquet",
}

CARACTERISTICAS = [
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]


for nome, arquivo in ARQUIVOS.items():

    df = pd.read_parquet(
        arquivo,
        columns=["Label"] + CARACTERISTICAS,
    )

    print("\n" + "=" * 65)
    print(nome)
    print("=" * 65)

    for coluna in CARACTERISTICAS:

        print(f"\n{'-' * 55}")
        print(coluna)
        print("-" * 55)

        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="coerce",
        )

        print("\nVALORES DISTINTOS POR CLASSE:")

        print(
            df.groupby("Label")[coluna]
            .nunique()
            .to_string()
        )

        for classe in ["BENIGN", "Bot"]:

            grupo = df[df["Label"] == classe][coluna]

            print(f"\n12 VALORES MAIS FREQUENTES — {classe}")

            contagem = grupo.value_counts(dropna=False).head(12)

            tabela = pd.DataFrame({
                "Quantidade": contagem,
                "Percentual": (
                    contagem / len(grupo) * 100
                ).round(2),
            })

            print(tabela.to_string())

        print("\nESTATÍSTICAS POR CLASSE:")

        print(
            df.groupby("Label")[coluna]
            .describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
            .round(2)
            .to_string()
        )


print("\nAUDITORIA CONCLUÍDA.")