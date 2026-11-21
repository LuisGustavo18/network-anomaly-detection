from pathlib import Path

import pandas as pd


PASTA = Path("data/CIC-IDS2017")

arquivos = {
    "TREINAMENTO": PASTA / "cic_ids2017_treino.parquet",
    "TESTE": PASTA / "cic_ids2017_teste.parquet",
}

for nome, arquivo in arquivos.items():

    df = pd.read_parquet(
        arquivo,
        columns=["Destination Port", "Label"],
    )

    df["Destination Port"] = pd.to_numeric(
        df["Destination Port"],
        errors="coerce",
    )

    tabela = pd.crosstab(
        df["Destination Port"],
        df["Label"],
    )

    for classe in ["BENIGN", "Bot"]:
        if classe not in tabela.columns:
            tabela[classe] = 0

    tabela["Total"] = tabela["BENIGN"] + tabela["Bot"]

    tabela["Bot_%_da_porta"] = (
        tabela["Bot"] / tabela["Total"] * 100
    ).round(2)

    total_bots = tabela["Bot"].sum()

    tabela["Participacao_dos_bots_%"] = (
        tabela["Bot"] / total_bots * 100
    ).round(2)

    print("\n" + "=" * 65)
    print(nome)
    print("=" * 65)

    print("\nPORTAS COM MAIS REGISTROS BOT:")

    print(
        tabela.sort_values("Bot", ascending=False)
        .head(15)
        .to_string()
    )

    print("\nPORTAS DISTINTAS UTILIZADAS PELOS BOTS:")
    print(tabela[tabela["Bot"] > 0].shape[0])


print("\nAUDITORIA CONCLUÍDA.")