from pathlib import Path

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score


PASTA = Path("data/CIC-IDS2017")

ARQUIVO_MODELO = Path(
    "models/detector_cic_ids2017_sem_porta.joblib"
)

print("Carregando modelo e teste...")

modelo = joblib.load(ARQUIVO_MODELO)

df = pd.read_parquet(
    PASTA / "cic_ids2017_teste.parquet"
)

X = df.drop(
    columns=[
        "Label",
        "grupo_duplicata",
        "Destination Port",
    ]
)

y = df["Label"].astype(str)

def calcular_ap_bot(estimator, X, y):
    classes = list(
        estimator.named_steps["classificador"].classes_
    )

    indice_bot = classes.index("Bot")

    probabilidades = estimator.predict_proba(X)[:, indice_bot]

    y_binario = (pd.Series(y).astype(str) == "Bot").astype(int)

    return average_precision_score(
        y_binario,
        probabilidades,
    )


print("\nCalculando importância das características...")
print("Serão utilizadas 3 repetições por característica.")

resultado = permutation_importance(
    modelo,
    X,
    y,
    scoring=calcular_ap_bot,
    n_repeats=3,
    random_state=42,
    n_jobs=1,
    max_samples=10000,
)

tabela = pd.DataFrame({
    "Caracteristica": X.columns,
    "Queda_AP_pp": resultado.importances_mean * 100,
    "Desvio": resultado.importances_std * 100,
})

tabela = tabela.sort_values(
    "Queda_AP_pp",
    ascending=False,
)

print("\n20 CARACTERÍSTICAS MAIS IMPORTANTES\n")
print(tabela.head(20).round(4).to_string(index=False))

saida = PASTA / "importancia_features_sem_porta.csv"

tabela.to_csv(saida, index=False)

print("\nRelatório salvo em:", saida)
print("\nANÁLISE CONCLUÍDA.")