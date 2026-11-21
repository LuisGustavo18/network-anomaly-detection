from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


PASTA = Path("data/CIC-IDS2017")

print("Carregando dados...")

treino = pd.read_parquet(
    PASTA / "cic_ids2017_treino.parquet"
)

teste = pd.read_parquet(
    PASTA / "cic_ids2017_teste.parquet"
)

y_treino = treino["Label"].astype(str)
y_teste = teste["Label"].astype(str)

COLUNAS_REMOVER = [
    "Label",
    "grupo_duplicata",
    "Destination Port",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

X_treino = treino.drop(columns=COLUNAS_REMOVER)
X_teste = teste.drop(columns=COLUNAS_REMOVER)

print("Características:", len(X_treino.columns))


def avaliar(nome, modelo):
    previsoes = modelo.predict(X_teste)

    classes = list(modelo.named_steps["classificador"].classes_)
    indice_bot = classes.index("Bot")

    probabilidades = modelo.predict_proba(X_teste)[:, indice_bot]

    tn, fp, fn, tp = confusion_matrix(
        y_teste,
        previsoes,
        labels=["BENIGN", "Bot"],
    ).ravel()

    precisao = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precisao * recall / (precisao + recall)
        if precisao + recall else 0
    )

    ap = average_precision_score(
        (y_teste == "Bot").astype(int),
        probabilidades,
    )

    print("\n" + "=" * 50)
    print(nome)
    print("=" * 50)

    print(f"Precisão: {precisao * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")
    print(f"Average Precision: {ap * 100:.2f}%")
    print("Falsos positivos:", fp)
    print("Bots perdidos:", fn)


# Modelo atual com 100 árvores
print("\nCarregando modelo de 100 árvores...")

modelo_100 = joblib.load(
    "models/detector_cic_ids2017_sem_porta_sem_janelas.joblib"
)

avaliar("MODELO COM 100 ÁRVORES", modelo_100)


# Novo modelo com 150 árvores
print("\nTreinando modelo com 150 árvores...")

modelo_150 = Pipeline([
    ("imputador", SimpleImputer(
        strategy="median",
        add_indicator=True,
    )),
    ("classificador", RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )),
])

modelo_150.fit(X_treino, y_treino)

print("Treinamento concluído.")

avaliar("MODELO COM 150 ÁRVORES", modelo_150)


saida = Path(
    "models/detector_cic_ids2017_sem_porta_sem_janelas_150.joblib"
)

joblib.dump(modelo_150, saida)

print("\nModelo salvo:", saida)
print("\nEXPERIMENTO CONCLUÍDO.")