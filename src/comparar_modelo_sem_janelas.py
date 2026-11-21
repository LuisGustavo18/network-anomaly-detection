from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


PASTA = Path("data/CIC-IDS2017")

TREINO = pd.read_parquet(
    PASTA / "cic_ids2017_treino.parquet"
)

TESTE = pd.read_parquet(
    PASTA / "cic_ids2017_teste.parquet"
)

y_treino = TREINO["Label"].astype(str)
y_teste = TESTE["Label"].astype(str)

# Remove campos auxiliares e porta de destino
COLUNAS_REMOVER = [
    "Label",
    "grupo_duplicata",
    "Destination Port",
]

X_treino = TREINO.drop(columns=COLUNAS_REMOVER)
X_teste = TESTE.drop(columns=COLUNAS_REMOVER)

# Remove as duas características de janela TCP
JANELAS = [
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

X_treino = X_treino.drop(columns=JANELAS)
X_teste = X_teste.drop(columns=JANELAS)

print("CARACTERÍSTICAS UTILIZADAS:", len(X_treino.columns))
print("Registros de treinamento:", len(X_treino))
print("Registros de teste:", len(X_teste))

print("\nTreinando modelo sem porta e sem janelas TCP...")

modelo = Pipeline([
    ("imputador", SimpleImputer(
        strategy="median",
        add_indicator=True,
    )),
    ("classificador", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )),
])

modelo.fit(X_treino, y_treino)

print("Treinamento concluído.")

# Avaliação
previsoes = modelo.predict(X_teste)

classes = list(modelo.named_steps["classificador"].classes_)
indice_bot = classes.index("Bot")

probabilidades = modelo.predict_proba(X_teste)[:, indice_bot]

matriz = confusion_matrix(
    y_teste,
    previsoes,
    labels=["BENIGN", "Bot"],
)

tn, fp, fn, tp = matriz.ravel()

print("\n" + "=" * 55)
print("RESULTADO — SEM PORTA E SEM JANELAS TCP")
print("=" * 55)

print(pd.DataFrame(
    matriz,
    index=["Real BENIGN", "Real Bot"],
    columns=["Previsto BENIGN", "Previsto Bot"],
).to_string())

print("\nRELATÓRIO:")
print(classification_report(
    y_teste,
    previsoes,
    labels=["BENIGN", "Bot"],
    zero_division=0,
    digits=4,
))

print(
    "Average Precision:",
    round(
        average_precision_score(
            y_teste == "Bot",
            probabilidades,
        ) * 100,
        2,
    ),
    "%",
)

print("\nCONTAGEM:")
print("Bots reais:", tp + fn)
print("Detectados:", tp)
print("Perdidos:", fn)
print("Falsos positivos:", fp)

# Salva o novo modelo
saida = Path("models/detector_cic_ids2017_sem_porta_sem_janelas.joblib")

saida.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(modelo, saida)

print("\nMODELO SALVO:", saida)
print("\nEXPERIMENTO CONCLUÍDO.")