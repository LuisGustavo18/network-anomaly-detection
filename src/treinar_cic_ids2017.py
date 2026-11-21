from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ARQUIVO = Path("data/CIC-IDS2017/cic_ids2017_treino.parquet")
SAIDA = Path("models/detector_cic_ids2017.joblib")

print("Carregando treinamento...")

df = pd.read_parquet(ARQUIVO)

# Identifica as características e o alvo
X = df.drop(columns=["Label", "grupo_duplicata"])
y = df["Label"].astype(str)

# Confere os tipos de dados
nao_numericas = X.select_dtypes(exclude="number").columns

if len(nao_numericas) > 0:
    raise ValueError(
        f"Características não numéricas encontradas: "
        f"{nao_numericas.tolist()}"
    )

print("\nREGISTROS:", len(df))
print("\nDISTRIBUIÇÃO DAS CLASSES:")
print(y.value_counts())

print("\nCARACTERÍSTICAS:", len(X.columns))
print("Valores ausentes:", X.isna().sum().sum())

modelo = Pipeline([
    (
        "imputador",
        SimpleImputer(
            strategy="median",
            add_indicator=True,
        ),
    ),
    (
        "classificador",
        RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

print("\nIniciando treinamento...")

modelo.fit(X, y)

print("\nTreinamento concluído.")

print("\nClasses aprendidas:")
print(modelo.named_steps["classificador"].classes_)

print("\nSalvando modelo...")

SAIDA.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(modelo, SAIDA)

print("\nMODELO SALVO:", SAIDA)
print("\nTREINAMENTO FINALIZADO.")