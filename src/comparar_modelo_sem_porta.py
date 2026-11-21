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

TREINO = pd.read_parquet(PASTA / "cic_ids2017_treino.parquet")
TESTE = pd.read_parquet(PASTA / "cic_ids2017_teste.parquet")

y_treino = TREINO["Label"].astype(str)
y_teste = TESTE["Label"].astype(str)

# Modelo original: mantém todas as características
X_treino_original = TREINO.drop(
    columns=["Label", "grupo_duplicata"]
)
X_teste_original = TESTE.drop(
    columns=["Label", "grupo_duplicata"]
)

# Novo modelo: remove Destination Port
X_treino_sem_porta = X_treino_original.drop(
    columns=["Destination Port"]
)
X_teste_sem_porta = X_teste_original.drop(
    columns=["Destination Port"]
)

print("Treinando modelo sem Destination Port...")

modelo_sem_porta = Pipeline([
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

modelo_sem_porta.fit(X_treino_sem_porta, y_treino)

print("Treinamento concluído.")

# Carrega o modelo original
modelo_original = joblib.load(
    "models/detector_cic_ids2017.joblib"
)


def avaliar(nome, modelo, X):
    previsoes = modelo.predict(X)

    classes = list(modelo.named_steps["classificador"].classes_)
    indice_bot = classes.index("Bot")

    probabilidades = modelo.predict_proba(X)[:, indice_bot]

    matriz = confusion_matrix(
        y_teste,
        previsoes,
        labels=["BENIGN", "Bot"],
    )

    tn, fp, fn, tp = matriz.ravel()

    relatorio = classification_report(
        y_teste,
        previsoes,
        labels=["BENIGN", "Bot"],
        output_dict=True,
        zero_division=0,
    )

    print("\n" + "=" * 55)
    print(nome)
    print("=" * 55)

    print("Matriz de confusão:")
    print(pd.DataFrame(
        matriz,
        index=["Real BENIGN", "Real Bot"],
        columns=["Previsto BENIGN", "Previsto Bot"],
    ).to_string())

    print("\nPrecisão Bot:", round(
        relatorio["Bot"]["precision"] * 100, 2
    ), "%")

    print("Recall Bot:", round(
        relatorio["Bot"]["recall"] * 100, 2
    ), "%")

    print("F1 Bot:", round(
        relatorio["Bot"]["f1-score"] * 100, 2
    ), "%")

    print("Average Precision:", round(
        average_precision_score(
            (y_teste == "Bot").astype(int),
            probabilidades,
        ) * 100, 2
    ), "%")

    print("Falsos positivos:", fp)
    print("Bots perdidos:", fn)


avaliar(
    "MODELO ORIGINAL — COM PORTA",
    modelo_original,
    X_teste_original,
)

avaliar(
    "MODELO NOVO — SEM PORTA",
    modelo_sem_porta,
    X_teste_sem_porta,
)

saida = Path("models/detector_cic_ids2017_sem_porta.joblib")
joblib.dump(modelo_sem_porta, saida)

print("\nModelo sem porta salvo em:", saida)
print("\nCOMPARAÇÃO CONCLUÍDA.")