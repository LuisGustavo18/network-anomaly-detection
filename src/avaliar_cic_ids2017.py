from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


ARQUIVO_TESTE = Path(
    "data/CIC-IDS2017/cic_ids2017_teste.parquet"
)

ARQUIVO_MODELO = Path(
    "models/detector_cic_ids2017.joblib"
)

CLASSES = ["BENIGN", "Bot"]

print("Carregando modelo e conjunto de teste...")

modelo = joblib.load(ARQUIVO_MODELO)
df = pd.read_parquet(ARQUIVO_TESTE)

X_teste = df.drop(columns=["Label", "grupo_duplicata"])
y_real = df["Label"].astype(str)

print("\nRegistros de teste:", len(df))

print("\nDistribuição real:")
print(y_real.value_counts())

print("\nRealizando previsões...")

previsoes = modelo.predict(X_teste)

# Probabilidade atribuída à classe Bot
classificador = modelo.named_steps["classificador"]
indice_bot = list(classificador.classes_).index("Bot")

probabilidades = modelo.predict_proba(X_teste)[:, indice_bot]

# Matriz de confusão
matriz = confusion_matrix(
    y_real,
    previsoes,
    labels=CLASSES,
)

tn, fp, fn, tp = matriz.ravel()

print("\n" + "=" * 55)
print("MATRIZ DE CONFUSÃO")
print("=" * 55)

print("Linhas = classe real | Colunas = classe prevista")
print(pd.DataFrame(
    matriz,
    index=["Real BENIGN", "Real Bot"],
    columns=["Previsto BENIGN", "Previsto Bot"],
).to_string())

print("\n" + "=" * 55)
print("RELATÓRIO DE CLASSIFICAÇÃO")
print("=" * 55)

print(
    classification_report(
        y_real,
        previsoes,
        labels=CLASSES,
        zero_division=0,
        digits=4,
    )
)

print("=" * 55)
print("MÉTRICAS COMPLEMENTARES")
print("=" * 55)

print(f"Acurácia: {accuracy_score(y_real, previsoes) * 100:.2f}%")
print(
    "Average Precision (Bot): "
    f"{average_precision_score(y_real == 'Bot', probabilidades) * 100:.2f}%"
)

print("\nCONTAGEM DE DETECÇÕES")
print("Bots reais:", tp + fn)
print("Bots detectados:", tp)
print("Bots perdidos:", fn)
print("Falsos positivos:", fp)
print("Verdadeiros negativos:", tn)

# Baseline: classificar todos os registros como BENIGN
baseline = (y_real == "BENIGN").mean() * 100

print(f"\nAcurácia do baseline (sempre BENIGN): {baseline:.2f}%")

print("\nAVALIAÇÃO CONCLUÍDA.")