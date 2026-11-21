
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


BASE = Path(__file__).resolve().parent
PASTA_DADOS = BASE / "data" / "CIC-IDS2017"
ARQUIVO_TESTE = PASTA_DADOS / "cic_ids2017_teste.parquet"

MODELOS = {
    "Original — 100 árvores": {
        "arquivo": "detector_cic_ids2017.joblib",
        "excluir": ("Label", "grupo_duplicata"),
    },
    "Sem porta — 100 árvores": {
        "arquivo": "detector_cic_ids2017_sem_porta.joblib",
        "excluir": (
            "Label", "grupo_duplicata", "Destination Port"
        ),
    },
    "Sem porta e janelas — 100 árvores": {
        "arquivo": "detector_cic_ids2017_sem_porta_sem_janelas.joblib",
        "excluir": (
            "Label",
            "grupo_duplicata",
            "Destination Port",
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
        ),
    },
    "Sem porta e janelas — 150 árvores": {
        "arquivo": "detector_cic_ids2017_sem_porta_sem_janelas_150.joblib",
        "excluir": (
            "Label",
            "grupo_duplicata",
            "Destination Port",
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
        ),
    },
}


st.set_page_config(
    page_title="Detector de Botnet",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def carregar_teste():
    return pd.read_parquet(ARQUIVO_TESTE)


@st.cache_data(show_spinner=False)
def avaliar_modelo(caminho_modelo, colunas_excluir):
    df = carregar_teste()

    modelo = joblib.load(caminho_modelo)

    X = df.drop(columns=list(colunas_excluir))
    y = df["Label"].astype(str).to_numpy()

    previsoes = modelo.predict(X)

    classificador = modelo.named_steps["classificador"]
    classes = list(classificador.classes_)
    indice_bot = classes.index("Bot")

    probabilidades = modelo.predict_proba(X)[:, indice_bot]

    matriz = confusion_matrix(
        y,
        previsoes,
        labels=["BENIGN", "Bot"],
    )

    tn, fp, fn, tp = matriz.ravel()

    relatorio = classification_report(
        y,
        previsoes,
        labels=["BENIGN", "Bot"],
        output_dict=True,
        zero_division=0,
    )

    metricas = {
        "Precisão": relatorio["Bot"]["precision"] * 100,
        "Recall": relatorio["Bot"]["recall"] * 100,
        "F1-score": relatorio["Bot"]["f1-score"] * 100,
        "Acurácia": accuracy_score(y, previsoes) * 100,
        "Average Precision": average_precision_score(
            y == "Bot",
            probabilidades,
        ) * 100,
        "Falsos positivos": int(fp),
        "Bots perdidos": int(fn),
        "Bots detectados": int(tp),
        "Verdadeiros negativos": int(tn),
    }

    # Registros classificados incorretamente
    contexto = [
        "Destination Port",
        "Flow Duration",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
    ]

    contexto = [c for c in contexto if c in df.columns]

    erros = df[contexto].copy()
    erros["Real"] = y
    erros["Previsto"] = previsoes
    erros["Probabilidade Bot"] = probabilidades

    falsos_positivos = erros[
        (erros["Real"] == "BENIGN") &
        (erros["Previsto"] == "Bot")
    ]

    bots_perdidos = erros[
        (erros["Real"] == "Bot") &
        (erros["Previsto"] == "BENIGN")
    ]

    return {
        "metricas": metricas,
        "matriz": matriz,
        "erros_fp": falsos_positivos,
        "erros_fn": bots_perdidos,
    }


# Verificação dos arquivos
arquivos_necessarios = [
    ARQUIVO_TESTE,
    *[
        BASE / "models" / config["arquivo"]
        for config in MODELOS.values()
    ],
]

ausentes = [str(p) for p in arquivos_necessarios if not p.exists()]

if ausentes:
    st.error("Arquivos necessários não encontrados:")
    for arquivo in ausentes:
        st.write(arquivo)
    st.stop()


# Cabeçalho
st.title("🛡️ Detector de Botnet")
st.caption("Experimento de Machine Learning — CIC-IDS2017 / ARES")

st.warning(
    "Avaliação realizada em uma divisão da mesma captura ARES. "
    "Os resultados não comprovam generalização para outras redes "
    "ou famílias de botnet."
)


# Dados gerais
df_teste = carregar_teste()
distribuicao = df_teste["Label"].value_counts()

col1, col2, col3 = st.columns(3)

col1.metric("Registros de teste", f"{len(df_teste):,}")
col2.metric("Tráfego benigno", f"{distribuicao.get('BENIGN', 0):,}")
col3.metric("Bots reais", f"{distribuicao.get('Bot', 0):,}")

st.divider()


# Seleção do modelo
st.subheader("Análise do modelo")

nome_modelo = st.selectbox(
    "Selecione o modelo",
    list(MODELOS.keys()),
)

config = MODELOS[nome_modelo]
caminho_modelo = BASE / "models" / config["arquivo"]

with st.spinner("Calculando previsões..."):
    resultado = avaliar_modelo(
        str(caminho_modelo),
        config["excluir"],
    )

m = resultado["metricas"]


# Indicadores
c1, c2, c3, c4 = st.columns(4)

c1.metric("Precisão Bot", f"{m['Precisão']:.2f}%")
c2.metric("Recall Bot", f"{m['Recall']:.2f}%")
c3.metric("F1-score", f"{m['F1-score']:.2f}%")
c4.metric("Average Precision", f"{m['Average Precision']:.2f}%")

st.divider()

# Matriz e resultados
esquerda, direita = st.columns(2)

with esquerda:
    st.subheader("Matriz de confusão")

    matriz_df = pd.DataFrame(
        resultado["matriz"],
        index=["Real BENIGN", "Real Bot"],
        columns=["Previsto BENIGN", "Previsto Bot"],
    )

    st.dataframe(matriz_df, use_container_width=True)

with direita:
    st.subheader("Resumo das detecções")

    st.metric("Bots detectados", m["Bots detectados"])
    st.metric("Bots perdidos", m["Bots perdidos"])
    st.metric("Falsos positivos", m["Falsos positivos"])
    st.metric("Verdadeiros negativos", m["Verdadeiros negativos"])

st.divider()


# Comparação dos modelos
st.subheader("Comparação entre os quatro modelos")

st.write(
    "Execute a comparação para consultar os resultados "
    "de todos os modelos no mesmo conjunto de teste."
)

if st.button("Comparar todos os modelos"):
    linhas = []

    with st.spinner("Comparando modelos..."):
        for nome, config in MODELOS.items():

            caminho = BASE / "models" / config["arquivo"]

            res = avaliar_modelo(
                str(caminho),
                config["excluir"],
            )

            metricas = res["metricas"]

            linhas.append({
                "Modelo": nome,
                "Precisão": round(metricas["Precisão"], 2),
                "Recall": round(metricas["Recall"], 2),
                "F1": round(metricas["F1-score"], 2),
                "Average Precision": round(
                    metricas["Average Precision"], 2
                ),
                "Falsos positivos": metricas["Falsos positivos"],
                "Bots perdidos": metricas["Bots perdidos"],
            })

    tabela = pd.DataFrame(linhas)

    st.dataframe(
        tabela,
        use_container_width=True,
        hide_index=True,
    )


# Investigação dos erros
st.divider()

st.subheader("Investigação dos erros")

aba_fp, aba_fn = st.tabs([
    "Falsos positivos",
    "Bots perdidos",
])

with aba_fp:
    st.write(
        f"Total de falsos positivos: {len(resultado['erros_fp'])}"
    )

    st.dataframe(
        resultado["erros_fp"].head(100),
        use_container_width=True,
        hide_index=True,
    )

with aba_fn:
    st.write(
        f"Total de bots perdidos: {len(resultado['erros_fn'])}"
    )

    st.dataframe(
        resultado["erros_fn"].head(100),
        use_container_width=True,
        hide_index=True,
    )


st.divider()

st.caption(
    "Projeto educacional de detecção de tráfego malicioso. "
    "Os resultados são experimentais e não representam "
    "um sistema de segurança validado para produção."
)
