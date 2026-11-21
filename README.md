\# 🛡️ Network Anomaly Detection — Botnet ARES



Projeto educacional de cibersegurança que utiliza Machine Learning para identificar tráfego de rede associado à botnet ARES.



Desenvolvido em Python, utilizando Random Forest e o dataset CIC-IDS2017.



O projeto também possui um dashboard interativo desenvolvido com Streamlit para visualizar resultados, métricas e erros de classificação.



\## Objetivos



\- Estudar Machine Learning aplicado à cibersegurança.

\- Identificar padrões de tráfego associados a botnets.

\- Investigar falsos positivos e falsos negativos.

\- Avaliar o impacto das características de rede no modelo.

\- Desenvolver uma interface para visualizar os resultados.



\## Tecnologias



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- PyArrow

\- Joblib

\- Streamlit

\- Random Forest



\## Dataset



Foi utilizado o CIC-IDS2017, especificamente o arquivo:



`Friday-WorkingHours-Morning.pcap\_ISCX.csv`



Essa captura contém tráfego benigno e tráfego associado à botnet ARES.



Distribuição inicial:



| Classe | Registros | Percentual |

|---|---:|---:|

| BENIGN | 189.067 | 98,97% |

| Bot | 1.966 | 1,03% |

| Total | 191.033 | 100% |



O conjunto foi dividido utilizando StratifiedGroupKFold, com cinco divisões e agrupamento de registros com características idênticas.



| Conjunto | Registros |

|---|---:|

| Treinamento | 152.826 |

| Teste | 38.207 |



Não foram encontrados grupos duplicados compartilhados entre treinamento e teste.



\## Experimentos



O algoritmo utilizado foi Random Forest Classifier.



Foram testadas quatro configurações:



1\. Modelo original: 61 características e 100 árvores.

2\. Modelo sem a característica Destination Port.

3\. Modelo sem Destination Port e sem as duas características de janela TCP, com 100 árvores.

4\. Mesmo conjunto reduzido de características, com 150 árvores.



\### Resultados



| Modelo | Precisão Bot | Recall Bot | F1 Bot | Falsos positivos |

|---|---:|---:|---:|---:|

| Original | 99,74% | 95,93% | 97,80% | 1 |

| Sem porta | 98,44% | 96,18% | 97,30% | 6 |

| Sem porta e janelas (100) | 29,49% | 95,67% | 45,08% | 899 |

| Sem porta e janelas (150) | 29,56% | 95,67% | 45,17% | 896 |



\### Principais descobertas



\- A remoção da porta de destino não causou uma queda expressiva no desempenho.

\- As características de janela TCP tiveram grande influência na separação entre tráfego benigno e malicioso.

\- A remoção das características de janela aumentou consideravelmente os falsos positivos.

\- Aumentar o número de árvores de 100 para 150 trouxe pouca diferença.



Os experimentos demonstraram que a escolha das características influencia tanto a detecção quanto a quantidade de falsos alarmes.



\## Dashboard



Foi desenvolvido um dashboard utilizando Streamlit.



Funcionalidades:



\- Seleção entre os modelos treinados.

\- Visualização de precisão, recall e F1-score.

\- Matriz de confusão.

\- Quantidade de bots detectados e perdidos.

\- Quantidade de falsos positivos.

\- Comparação entre os modelos.

\- Consulta dos erros de classificação.



Para executar:



```powershell

python -m streamlit run dashboard.py

```



\## Como executar o projeto



Clone o repositório e crie o ambiente virtual:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



Instale as dependências:



```powershell

pip install -r requirements.txt

```



O dataset e os arquivos de teste não estão incluídos no repositório. Eles precisam ser obtidos e preparados separadamente para executar o treinamento ou utilizar o dashboard.



Coloque os arquivos necessários nas pastas indicadas pelos scripts em `src/`.



\## Estrutura



```text

network-anomaly-detection/

│

├── data/                 # Dados locais, não publicados

├── models/               # Modelos CIC-IDS2017

├── src/                  # Scripts de auditoria e experimentação

├── dashboard.py          # Dashboard Streamlit

├── requirements.txt      # Dependências

├── .gitignore

└── README.md

```



\## Limitações



Este é um experimento educacional, não um sistema de segurança pronto para produção.



\- A avaliação foi realizada em uma divisão da mesma captura ARES.

\- O conjunto de teste foi reutilizado em análises e comparações sucessivas.

\- Ainda não foi demonstrada a generalização para outras famílias de botnets.

\- O modelo com características reduzidas apresentou muitos falsos positivos.

\- Não foram realizados testes em redes independentes ou em tempo real.



Os resultados não garantem o funcionamento do detector em ambientes reais.



\## Aprendizados



Durante o projeto foram estudados:



\- Análise de tráfego de rede.

\- Preparação e auditoria de datasets.

\- Tratamento de valores ausentes e infinitos.

\- Desequilíbrio de classes.

\- Prevenção de vazamento de dados.

\- Random Forest.

\- Precisão, recall, F1-score e Average Precision.

\- Permutation Importance.

\- Análise de falsos positivos e falsos negativos.

\- Desenvolvimento de dashboards com Streamlit.



\## Finalidade



Projeto desenvolvido para fins de estudo, prática de programação e construção de portfólio na área de cibersegurança.



\*\*Desenvolvimento experimental e educacional.\*\*