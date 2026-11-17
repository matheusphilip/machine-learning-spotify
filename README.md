# 🎵 Predição de Hits no Spotify com Machine Learning

Projeto desenvolvido durante o **#7DaysOfCode de Machine Learning (Alura)** com o objetivo de prever faixas com alto potencial de popularidade no Spotify usando exclusivamente atributos acústicos e musicais extraídos da plataforma.

---

## 🎯 Problema de Negócio e Desafio Técnico

Identificar músicas com potencial de sucesso comercial é um problema clássico de **classificação binária severamente desbalanceada**: a grande maioria do catálogo não atinge o patamar de popularidade.

Além do desbalanceamento natural, o dataset apresenta um desafio crítico de integridade: **faixas duplicadas catalogadas em múltiplos gêneros musicais ou relançadas em álbuns/coletâneas diferentes**. Sem o tratamento adequado antes da divisão dos dados, modelos baseados em árvores memorizam registros idênticos entre treino e teste, gerando métricas artificialmente infladas que não se sustentam em produção.

---

## 🛠️ Stack Tecnológica

* **Linguagem:** Python 3
* **Manipulação e Análise de Dados:** `pandas`, `NumPy`
* **Visualização:** `Matplotlib`, `Seaborn`
* **Machine Learning e Pipelines:** `scikit-learn`, `imbalanced-learn`
* **Serialização:** `joblib`

---

## 🔍 Análise Exploratória e Engenharia de Dados

* **Deduplicação Inteligente:** Remoção de mais de 30.000 registros duplicados com base na combinação `['artists', 'track_name']` antes do `train_test_split`, evitando vazamento de faixas idênticas entre treino e teste.
* **Análise Granular de Artistas e Gêneros:** Separação de colaborações musicais (`explode`) para cálculo justo da popularidade média por artista (filtrando artistas com pelo menos 2 faixas e popularidade > 0) e análise de gêneros sobre a base bruta para preservar a representatividade de faixas multigênero.
* **Correlação Acústica:** Identificação de forte correlação positiva entre `loudness` e `energy`, e correlação negativa com `acousticness`, além da baixa correlação linear direta entre atributos sonoros isolados e a popularidade comercial.

---

## ⚙️ Metodologia e Rigor Experimental

1. **Estabelecimento de Baseline:**
   * Treinamento inicial com `LogisticRegression` (`class_weight='balanced'`) utilizando `StratifiedKFold` (5 *folds*) e normalização `MinMaxScaler` ajustada exclusivamente dentro de cada dobra de treino.
2. **Comparação de Técnicas de Reamostragem e Modelos:**
   * Avaliação cruzada combinando `RandomUnderSampler` (RUS), `RandomOverSampler` (ROS) e `SMOTE` com `LogisticRegression`, `DecisionTreeClassifier` e `RandomForestClassifier`.
3. **Otimização com `Pipeline` Blindado:**
   * Encapsulamento do `MinMaxScaler`, `RandomUnderSampler` e `RandomForestClassifier` em um `imblearn.pipeline.Pipeline` otimizado via `RandomizedSearchCV` (20 iterações, 5 *folds* estratificados).
   * **Escolha Técnica (RUS vs. SMOTE):** Embora o `SMOTE` tenha alcançado `F1 = 0.1281` na validação cruzada contra `0.1183` do `RUS`, o custo computacional do `SMOTE` foi muito superior para um ganho marginal de `0.0098`. No final, o **RUS** foi escolhido pela eficiência operacional e estabilidade.

---

## 📊 Resultados Finais (`X_test`)

Após o tuning de hiperparâmetros, o pipeline vencedor foi retreinado em 100% do conjunto de treino (`X_train`) e avaliado uma única vez no conjunto de teste inédito (`X_test`):

| Etapa de Avaliação | F1-Score (Classe Hit) | Observação |
| :--- | :---: | :--- |
| **Validação Cruzada (Média 5-Folds)** | `0.1183` | Desempenho médio durante o `RandomizedSearchCV` |
| **Conjunto de Teste Inédito (`X_test`)** | **`0.1207`** | Generalização consistente sem sinais de *overfitting* |

### Por que 0.1207 e não 0.90+?
Abordagens ingênuas neste dataset podem reportar F1-Scores acima de `0.90` ao aplicar *oversampling* (`ROS`/`SMOTE`) antes do `train_test_split` ou ao manter faixas duplicadas na base. Essa prática clona músicas idênticas entre treino e teste, fazendo com que árvores de decisão apenas decorem os registros. O resultado de **0.1207** representa a **capacidade real de generalização** em faixas inéditas sob a distribuição desbalanceada do mundo real.

### Visão de Negócio
* **Perfil de Triagem (Alto Recall vs. Baixa Precisão):** O balanceamento 50/50 no treino orientou o modelo a priorizar a captura de potenciais hits (*Recall* elevado), aceitando mais falsos positivos na base real (*Precisão* reduzida). Na prática, o modelo funciona como um **filtro inicial de descoberta** para equipes de curadoria musical.
* **Teto Preditivo das Variáveis Acústicas:** Atributos puramente sonoros (`danceability`, `energy`, `loudness`, `valence`, etc.) explicam apenas parte do sucesso comercial. Fatores externos decisivos — como investimento em marketing, playlists editoriais, base prévia de fãs e viralização em redes sociais — não constam no dataset e impõem um limite natural ao poder preditivo do áudio isolado.

---

## 📁 Arquivos e Uso do Modelo

* `spotify.ipynb`: Notebook completo com limpeza de dados, análise exploratória, validação cruzada e tuning do modelo.
* `modelo_spotify_previsao_popular_rf.joblib`: Pipeline final serializado contendo o `MinMaxScaler` e o `RandomForestClassifier` treinados.

### Fazendo previsões com o modelo salvo
Como o arquivo `.joblib` já encapsula o `Pipeline` completo, basta passar um DataFrame com os 10 atributos sonoros na escala original (`duration_ms`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence` e `tempo`):

```python
import joblib

modelo = joblib.load('modelo_spotify_previsao_popular_rf.joblib')
previsoes = modelo.predict(novas_musicas)
