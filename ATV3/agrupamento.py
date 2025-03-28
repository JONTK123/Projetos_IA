
REESCOLHER O ALGORITMO PARA CADA DATASET DPS DE TRATAR... OLHAR REVISAO NO NOTION
1- TESTAR TODOS HIERARQUICOS -> VER QUAL FICA DIVIDIDINHO BONITINHO PARA AS DUAS BASES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.datasets import load_iris
import kagglehub

# -----------------------
# 1) CARREGANDO OS DADOS
# -----------------------
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
df_kaggle_original = pd.read_csv(f"{path}/tmdb_5000_movies.csv")
df_kaggle_original = df_kaggle_original[['budget', 'popularity', 'revenue', 'vote_average', 'vote_count']].dropna()

# --------------------------------------------------------
# 2) VISUALIZAÇÃO EM PARES (SEM NORMALIZAÇÃO)
# --------------------------------------------------------
sns.pairplot(df_iris)
plt.suptitle("Iris - Pairplot (Sem Normalização)", y=1.02)
plt.show()

sns.pairplot(df_kaggle_original)
plt.suptitle("Kaggle (Original) - Pairplot (Sem Normalização)", y=1.02)
plt.show()

# --------------------------------------------------------
# 3) NORMALIZAÇÃO DAS BASES (STANDARD SCALER)
# --------------------------------------------------------
scaler_iris = StandardScaler()
iris_scaled = scaler_iris.fit_transform(df_iris)
df_iris_scaled = pd.DataFrame(iris_scaled, columns=df_iris.columns)

scaler_kaggle = StandardScaler()
kaggle_scaled = scaler_kaggle.fit_transform(df_kaggle_original)
df_kaggle_scaled = pd.DataFrame(kaggle_scaled, columns=df_kaggle_original.columns)

# --------------------------------------------------------
# 4) POWERTRANSFORMER APENAS NO KAGGLE
# --------------------------------------------------------
pt = PowerTransformer(method='yeo-johnson')
kaggle_transformed = pt.fit_transform(df_kaggle_scaled)
df_kaggle_transformed = pd.DataFrame(kaggle_transformed, columns=df_kaggle_original.columns)

# -------------------------
# PLOTS COM NORMALIZAÇÃO
# -------------------------
sns.pairplot(df_iris_scaled)
plt.suptitle("Iris - Pairplot (Apenas Normalizado)", y=1.02)
plt.show()

sns.pairplot(df_kaggle_scaled)
plt.suptitle("Kaggle - Pairplot (Apenas Normalizado)", y=1.02)
plt.show()

sns.pairplot(df_kaggle_transformed)
plt.suptitle("Kaggle - Pairplot (StandardScaler + PowerTransformer)", y=1.02)
plt.show()

# --------------------------------------------------------
# 5) ELBOW METHOD PARA DEFINIR K (3 CASOS)
# --------------------------------------------------------
def elbow_method(data, title):
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title(f"Elbow - {title}")
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('WCSS')
    plt.show()

elbow_method(iris_scaled, "Iris (Normalizado)")
elbow_method(kaggle_scaled, "Kaggle (Apenas Normalizado)")
elbow_method(kaggle_transformed, "Kaggle (Normalizado + PowerTransformer)")

print("Analise os gráficos Elbow e escolha K para cada base.")
k_iris = int(input("Escolha K para IRIS: "))
k_kaggle = int(input("Escolha K para KAGGLE: "))

# --------------------------------------------------------
# 6) K-MEANS E HIERÁRQUICO (WARD)
# --------------------------------------------------------
kmeans_iris = KMeans(n_clusters=k_iris, random_state=42, n_init=10)
iris_clusters_kmeans = kmeans_iris.fit_predict(iris_scaled)

kmeans_kaggle_scaled = KMeans(n_clusters=k_kaggle, random_state=42, n_init=10)
kaggle_scaled_clusters_kmeans = kmeans_kaggle_scaled.fit_predict(kaggle_scaled)

kmeans_kaggle_transformed = KMeans(n_clusters=k_kaggle, random_state=42, n_init=10)
kaggle_transformed_clusters_kmeans = kmeans_kaggle_transformed.fit_predict(kaggle_transformed)

def hierarchical_clustering(data, title, k, method='ward'):
    linked = linkage(data, method=method)
    plt.figure(figsize=(8, 4))
    dendrogram(linked)
    plt.title(f"Dendrograma ({method}) - {title}")
    plt.xlabel('Amostras')
    plt.ylabel('Distância')
    plt.show()
    return fcluster(linked, t=k, criterion='maxclust')

iris_clusters_hier = hierarchical_clustering(iris_scaled, "Iris (Normalizado)", k_iris, 'ward')
kaggle_scaled_clusters_hier = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", k_kaggle, 'ward')
kaggle_transformed_clusters_hier = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)", k_kaggle, 'ward')

# --------------------------------------------------------
# 7) SILHOUETTE SCORE
# --------------------------------------------------------
score_kmeans_iris = silhouette_score(iris_scaled, iris_clusters_kmeans)
score_hier_iris   = silhouette_score(iris_scaled, iris_clusters_hier)

score_kmeans_kg_scaled = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_kmeans)
score_hier_kg_scaled   = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_hier)

score_kmeans_kg_transf = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_kmeans)
score_hier_kg_transf   = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_hier)

print("\n===== COMPARATIVO SILHOUETTE SCORE =====\n")
print("Base IRIS (apenas normalizada)")
print(f"  • K-Means:              {score_kmeans_iris:.4f}")
print(f"  • Hierárquico (Ward):   {score_hier_iris:.4f}\n")

print("Base KAGGLE (apenas normalizada)")
print(f"  • K-Means:              {score_kmeans_kg_scaled:.4f}")
print(f"  • Hierárquico (Ward):   {score_hier_kg_scaled:.4f}\n")

print("Base KAGGLE (StandardScaler + PowerTransformer)")
print(f"  • K-Means:              {score_kmeans_kg_transf:.4f}")
print(f"  • Hierárquico (Ward):   {score_hier_kg_transf:.4f}")

# --------------------------------------------------------
# RELATÓRIO FINAL (DETALHES, ESCOLHAS E JUSTIFICATIVAS)
# --------------------------------------------------------
'''
RELATÓRIO FINAL (VERSÃO APROFUNDADA):

1) DATASETS E DIMENSÕES:

   a) Base Iris:
      - Possui 4 dimensões: sepal length, sepal width, petal length e petal width.
      - São 150 instâncias (amostras), normalmente classificadas em 3 espécies de flor (Setosa, Versicolor, Virginica).
      - Aqui, o objetivo é clustering. isso significa que estamos fazendo Aprendizado de Máquina Não Supervisionado, ou seja,
        não usamos os rótulos originais (as espécies) para treinar, mas apenas para comparar ou avaliar depois, se quisermos.

   b) Base Kaggle (TMDB):
      - Selecionamos 5 dimensões numéricas: budget, popularity, revenue, vote_average e vote_count.
      - O método dropna() remove linhas com valores ausentes (NaNs). Isso garante que só tenhamos dados completos para cada coluna.
      - Escolhemos essas 5 dimensões porque são variáveis numéricas relacionadas ao desempenho e métricas de filmes
        (ex.: dinheiro investido, receita, popularidade e avaliações). Outras colunas poderiam estar muito correlacionadas
        ou sem relevância numérica, então optamos por essas 5 para simplificar o modelo.

2) POR QUE NORMALIZAMOS?

   - Tanto em Iris quanto em Kaggle, usamos algoritmos de clustering baseados em distância Euclidiana (muitos teoremas de pitágoras...), como o K-Means e o Ward.
   - A distância Euclidiana calcula a "hipotenusa" entre pontos no espaço n-dimensional:
       d(p, q) = sqrt( (p1 - q1)^2 + (p2 - q2)^2 + ... + (pn - qn)^2 )
   - Se uma variável tem valores muito maiores que as outras (por exemplo, budget em milhões vs. vote_average em 0~10), ela domina o cálculo de distância.
   - Para evitar isso, aplicamos StandardScaler, que transforma cada coluna para ter média 0 e desvio padrão 1, deixando-as com “peso” equilibrado.
   - O StandardScaler é fornecido pela biblioteca scikit-learn (sklearn.preprocessing). Ele é comum em pipelines de machine learning.
   - Apesar das variáveis do Iris estarem em escalas parecidas (todas em centímetros), **a diferença entre seus desvios padrões ainda pode distorcer distâncias**,
    o que afeta negativamente o clustering. Por isso, normalizamos também (petal width varia muito menos que petal length, 2.5cm para 7cm).

3) POR QUE APLICAMOS POWERTRANSFORMER NO KAGGLE?

   - Mesmo após normalizar (StandardScaler), as variáveis budget e revenue, por exemplo, mantêm distribuições muito assimétricas (skewed).
   - Skewness (assimetria) indica que a maior parte dos valores se concentra em uma faixa pequena, enquanto há caudas muito longas em um lado.
   - O PowerTransformer, também da biblioteca scikit-learn, método 'yeo-johnson', tenta corrigir essa assimetria aplicando transformações matemáticas
     em cada coluna, aproximando-a de uma distribuição Gaussiana (ou "normal").
   - Gaussiana (ou distribuição normal) é aquela curva em formato de sino (bell curve), centrada na média, com simetria e uma variância bem definida.
   - Algoritmos como K-Means e Ward “assumem” clusters mais ou menos esféricos; se os dados estiverem muito alongados ou com outliers extremos,
     o centróide ou a variância intracluster fica distorcida.
   - Escolhemos 'yeo-johnson' porque ele funciona com valores negativos, o que pode ocorrer após a normalização. 
     O método 'box-cox' foi descartado porque só funciona com dados estritamente positivos.
     • PowerTransformer (Yeo-Johnson) — útil para reduzir skewness e tornar os dados mais gaussianos, mesmo com valores negativos. Aplica transformações logarítmicas ou de potência.
     • PowerTransformer (Box-Cox) — também reduz skewness, similar ao yeo, mas só funciona com valores estritamente positivos. 
   - Outros métodos que poderiam ser considerados:
     • Log1p (log(1 + x)) — simples, mas não corrige bem valores negativos.
     • QuantileTransformer — força os dados a uma distribuição desejada, mas distorce relações locais.
     • RobustScaler — útil para outliers, mas não resolve skewness.
   - O Yeo-Johnson foi a melhor escolha aqui por ser versátil, eficaz e automático para cada coluna.

4) MÉTODO DO JOELHO (ELBOW METHOD):

   - O Elbow Method é usado para sugerir o número adequado de clusters (K) num algoritmo como K-Means.
   - Calcula-se a soma das distâncias (WCSS) de cada ponto ao seu centróide em função de K. Normalmente, conforme K cresce, o WCSS diminui,
     mas tende a “estabilizar” após certo ponto. Esse ponto de “curva” ou “joelho” indica um bom equilíbrio entre coesão intracluster e complexidade.

5) ALGORITMOS DE AGRUPAMENTO:

   - K-Means: biblioteca scikit-learn (sklearn.cluster.KMeans).
     - Ele aceita parâmetros como:
       n_clusters (quantidade de grupos),
       init (método de inicialização dos centróides),
       n_init (quantas vezes ele repete o processo para evitar mínimo local),
       random_state (para reprodutibilidade) e outros.
     - Escolhemos n_init=10 para rodar várias inicializações e pegar o melhor resultado
       (evitando cair em um centróide ruim).
     - random_state=42 é só para garantir que o resultado seja reproduzível em diferentes execuções.

   - Hierarchical Clustering (Ward):
     - Usamos linkage='ward', que minimiza a variância dentro de cada cluster a cada fusão.
     • Optamos pelo método 'ward' porque ele minimiza a variância intracluster a cada fusão,
       garantindo que os clusters formados sejam mais compactos e homogêneos internamente.
       Ele calcula a soma dos quadrados das diferenças dentro dos grupos (similar ao K-Means),
       tornando os resultados mais consistentes e comparáveis.
     • Esse método funciona especialmente bem em dados que foram normalizados e transformados
     para se aproximarem de uma distribuição gaussiana — exatamente o caso do nosso pipeline
     com StandardScaler e PowerTransformer.
     • Comparando com outros métodos:
       - 'single' usa a menor distância entre pontos, mas é muito sensível a outliers e tende
          a formar clusters "esticados" ou encadeados.
       - 'complete' considera a maior distância entre pontos, o que pode exagerar a separação
          e formar clusters pequenos e distantes.
       - 'average' usa a média das distâncias entre todos os pares, mas não leva em conta a
          variância interna como o 'ward' faz.
          
6) SILHOUETTE SCORE:

   - É uma métrica interna de validação de clusters, também disponibilizada pelo scikit-learn (sklearn.metrics.silhouette_score).
   - Para cada ponto, mede a coesão com seu cluster e a separação em relação aos outros clusters, gerando valores entre -1 e +1.
   - Quanto mais próximo de +1, mais satisfatória a separação; valores negativos sugerem pontos “mal clusterizados”.
   - Usamos para comparar rapidamente a qualidade dos grupos gerados por K-Means e Hierarchical nas diferentes versões: “sem transformar”, “só normalizar” e “normalizar + PowerTransformer”.

CONCLUSÃO SOBRE O PROJETO E ATIVIDADE:

- O projeto aplicou dois algoritmos (K-Means e Hierárquico) em duas bases (Iris e Kaggle).
- Em Iris, normalização já é suficiente devido às escalas moderadamente diferentes (4 ~ 8 cm vs. 0.1 ~ 2.5 cm).
- Em Kaggle, as diferenças de escalas e a skewness das variáveis levaram a uma grande melhora após usar “StandardScaler + PowerTransformer”.
- O Elbow Method ajudou a escolher K, e a métrica Silhouette Score validou a qualidade das partições.
- Ward foi preferido no hierárquico por seu critério de minimização da variância, gerando clusters mais coesos.
- Tudo se encaixa nos requisitos da atividade: experimentamos diferentes parâmetros, discutimos resultados, e apresentamos gráficos e métricas para confirmar as escolhas.

- AGORA SO FALTA COLOCAR PRINTS NO RELATORIO FINAL E DESENVOLVER UM POUCO O TEXTO MAS A IDEIA ESTA AI, DESENVOLVER CADA DETALHE
'''

