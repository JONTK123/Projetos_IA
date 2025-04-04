import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.datasets import load_iris
import kagglehub
import sys


# Aumentando a recursão para dendrogramas grandes
sys.setrecursionlimit(10000)

# -----------------------
# 1) CARREGANDO OS DADOS
# -----------------------
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

# Baixa e carrega dados do Kaggle (TMDB)
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
# 6) K-MEANS
# --------------------------------------------------------
kmeans_iris = KMeans(n_clusters=k_iris, random_state=42, n_init=10)
iris_clusters_kmeans = kmeans_iris.fit_predict(iris_scaled)

kmeans_kaggle_scaled = KMeans(n_clusters=k_kaggle, random_state=42, n_init=10)
kaggle_scaled_clusters_kmeans = kmeans_kaggle_scaled.fit_predict(kaggle_scaled)

kmeans_kaggle_transformed = KMeans(n_clusters=k_kaggle, random_state=42, n_init=10)
kaggle_transformed_clusters_kmeans = kmeans_kaggle_transformed.fit_predict(kaggle_transformed)

# Função de plotagem para Iris com K-Means
def plot_kmeans_iris(data_scaled, labels, title="K-Means - Iris"):
    df = pd.DataFrame(data_scaled, columns=iris.feature_names)
    df['cluster'] = labels.astype(str)
    sns.pairplot(df, hue='cluster')
    plt.suptitle(title, y=1.02)
    plt.show()

# Função de plotagem para Kaggle com K-Means
def plot_kmeans_kaggle(data_scaled, labels, title="K-Means - Kaggle"):
    df = pd.DataFrame(data_scaled, columns=["budget", "popularity", "revenue", "vote_average", "vote_count"])
    df['cluster'] = labels.astype(str)
    sns.pairplot(df, hue='cluster')
    plt.suptitle(title, y=1.02)
    plt.show()

# Plotando K-Means
plot_kmeans_iris(iris_scaled, iris_clusters_kmeans, "K-Means - Iris")
plot_kmeans_kaggle(kaggle_scaled, kaggle_scaled_clusters_kmeans, "K-Means - Kaggle (Normalizado)")
plot_kmeans_kaggle(kaggle_transformed, kaggle_transformed_clusters_kmeans, "K-Means - Kaggle (Transformado)")

# --------------------------------------------------------
# 6.1) BISECTING K-MEANS
# --------------------------------------------------------

# NOTA: Embora o Bisecting K-Means não seja compatível com o método do cotovelo (Elbow Method),
# ele ainda exige que o número de clusters K seja informado previamente através do parâmetro `n_clusters`.
# Para manter a coerência na análise comparativa entre os algoritmos,
# utilizamos o mesmo valor de K que foi determinado via Elbow para o K-Means tradicional.
# Isso nos permite comparar diretamente os resultados entre os métodos de clusterização.

# Aplicação do Bisecting K-Means em cada base com os Ks definidos
bisecting_kmeans_iris = BisectingKMeans(n_clusters=k_iris, random_state=42)
iris_clusters_bkm = bisecting_kmeans_iris.fit_predict(iris_scaled)

bisecting_kmeans_kaggle_scaled = BisectingKMeans(n_clusters=k_kaggle, random_state=42)
kaggle_clusters_bkm_scaled = bisecting_kmeans_kaggle_scaled.fit_predict(kaggle_scaled)

bisecting_kmeans_kaggle_transf = BisectingKMeans(n_clusters=k_kaggle, random_state=42)
kaggle_clusters_bkm_transf = bisecting_kmeans_kaggle_transf.fit_predict(kaggle_transformed)

# Função de plotagem para Iris com Pairplot
def plot_bkm_iris(data_scaled, labels, title="Bisecting K-Means - Iris"):
    df = pd.DataFrame(data_scaled, columns=iris.feature_names)
    df['cluster'] = labels.astype(str)
    sns.pairplot(df, hue='cluster')
    plt.suptitle(title, y=1.02)
    plt.show()

# Função de plotagem para Kaggle com Pairplot
def plot_bkm_kaggle(data_scaled, labels, title="Bisecting K-Means - Kaggle"):
    df = pd.DataFrame(data_scaled, columns=["budget", "popularity", "revenue", "vote_average", "vote_count"])
    df['cluster'] = labels.astype(str)
    sns.pairplot(df, hue='cluster')
    plt.suptitle(title, y=1.02)
    plt.show()

# Gerando os gráficos
plot_bkm_iris(iris_scaled, iris_clusters_bkm, "Bisecting K-Means - Iris")
plot_bkm_kaggle(kaggle_scaled, kaggle_clusters_bkm_scaled, "Bisecting K-Means - Kaggle (Normalizado)")
plot_bkm_kaggle(kaggle_transformed, kaggle_clusters_bkm_transf, "Bisecting K-Means - Kaggle (Transformado)")


# --------------------------------------------------------
# 7) HIERARQUICO
# --------------------------------------------------------
def hierarchical_clustering(data, title, k, method):
    linked = linkage(data, method=method)
    plt.figure(figsize=(10, 5))
    dendrogram(linked,
               truncate_mode='lastp',  # mostra só os últimos clusters
               p=50,
               show_leaf_counts=True,
               leaf_rotation=90,
               leaf_font_size=10,
               show_contracted=True)
    plt.title(f"Dendrograma Resumido ({method}) - {title}")
    plt.xlabel('Clusters')
    plt.ylabel('Distância')
    plt.tight_layout()
    plt.show()
    return fcluster(linked, t=k, criterion='maxclust')

# Ward method
iris_clusters_hier_ward = hierarchical_clustering(iris_scaled, "Iris (Normalizado)", k_iris, 'ward')
kaggle_scaled_clusters_hier_ward = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", k_kaggle, 'ward')
kaggle_transformed_clusters_hier_ward = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)", k_kaggle,
                                                                'ward')

# Single method
iris_clusters_hier_single = hierarchical_clustering(iris_scaled, "Iris (Normalizado)", k_iris, 'single')
kaggle_scaled_clusters_hier_single = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", k_kaggle, 'single')
kaggle_transformed_clusters_hier_single = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)", k_kaggle,
                                                                  'single')

# Complete method
iris_clusters_hier_complete = hierarchical_clustering(iris_scaled, "Iris (Normalizado)", k_iris, 'complete')
kaggle_scaled_clusters_hier_complete = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", k_kaggle,
                                                               'complete')
kaggle_transformed_clusters_hier_complete = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)",
                                                                    k_kaggle, 'complete')

# Average method
iris_clusters_hier_average = hierarchical_clustering(iris_scaled, "Iris (Normalizado)", k_iris, 'average')
kaggle_scaled_clusters_hier_average = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", k_kaggle,
                                                              'average')
kaggle_transformed_clusters_hier_average = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)",
                                                                   k_kaggle, 'average')

# --------------------------------------------------------
# 8) SILHOUETTE SCORE - TODOS OS MÉTODOS (Incluindo BKM)
# --------------------------------------------------------

# ===== IRIS (apenas normalizada) =====
score_kmeans_iris          = silhouette_score(iris_scaled, iris_clusters_kmeans)
score_bkm_iris             = silhouette_score(iris_scaled, iris_clusters_bkm)
score_hier_iris_ward       = silhouette_score(iris_scaled, iris_clusters_hier_ward)
score_hier_iris_single     = silhouette_score(iris_scaled, iris_clusters_hier_single)
score_hier_iris_complete   = silhouette_score(iris_scaled, iris_clusters_hier_complete)
score_hier_iris_average    = silhouette_score(iris_scaled, iris_clusters_hier_average)

# ===== KAGGLE (apenas normalizada) =====
score_kmeans_kaggle_scaled        = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_kmeans)
score_bkm_kaggle_scaled           = silhouette_score(kaggle_scaled, kaggle_clusters_bkm_scaled)
score_hier_kaggle_scaled_ward     = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_hier_ward)
score_hier_kaggle_scaled_single   = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_hier_single)
score_hier_kaggle_scaled_complete = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_hier_complete)
score_hier_kaggle_scaled_average  = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_hier_average)

# ===== KAGGLE (normalizada + PowerTransformer) =====
score_kmeans_kaggle_transf        = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_kmeans)
score_bkm_kaggle_transf           = silhouette_score(kaggle_transformed, kaggle_clusters_bkm_transf)
score_hier_kaggle_transf_ward     = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_hier_ward)
score_hier_kaggle_transf_single   = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_hier_single)
score_hier_kaggle_transf_complete = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_hier_complete)
score_hier_kaggle_transf_average  = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_hier_average)

# --------------------------------------------------------
# PRINT DOS RESULTADOS
# --------------------------------------------------------

print("\n===== COMPARATIVO SILHOUETTE SCORE =====\n")

print("📌 Base IRIS (apenas normalizada):")
print(f"  • K-Means:                {score_kmeans_iris:.4f}")
print(f"  • Bisecting K-Means:      {score_bkm_iris:.4f}")
print(f"  • Hierárquico (Ward):     {score_hier_iris_ward:.4f}")
print(f"  • Hierárquico (Single):   {score_hier_iris_single:.4f}")
print(f"  • Hierárquico (Complete): {score_hier_iris_complete:.4f}")
print(f"  • Hierárquico (Average):  {score_hier_iris_average:.4f}\n")

print("📌 Base KAGGLE (apenas normalizada):")
print(f"  • K-Means:                {score_kmeans_kaggle_scaled:.4f}")
print(f"  • Bisecting K-Means:      {score_bkm_kaggle_scaled:.4f}")
print(f"  • Hierárquico (Ward):     {score_hier_kaggle_scaled_ward:.4f}")
print(f"  • Hierárquico (Single):   {score_hier_kaggle_scaled_single:.4f}")
print(f"  • Hierárquico (Complete): {score_hier_kaggle_scaled_complete:.4f}")
print(f"  • Hierárquico (Average):  {score_hier_kaggle_scaled_average:.4f}\n")

print("📌 Base KAGGLE (StandardScaler + PowerTransformer):")
print(f"  • K-Means:                {score_kmeans_kaggle_transf:.4f}")
print(f"  • Bisecting K-Means:      {score_bkm_kaggle_transf:.4f}")
print(f"  • Hierárquico (Ward):     {score_hier_kaggle_transf_ward:.4f}")
print(f"  • Hierárquico (Single):   {score_hier_kaggle_transf_single:.4f}")
print(f"  • Hierárquico (Complete): {score_hier_kaggle_transf_complete:.4f}")
print(f"  • Hierárquico (Average):  {score_hier_kaggle_transf_average:.4f}")


# --------------------------------------------------------
# RELATÓRIO FINAL (DETALHES, ESCOLHAS E JUSTIFICATIVAS)
# --------------------------------------------------------
'''
RELATÓRIO FINAL (VERSÃO ATUALIZADA):

1) DATASETS E DIMENSÕES:

   a) Base Iris:
      - Possui 4 dimensões: sepal length, sepal width, petal length e petal width.
      - São 150 instâncias (amostras), normalmente classificadas em 3 espécies de flor (Setosa, Versicolor, Virginica).
      - Aqui, o objetivo é clustering, ou seja, aprendizado não supervisionado.
        Não utilizamos os rótulos originais para treinar, apenas para comparar ou avaliar depois.

   b) Base Kaggle (TMDB):
      - Selecionamos 5 dimensões numéricas: budget, popularity, revenue, vote_average e vote_count.
      - O método dropna() remove linhas com valores ausentes (NaNs). Isso garante dados completos.
      - Escolhemos essas 5 dimensões pois são variáveis numéricas diretamente ligadas ao desempenho e métricas de filmes.

2) POR QUE NORMALIZAMOS?

   - Tanto em Iris quanto em Kaggle, utilizamos algoritmos baseados em distância Euclidiana (K-Means, Ward etc.).
   - Se uma variável tiver escala muito maior que as outras (ex.: budget vs vote_average), ela dominaria o cálculo.
   - O StandardScaler ajusta cada coluna para média 0 e desvio padrão 1, equilibrando-as.
   - Em Iris, mesmo sendo todas em cm, há diferenças de amplitude (petal vs sepal) que podem distorcer distâncias.

3) POR QUE APLICAMOS POWERTRANSFORMER NO KAGGLE?

   - Mesmo após normalizar, variáveis como budget e revenue podem permanecer com skewness muito alta.
   - O PowerTransformer (Yeo-Johnson) corrige essa assimetria, aproximando a distribuição de uma Gaussiana.
   - Isso ajuda o clustering (especialmente Ward e K-Means) a encontrar clusters mais "esféricos" e separados.

4) MÉTODO DO JOELHO (ELBOW METHOD):

   - Feito para sugerir um K, analisando a soma das distâncias intracluster (WCSS) à medida que K cresce.
   - Depois de um certo ponto (joelho), o ganho marginal fica pequeno, sendo um bom ponto de escolha.

5) ALGORITMOS DE AGRUPAMENTO:

   a) K-Means:
      - Usa minimização da variância intracluster.
      - n_init=10 para melhor robustez, random_state=42 para reprodutibilidade.

   b) Bisecting K-Means:
      - Também conhecido como 2-means iterativo.
      - Divide recursivamente os clusters em 2 até chegar a K.
      - Tende a ser mais rápido que o Hierarchical e com qualidade comparável ao K-Means.

   c) Hierarchical Clustering:
      - Quatro métodos: ward, single, complete, average.
      - Plotamos dendrogramas truncados para bases grandes.
      - ward minimiza variância intracluster (similar a K-Means).
      - single é sensível a outliers e tende a encadear.
      - complete exagera distâncias e pode formar clusters compactos pequenos.
      - average é intermediário entre single e complete.

6) SILHOUETTE SCORE:

   - Avalia coesão intra-cluster vs separação entre clusters, variando de -1 a +1.
   - Quanto mais próximo de +1, melhor.
   - Nos ajuda a comparar K-Means, Bisecting K-Means e Hierarchical sob diferentes pré-processamentos.

CONCLUSÃO E ANÁLISE:

- Testamos K-Means, Bisecting K-Means e Hierarchical (Ward, Single, Complete, Average) em duas bases.
- Observamos a importância da normalização e, no caso da Kaggle, também de transformar dados assimétricos (PowerTransformer).
- O Elbow Method auxiliou na escolha de K para cada base.
- O Silhouette Score confirmou que Ward e K-Means/BKM costumam ter melhores resultados em dados normalizados e transformados.
- Bisecting K-Means se mostrou uma alternativa interessante, combinando velocidade e bons resultados.

'''
