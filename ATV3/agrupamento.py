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

# -------------------------------------------------------------
#                DESCRIÇÃO DESTE SCRIPT
# -------------------------------------------------------------
#
# 1) Carregamos duas bases:
#    - Iris (4 dimensões: sepal length, sepal width, petal length, petal width)
#    - Kaggle TMDB (5 dimensões numéricas: budget, popularity, revenue, vote_average, vote_count)
#
# 2) Fazemos Pairplot SEM normalização, para ver escalas reais.
#
# 3) Normalizamos (StandardScaler) AMBAS as bases.
#    - A base IRIS fica só com essa normalização.
#    - A base KAGGLE terá duas versões:
#         a) Kaggle_Scaled: apenas com StandardScaler
#         b) Kaggle_Transformed: StandardScaler + PowerTransformer
#           (usando método 'yeo-johnson')
#
# 4) Visualizamos Pairplot nas versões normalizadas.
#    - Para Kaggle, compararemos Kaggle_Scaled e Kaggle_Transformed.
#
# 5) Aplicamos o Método do "Joelho" (Elbow Method) e rodamos K-Means e Hierárquico (Ward) em:
#    - IRIS (apenas normalizada)
#    - KAGGLE sem transformação extra (apenas escalada)
#    - KAGGLE com PowerTransformer
#
# 6) Calculamos o Silhouette Score para cada caso, comparando a qualidade do cluster.
#
# 7) Explicamos, ao final, por que se escolheu Ward no Hierarchical, por que normalizar,
#    o que é PowerTransformer, e o que é Silhouette Score.
#
# -------------------------------------------------------------

sns.set_theme(style="whitegrid")

# -------------------------------------------------------------
# 1) IMPORTANDO E PREPARANDO OS DADOS
# -------------------------------------------------------------
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

# Baixando base Kaggle (TMDB)
path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
df_kaggle_original = pd.read_csv(f"{path}/tmdb_5000_movies.csv")

# Mantendo apenas as colunas numéricas relevantes
df_kaggle_original = df_kaggle_original[['budget', 'popularity', 'revenue', 'vote_average', 'vote_count']].dropna()

# -------------------------------------------------------------
# 2) VISUALIZAÇÃO EM PARES (SEM NORMALIZAÇÃO)
# -------------------------------------------------------------
sns.pairplot(df_iris)
plt.suptitle("Iris - Pairplot (Sem Normalização)", y=1.02)
plt.show()

sns.pairplot(df_kaggle_original)
plt.suptitle("Kaggle (Original) - Pairplot (Sem Normalização)", y=1.02)
plt.show()

# -------------------------------------------------------------
# 3) NORMALIZAÇÃO COM STANDARD SCALER
# -------------------------------------------------------------
#   Explicação:
#   O StandardScaler z-normaliza cada coluna:
#       z = (x - média) / desvio_padrao
#   Isso faz cada variável ter média 0 e desvio padrão 1,
#   evitando que escalas muito diferentes (ex: budget em milhões
#   vs. vote_average 0~10) dominem o cálculo de distância.
# -------------------------------------------------------------

scaler_iris = StandardScaler()
iris_scaled = scaler_iris.fit_transform(df_iris)
df_iris_scaled = pd.DataFrame(iris_scaled, columns=df_iris.columns)

scaler_kaggle = StandardScaler()
kaggle_scaled = scaler_kaggle.fit_transform(df_kaggle_original)
df_kaggle_scaled = pd.DataFrame(kaggle_scaled, columns=df_kaggle_original.columns)

# -------------------------------------------------------------
# (SOMENTE PARA KAGGLE) 4a) APLICAR POWERTRANSFORMER
#     -> Transformação extra para reduzir skewness
#        e deixar a distribuição mais 'gaussiana'
#     -> Método: 'yeo-johnson', que lida com valores < 0
# -------------------------------------------------------------
pt = PowerTransformer(method='yeo-johnson')
kaggle_transformed = pt.fit_transform(df_kaggle_scaled)
df_kaggle_transformed = pd.DataFrame(kaggle_transformed, columns=df_kaggle_original.columns)

# -------------------------------------------------------------
# 4b) VISUALIZAÇÃO DOS DADOS (COM NORMALIZAÇÃO E TRANSFORMAÇÃO)
#    1) IRIS: apenas df_iris_scaled
#    2) KAGGLE: comparando df_kaggle_scaled vs. df_kaggle_transformed
# -------------------------------------------------------------
sns.pairplot(df_iris_scaled)
plt.suptitle("Iris - Pairplot (Apenas Normalizado)", y=1.02)
plt.show()

sns.pairplot(df_kaggle_scaled)
plt.suptitle("Kaggle - Pairplot (Apenas Normalizado)", y=1.02)
plt.show()

sns.pairplot(df_kaggle_transformed)
plt.suptitle("Kaggle - Pairplot (StandardScaler + PowerTransformer)", y=1.02)
plt.show()

# -------------------------------------------------------------
# 5) MÉTODO DO JOELHO (ELBOW METHOD)
#    Vamos fazer 3 Elbow Methods:
#       A) IRIS (somente normalizado)
#       B) KAGGLE_SCALED (somente normalizado)
#       C) KAGGLE_TRANSFORMED (normalizado + power)
# -------------------------------------------------------------

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

# IRIS (somente normalizado)
elbow_method(iris_scaled, "Iris (Normalizado)")

# KAGGLE sem PowerTransformer
elbow_method(kaggle_scaled, "Kaggle (Apenas Normalizado)")

# KAGGLE com PowerTransformer
elbow_method(kaggle_transformed, "Kaggle (Normalizado + PowerTransformer)")

# -------------------------------------------------------------
# 6) RODANDO K-MEANS E HIERÁRQUICO (WARD)
#    -> Em IRIS, K=3 (clássico)
#    -> Em KAGGLE, tentaremos K=5 (exemplo).
#    -> Testando:
#        - Kaggle_Scaled
#        - Kaggle_Transformed
# -------------------------------------------------------------

k_iris = 3
k_kaggle = 5

# 6a) K-Means
kmeans_iris = KMeans(n_clusters=k_iris, random_state=42, n_init=10)
iris_clusters_kmeans = kmeans_iris.fit_predict(iris_scaled)

kmeans_kaggle_scaled = KMeans(n_clusters=k_kaggle, random_state=42, n_init=10)
kaggle_scaled_clusters_kmeans = kmeans_kaggle_scaled.fit_predict(kaggle_scaled)

kmeans_kaggle_transformed = KMeans(n_clusters=k_kaggle, random_state=42, n_init=10)
kaggle_transformed_clusters_kmeans = kmeans_kaggle_transformed.fit_predict(kaggle_transformed)

# 6b) Hierarchical Clustering
def hierarchical_clustering(data, title, k, method='ward'):
    # Explicação:
    #  - method='ward' minimiza a variância intracluster
    #  - gera dendrograma com distâncias baseadas em soma de quadrados
    linked = linkage(data, method=method)
    plt.figure(figsize=(8, 4))
    dendrogram(linked)
    plt.title(f'Dendrograma ({method}) - {title}')
    plt.xlabel('Amostras')
    plt.ylabel('Distância')
    plt.show()
    return fcluster(linked, t=k, criterion='maxclust')

iris_clusters_hier = hierarchical_clustering(iris_scaled, "Iris (Normalizado)", k_iris, 'ward')
kaggle_scaled_clusters_hier = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", k_kaggle, 'ward')
kaggle_transformed_clusters_hier = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)", k_kaggle, 'ward')

# -------------------------------------------------------------
# 7) CALCULANDO O SILHOUETTE SCORE
#    -> Avaliação de consistência interna dos clusters
#    -> Quanto mais perto de 1, melhor a separação.
# -------------------------------------------------------------
score_kmeans_iris = silhouette_score(iris_scaled, iris_clusters_kmeans)
score_hier_iris   = silhouette_score(iris_scaled, iris_clusters_hier)

score_kmeans_kg_scaled = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_kmeans)
score_hier_kg_scaled   = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_hier)

score_kmeans_kg_transf = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_kmeans)
score_hier_kg_transf   = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_hier)

print("===== Silhouette Scores =====")
print(f"Iris (K-Means): {score_kmeans_iris:.4f}")
print(f"Iris (Hierárquico Ward): {score_hier_iris:.4f}\n")

print(f"Kaggle (Normalizado) - KMeans: {score_kmeans_kg_scaled:.4f}")
print(f"Kaggle (Normalizado) - Hierárquico: {score_hier_kg_scaled:.4f}\n")

print(f"Kaggle (Transformado) - KMeans: {score_kmeans_kg_transf:.4f}")
print(f"Kaggle (Transformado) - Hierárquico: {score_hier_kg_transf:.4f}")

# ----------------------------------------------------------------------------------------------#