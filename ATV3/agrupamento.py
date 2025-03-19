# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris

# Estilização dos gráficos
sns.set(style="whitegrid")

# ====== 1️⃣ CARREGAR E PRÉ-PROCESSAR OS DADOS ====== #
# 📌 Base 1: Carregar a base Iris
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

# 📌 Base 2: Carregar dados do Kaggle (substitua pelo caminho do seu arquivo CSV)
df_kaggle = pd.read_csv('/Users/alexinsel/Desktop/PUC/IA/dataset_movies.csv')  # Modifique aqui com seu dataset
df_kaggle = df_kaggle.select_dtypes(include=[np.number])  # Apenas colunas numéricas

# Normalização dos dados
scaler = StandardScaler()
df_iris_scaled = scaler.fit_transform(df_iris)
df_kaggle_scaled = scaler.fit_transform(df_kaggle)

# ====== 2️⃣ MÉTODO DO JOELHO PARA ESCOLHER K ====== #
def elbow_method(data, title):
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('WCSS')
    plt.title(f'Método do Joelho - {title}')
    plt.show()

# Rodando o Método do Joelho para ambas as bases
elbow_method(df_iris_scaled, "Base Iris")
elbow_method(df_kaggle_scaled, "Base Kaggle")

# ====== 3️⃣ APLICAR K-MEANS ====== #
# Definir K com base no gráfico do joelho (exemplo: 3)
k_iris = 3
k_kaggle = 3

kmeans_iris = KMeans(n_clusters=k_iris, random_state=42, n_init=10)
clusters_iris = kmeans_iris.fit_predict(df_iris_scaled)

kmeans_kaggle = KMeans(n_clusters=k_kaggle, random_state=42, n_init=10)
clusters_kaggle = kmeans_kaggle.fit_predict(df_kaggle_scaled)

# Adicionar clusters ao DataFrame
df_iris["Cluster_KMeans"] = clusters_iris
df_kaggle["Cluster_KMeans"] = clusters_kaggle

# ====== 4️⃣ APLICAR CLUSTERING HIERÁRQUICO (LINKAGE) ====== #
def hierarchical_clustering(data, title, k):
    linked = linkage(data, method='ward')

    plt.figure(figsize=(10, 5))
    dendrogram(linked)
    plt.title(f'Dendrograma - {title}')
    plt.xlabel('Amostras')
    plt.ylabel('Distância')
    plt.show()

    # Criar clusters cortando o dendrograma
    return fcluster(linked, t=k, criterion="maxclust")

# Rodando o clustering hierárquico
df_iris["Cluster_Hierarchical"] = hierarchical_clustering(df_iris_scaled, "Base Iris", k_iris)
df_kaggle["Cluster_Hierarchical"] = hierarchical_clustering(df_kaggle_scaled, "Base Kaggle", k_kaggle)

# ====== 5️⃣ AVALIAR OS RESULTADOS COM SILHOUETTE SCORE ====== #
score_kmeans_iris = silhouette_score(df_iris_scaled, clusters_iris)
score_hierarchical_iris = silhouette_score(df_iris_scaled, df_iris["Cluster_Hierarchical"])

score_kmeans_kaggle = silhouette_score(df_kaggle_scaled, clusters_kaggle)
score_hierarchical_kaggle = silhouette_score(df_kaggle_scaled, df_kaggle["Cluster_Hierarchical"])

print(f"Silhouette Score - K-Means (Iris): {score_kmeans_iris}")
print(f"Silhouette Score - Clustering Hierárquico (Iris): {score_hierarchical_iris}")
print(f"Silhouette Score - K-Means (Kaggle): {score_kmeans_kaggle}")
print(f"Silhouette Score - Clustering Hierárquico (Kaggle): {score_hierarchical_kaggle}")

# ====== 6️⃣ VISUALIZAÇÃO DOS CLUSTERS ====== #
def plot_clusters(df, title, cluster_col):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df[cluster_col], palette="tab10")
    plt.title(f'Clusters - {title}')
    plt.show()

# Plotar os clusters
plot_clusters(df_iris, "Base Iris - K-Means", "Cluster_KMeans")
plot_clusters(df_iris, "Base Iris - Hierarchical", "Cluster_Hierarchical")
plot_clusters(df_kaggle, "Base Kaggle - K-Means", "Cluster_KMeans")
plot_clusters(df_kaggle, "Base Kaggle - Hierarchical", "Cluster_Hierarchical")