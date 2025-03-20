import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import kagglehub

# Estilização dos gráficos
sns.set_theme(style="whitegrid")

# Base 1: Carregar a base Iris
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

# Base 2: Baixar e carregar os dados do Kaggle
path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
df_kaggle = pd.read_csv(f"{path}/tmdb_5000_movies.csv")

# Manter apenas colunas numéricas e remover valores faltantes
df_kaggle = df_kaggle.select_dtypes(include=[np.number]).dropna()

# Como as duas bases de dados possuem + de 2 dimensões, vamos aplicar PCA para reduzir para 2 dimensões
# Aplicar PCA na base Iris (2 dimensões) SEM normalização
pca_iris = PCA(n_components=2)
iris_pca = pca_iris.fit_transform(df_iris)
df_iris_pca = pd.DataFrame(iris_pca, columns=['PC1', 'PC2'])

# Aplicar PCA na base Kaggle (2 dimensões) SEM normalização
pca_kaggle = PCA(n_components=2)
kaggle_pca = pca_kaggle.fit_transform(df_kaggle)  # Agora usando TODAS as dimensões!
df_kaggle_pca = pd.DataFrame(kaggle_pca, columns=['PC1', 'PC2'])

# Plotando os gráficos após PCA sem normalização apenas para comparação
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_iris_pca, x='PC1', y='PC2')
plt.title("Base Iris - PCA (Sem Normalização)")
plt.xlabel("Componente Principal 1 (PC1)")
plt.ylabel("Componente Principal 2 (PC2)")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_kaggle_pca, x='PC1', y='PC2')
plt.title("Base Kaggle - PCA (Sem Normalização)")
plt.xlabel("Componente Principal 1 (PC1)")
plt.ylabel("Componente Principal 2 (PC2)")
plt.show()

# Normalizando os dados antes do PCA pois as escalas são diferentes
scaler_iris = StandardScaler()
iris_scaled = scaler_iris.fit_transform(df_iris)

scaler_kaggle = StandardScaler()
kaggle_scaled = scaler_kaggle.fit_transform(df_kaggle)

# Aplicar PCA após normalização
pca_iris = PCA(n_components=2)
iris_pca_scaled = pca_iris.fit_transform(iris_scaled)
df_iris_pca_scaled = pd.DataFrame(iris_pca_scaled, columns=['PC1', 'PC2'])

pca_kaggle = PCA(n_components=2)
kaggle_pca_scaled = pca_kaggle.fit_transform(kaggle_scaled)
df_kaggle_pca_scaled = pd.DataFrame(kaggle_pca_scaled, columns=['PC1', 'PC2'])

# Plotando os gráficos após normalização e PCA
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_iris_pca_scaled, x='PC1', y='PC2')
plt.title("Base Iris - PCA (Com Normalização)")
plt.xlabel("Componente Principal 1 (PC1)")
plt.ylabel("Componente Principal 2 (PC2)")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_kaggle_pca_scaled, x='PC1', y='PC2')
plt.title("Base Kaggle - PCA (Com Normalização)")
plt.xlabel("Componente Principal 1 (PC1)")
plt.ylabel("Componente Principal 2 (PC2)")
plt.show()

# Método do Joelho para o algoritmo K-Means
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
elbow_method(df_iris_pca_scaled, "Base Iris")
elbow_method(df_kaggle_pca_scaled, "Base Kaggle")

# Definir K com base no gráfico do joelho
k_iris = 3
k_kaggle = 5

kmeans_iris = KMeans(n_clusters=k_iris, random_state=42, n_init=10)
clusters_iris = kmeans_iris.fit_predict(df_iris_pca_scaled)

kmeans_kaggle = KMeans(n_clusters=k_kaggle, random_state=42, n_init=10)
clusters_kaggle = kmeans_kaggle.fit_predict(df_kaggle_pca_scaled)

# Adicionar clusters ao DataFrame
df_iris["Cluster_KMeans"] = clusters_iris
df_kaggle["Cluster_KMeans"] = clusters_kaggle

# Aplicando o clustering hierárquico ward pois minimiza a variância
def hierarchical_clustering_ward(data, title, k):
    linked = linkage(data, method='ward')

    plt.figure(figsize=(10, 5))
    dendrogram(linked)
    plt.title(f'Dendrograma - {title}')
    plt.xlabel('Amostras')
    plt.ylabel('Distância')
    plt.show()

    # Criar clusters cortando o dendrograma
    return fcluster(linked, t=k, criterion="maxclust")

# Aplicando o clustering hierárquico average pois evita clusters alongados
def hierarchical_clustering_average(data, title, k):
    linked = linkage(data, method='average')

    plt.figure(figsize=(10, 5))
    dendrogram(linked)
    plt.title(f'Dendrograma - {title}')
    plt.xlabel('Amostras')
    plt.ylabel('Distância')
    plt.show()

    # Criar clusters cortando o dendrograma
    return fcluster(linked, t=k, criterion="maxclust")

# Rodando o clustering hierárquico
df_iris["Cluster_Hierarchical"] = hierarchical_clustering_ward(df_iris_pca_scaled, "Base Iris", k_iris)
df_kaggle["Cluster_Hierarchical"] = hierarchical_clustering_average(df_kaggle_pca_scaled, "Base Kaggle", k_kaggle)

score_kmeans_iris = silhouette_score(df_iris_pca_scaled, clusters_iris)
score_hierarchical_iris = silhouette_score(df_iris_pca_scaled, df_iris["Cluster_Hierarchical"])

score_kmeans_kaggle = silhouette_score(df_kaggle_pca_scaled, clusters_kaggle)
score_hierarchical_kaggle = silhouette_score(df_kaggle_pca_scaled, df_kaggle["Cluster_Hierarchical"])

print(f"Silhouette Score - K-Means (Iris): {score_kmeans_iris}")
print(f"Silhouette Score - Clustering Hierárquico (Iris): {score_hierarchical_iris}")
print(f"Silhouette Score - K-Means (Kaggle): {score_kmeans_kaggle}")
print(f"Silhouette Score - Clustering Hierárquico (Kaggle): {score_hierarchical_kaggle}")

def plot_algorithm_clusters(df, title, cluster_col):
    plt.figure(figsize=(6, 4))  # Define o tamanho do gráfico
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df[cluster_col], palette="tab10")
    plt.title(f'Clusters - {title}')  # Adiciona o título ao gráfico
    plt.xlabel("Componente Principal 1 (PC1)")  # Nome do eixo X
    plt.ylabel("Componente Principal 2 (PC2)")  # Nome do eixo Y
    plt.legend(title="Clusters")  # Adiciona legenda
    plt.show()  # Exibe o gráfico

plot_algorithm_clusters(df_iris, "Base Iris - K-Means", "Cluster_KMeans")
plot_algorithm_clusters(df_iris, "Base Iris - Hierarchical", "Cluster_Hierarchical")
plot_algorithm_clusters(df_kaggle, "Base Kaggle - K-Means", "Cluster_KMeans")
plot_algorithm_clusters(df_kaggle, "Base Kaggle - Hierarchical", "Cluster_Hierarchical")