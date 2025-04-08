import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.datasets import load_iris
import kagglehub
import sys

sys.setrecursionlimit(10000)

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
df_kaggle_original = pd.read_csv(f"{path}/tmdb_5000_movies.csv")
df_kaggle_original = df_kaggle_original[['budget', 'popularity', 'revenue', 'vote_average', 'vote_count']].dropna()

sns.pairplot(df_iris)
plt.suptitle("Iris - Pairplot (Sem Normalização)", y=1.02)
plt.show()

sns.pairplot(df_kaggle_original)
plt.suptitle("Kaggle (Original) - Pairplot (Sem Normalização)", y=1.02)
plt.show()

scaler_iris = StandardScaler()
iris_scaled = scaler_iris.fit_transform(df_iris)
df_iris_scaled = pd.DataFrame(iris_scaled, columns=df_iris.columns)

scaler_kaggle = StandardScaler()
kaggle_scaled = scaler_kaggle.fit_transform(df_kaggle_original)
df_kaggle_scaled = pd.DataFrame(kaggle_scaled, columns=df_kaggle_original.columns)

pt = PowerTransformer(method='yeo-johnson')
kaggle_transformed = pt.fit_transform(df_kaggle_scaled)
df_kaggle_transformed = pd.DataFrame(kaggle_transformed, columns=df_kaggle_original.columns)

sns.pairplot(df_iris_scaled)
plt.suptitle("Iris - Pairplot (Apenas Normalizado)", y=1.02)
plt.show()

sns.pairplot(df_kaggle_scaled)
plt.suptitle("Kaggle - Pairplot (Apenas Normalizado)", y=1.02)
plt.show()

sns.pairplot(df_kaggle_transformed)
plt.suptitle("Kaggle - Pairplot (StandardScaler + PowerTransformer)", y=1.02)
plt.show()

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

kmeans_iris = KMeans(n_clusters=k_iris, random_state=42, n_init=10)
iris_clusters_kmeans = kmeans_iris.fit_predict(iris_scaled)

kmeans_kaggle_scaled = KMeans(n_clusters=k_kaggle, random_state=42, n_init=10)
kaggle_scaled_clusters_kmeans = kmeans_kaggle_scaled.fit_predict(kaggle_scaled)

kmeans_kaggle_transformed = KMeans(n_clusters=k_kaggle, random_state=42, n_init=10)
kaggle_transformed_clusters_kmeans = kmeans_kaggle_transformed.fit_predict(kaggle_transformed)

bisecting_kmeans_iris = BisectingKMeans(n_clusters=k_iris, random_state=42)
iris_clusters_bkm = bisecting_kmeans_iris.fit_predict(iris_scaled)

bisecting_kmeans_kaggle_scaled = BisectingKMeans(n_clusters=k_kaggle, random_state=42)
kaggle_clusters_bkm_scaled = bisecting_kmeans_kaggle_scaled.fit_predict(kaggle_scaled)

bisecting_kmeans_kaggle_transf = BisectingKMeans(n_clusters=k_kaggle, random_state=42)
kaggle_clusters_bkm_transf = bisecting_kmeans_kaggle_transf.fit_predict(kaggle_transformed)

def hierarchical_clustering(data, title, k, method):
    linked = linkage(data, method=method)
    plt.figure(figsize=(10, 5))
    dendrogram(linked, truncate_mode='lastp', p=50, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=10, show_contracted=True)
    plt.title(f"Dendrograma Resumido ({method}) - {title}")
    plt.xlabel('Clusters')
    plt.ylabel('Distância')
    plt.tight_layout()
    plt.show()
    return fcluster(linked, t=k, criterion='maxclust')

iris_clusters_hier_ward = hierarchical_clustering(iris_scaled, "Iris (Normalizado)", k_iris, 'ward')
kaggle_scaled_clusters_hier_ward = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", k_kaggle, 'ward')
kaggle_transformed_clusters_hier_ward = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)", k_kaggle, 'ward')

iris_clusters_hier_single = hierarchical_clustering(iris_scaled, "Iris (Normalizado)", k_iris, 'single')
kaggle_scaled_clusters_hier_single = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", k_kaggle, 'single')
kaggle_transformed_clusters_hier_single = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)", k_kaggle, 'single')

iris_clusters_hier_complete = hierarchical_clustering(iris_scaled, "Iris (Normalizado)", k_iris, 'complete')
kaggle_scaled_clusters_hier_complete = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", k_kaggle, 'complete')
kaggle_transformed_clusters_hier_complete = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)", k_kaggle, 'complete')

iris_clusters_hier_average = hierarchical_clustering(iris_scaled, "Iris (Normalizado)", k_iris, 'average')
kaggle_scaled_clusters_hier_average = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", k_kaggle, 'average')
kaggle_transformed_clusters_hier_average = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)", k_kaggle, 'average')

score_kmeans_iris = silhouette_score(iris_scaled, iris_clusters_kmeans)
score_bkm_iris = silhouette_score(iris_scaled, iris_clusters_bkm)
score_hier_iris_ward = silhouette_score(iris_scaled, iris_clusters_hier_ward)
score_hier_iris_single = silhouette_score(iris_scaled, iris_clusters_hier_single)
score_hier_iris_complete = silhouette_score(iris_scaled, iris_clusters_hier_complete)
score_hier_iris_average = silhouette_score(iris_scaled, iris_clusters_hier_average)

score_kmeans_kaggle_scaled = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_kmeans)
score_bkm_kaggle_scaled = silhouette_score(kaggle_scaled, kaggle_clusters_bkm_scaled)
score_hier_kaggle_scaled_ward = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_hier_ward)
score_hier_kaggle_scaled_single = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_hier_single)
score_hier_kaggle_scaled_complete = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_hier_complete)
score_hier_kaggle_scaled_average = silhouette_score(kaggle_scaled, kaggle_scaled_clusters_hier_average)

score_kmeans_kaggle_transf = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_kmeans)
score_bkm_kaggle_transf = silhouette_score(kaggle_transformed, kaggle_clusters_bkm_transf)
score_hier_kaggle_transf_ward = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_hier_ward)
score_hier_kaggle_transf_single = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_hier_single)
score_hier_kaggle_transf_complete = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_hier_complete)
score_hier_kaggle_transf_average = silhouette_score(kaggle_transformed, kaggle_transformed_clusters_hier_average)

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
print(f"  • Hierárquico (Average):  {score_hier_kaggle_transf_average:.4f}\n")

