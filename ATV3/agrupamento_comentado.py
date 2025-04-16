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

# USAR SILHOUTTE PARA CONFIRMAR SUAS ESCOLHAS
# OBJETIVO NAO ERA MOSTRAR TODAS POSSIBILIDADES, MAS SIM MOSTRAR AS POSSIBILDIADES ESPECIFICAS

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
# 3) NORMALIZAÇÃO DAS BASES (STANDARD SCALER) e POWERTRANSFORMER (YEO-JOHNSON) APENAS NO KAGGLE
# --------------------------------------------------------

# Inicialmente normalizamos a base Iris pois achavamos q podia estar em escalas ( valores mt diferentes ) mas nada ver

# E também usamos PCA pois achavamos que a analise dos dados se dava somente atraves de 2 dimensoes e nao pairplot ( maneira ideal ).
# Logo, poucas dimensões, nao foi necessário.

scaler_kaggle = StandardScaler()
kaggle_scaled = scaler_kaggle.fit_transform(df_kaggle_original)
df_kaggle_scaled = pd.DataFrame(kaggle_scaled, columns=df_kaggle_original.columns)

pt = PowerTransformer(method='yeo-johnson')
kaggle_transformed = pt.fit_transform(df_kaggle_scaled)
df_kaggle_transformed = pd.DataFrame(kaggle_transformed, columns=df_kaggle_original.columns)

# -------------------------
# PLOTS COM NORMALIZAÇÃO
# -------------------------
sns.pairplot(df_iris)
plt.suptitle("Iris - Pairplot (Sem Normalização)", y=1.02)
plt.show()

sns.pairplot(df_kaggle_scaled)
plt.suptitle("Kaggle - Pairplot (Apenas Normalizado)", y=1.02)
plt.show()

sns.pairplot(df_kaggle_transformed)
plt.suptitle("Kaggle - Pairplot (StandardScaler + PowerTransformer)", y=1.02)
plt.show()

# --------------------------------------------------------
# 5) ELBOW METHOD PARA DEFINIR K
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

elbow_method(df_iris, "Iris (Sem Normalização)")
elbow_method(kaggle_scaled, "Kaggle (Apenas Normalizado)")
elbow_method(kaggle_transformed, "Kaggle (Normalizado + PowerTransformer)")

print("Analise os gráficos Elbow e escolha K para cada base.")
k_iris = int(input("Escolha K para IRIS: "))
k_kaggle = int(input("Escolha K para KAGGLE: "))

# --------------------------------------------------------
# 6) K-MEANS
# --------------------------------------------------------
kmeans_iris = KMeans(n_clusters=k_iris, random_state=42, n_init=10)
iris_clusters_kmeans = kmeans_iris.fit_predict(df_iris)

kmeans_kaggle_scaled = KMeans(n_clusters=k_kaggle, random_state=42, n_init=10)
kaggle_scaled_clusters_kmeans = kmeans_kaggle_scaled.fit_predict(kaggle_scaled)

kmeans_kaggle_transformed = KMeans(n_clusters=k_kaggle, random_state=42, n_init=10)
kaggle_transformed_clusters_kmeans = kmeans_kaggle_transformed.fit_predict(kaggle_transformed)

def plot_kmeans_iris(data_scaled, labels, title="K-Means - Iris"):
    df = pd.DataFrame(data_scaled, columns=iris.feature_names)
    df['cluster'] = labels.astype(str)
    sns.pairplot(df, hue='cluster')
    plt.suptitle(title, y=1.02)
    plt.show()

def plot_kmeans_kaggle(data_scaled, labels, title="K-Means - Kaggle"):
    df = pd.DataFrame(data_scaled, columns=["budget", "popularity", "revenue", "vote_average", "vote_count"])
    df['cluster'] = labels.astype(str)
    sns.pairplot(df, hue='cluster')
    plt.suptitle(title, y=1.02)
    plt.show()

plot_kmeans_iris(df_iris, iris_clusters_kmeans, "K-Means - Iris")
plot_kmeans_kaggle(kaggle_scaled, kaggle_scaled_clusters_kmeans, "K-Means - Kaggle (Normalizado)")
plot_kmeans_kaggle(kaggle_transformed, kaggle_transformed_clusters_kmeans, "K-Means - Kaggle (Transformado)")

# --------------------------------------------------------
# 6.1) BISECTING K-MEANS
# --------------------------------------------------------
bisecting_kmeans_iris = BisectingKMeans(n_clusters=k_iris, random_state=42)
iris_clusters_bkm = bisecting_kmeans_iris.fit_predict(df_iris)

bisecting_kmeans_kaggle_scaled = BisectingKMeans(n_clusters=k_kaggle, random_state=42)
kaggle_clusters_bkm_scaled = bisecting_kmeans_kaggle_scaled.fit_predict(kaggle_scaled)

bisecting_kmeans_kaggle_transf = BisectingKMeans(n_clusters=k_kaggle, random_state=42)
kaggle_clusters_bkm_transf = bisecting_kmeans_kaggle_transf.fit_predict(kaggle_transformed)

def plot_bkm_iris(data_scaled, labels, title="Bisecting K-Means - Iris"):
    df = pd.DataFrame(data_scaled, columns=iris.feature_names)
    df['cluster'] = labels.astype(str)
    sns.pairplot(df, hue='cluster')
    plt.suptitle(title, y=1.02)
    plt.show()

def plot_bkm_kaggle(data_scaled, labels, title="Bisecting K-Means - Kaggle"):
    df = pd.DataFrame(data_scaled, columns=["budget", "popularity", "revenue", "vote_average", "vote_count"])
    df['cluster'] = labels.astype(str)
    sns.pairplot(df, hue='cluster')
    plt.suptitle(title, y=1.02)
    plt.show()

plot_bkm_iris(df_iris, iris_clusters_bkm, "Bisecting K-Means - Iris")
plot_bkm_kaggle(kaggle_scaled, kaggle_clusters_bkm_scaled, "Bisecting K-Means - Kaggle (Normalizado)")
plot_bkm_kaggle(kaggle_transformed, kaggle_clusters_bkm_transf, "Bisecting K-Means - Kaggle (Transformado)")

# --------------------------------------------------------
# 7) HIERARQUICO
# --------------------------------------------------------
def hierarchical_clustering(data, title, method):
    linked = linkage(data, method=method)

    # Plota o dendrograma
    plt.figure(figsize=(10, 5))
    if "Iris" in title:
        dendrogram(linked, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=10)
    else:
        dendrogram(linked,
                   truncate_mode='lastp',  # Para bases grandes como Kaggle
                   p=50,
                   show_leaf_counts=True,
                   leaf_rotation=90,
                   leaf_font_size=10,
                   show_contracted=True)

    # Título e rótulos do dendrograma
    plt.title(f"Dendrograma Resumido ({method}) - {title}")
    plt.xlabel('Clusters')
    plt.ylabel('Distância')
    plt.tight_layout()
    plt.show()

    # Solicita o valor de corte após visualização do dendrograma
    k = int(input(f"Escolha o número de clusters (K) para {title} usando o método {method}: "))

    if k < 2:
        print(f"Erro: o número de clusters K = {k} é inválido. O número mínimo de clusters é 2.")
        return None

    # Gera os clusters com base no valor de K escolhido -> Corta dendograma para gerar esse numero de gtupos
    clusters = fcluster(linked, t=k, criterion='maxclust')

    # Calcula o Silhouette Score
    score = silhouette_score(data, clusters)
    print(f"Silhouette Score para K = {k}: {score:.4f}")

    return score

# Armazenando os resultados em um dicionário
scores = {}

# Chamada da função para diferentes cenários
scores["Iris (Sem Normalizado) - Ward"] = hierarchical_clustering(df_iris, "Iris (Sem Normalizado)", 'ward')
scores["Kaggle (Normalizado) - Ward"] = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", 'ward')
scores["Kaggle (Transformado) - Ward"] = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)", 'ward')

scores["Iris (Sem Normalizado) - Single"] = hierarchical_clustering(df_iris, "Iris (Sem Normalizado)", 'single')
scores["Kaggle (Normalizado) - Single"] = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", 'single')
scores["Kaggle (Transformado) - Single"] = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)", 'single')

scores["Iris (Sem Normalizado) - Complete"] = hierarchical_clustering(df_iris, "Iris (Sem Normalizado)", 'complete')
scores["Kaggle (Normalizado) - Complete"] = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", 'complete')
scores["Kaggle (Transformado) - Complete"] = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)", 'complete')

scores["Iris (Sem Normalizado) - Average"] = hierarchical_clustering(df_iris, "Iris (Sem Normalizado)", 'average')
scores["Kaggle (Normalizado) - Average"] = hierarchical_clustering(kaggle_scaled, "Kaggle (Normalizado)", 'average')
scores["Kaggle (Transformado) - Average"] = hierarchical_clustering(kaggle_transformed, "Kaggle (Transformado)", 'average')

# Mostra os Silhouette Scores de cada modelo
print("\n===== COMPARATIVO SILHOUETTE SCORE =====\n")
for title, score in scores.items():
    print(f"  • {title}: {score:.4f}")

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
      - Escolhemos essas 5 dimensões pois são variáveis numéricas diretamente ligadas ao desempenho e métricas de filmes. Dados uteis

2) POR QUE NORMALIZAMOS?

   - Tanto em Iris quanto em Kaggle, utilizamos algoritmos baseados em distância quadratica Euclidiana (K-Means, Ward etc.).
   - Se uma variável tiver escala muito maior que as outras (ex.: budget vs vote_average), ela dominaria o cálculo.
   - O StandardScaler ajusta cada coluna para média 0 e desvio padrão 1, equilibrando-as.
   - Iris achamos que não havia necessidade mesmo tendo feito isso antes.

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

- Testando os **Silhouette Scores** para diferentes valores de **K** (de 2 a 6) nos três datasets:

    a) **IRIS (Sem Normalização)**:
       - **Melhor Resultado**: Hierarchical (single) para **K=2** com **Silhouette Score = 0.6867**.
       - **Pior Resultado**: K-Means para **K=6** com **Silhouette Score = 0.3648**.
    
    b) **KAGGLE (Apenas Normalizado)**:
       - **Melhor Resultado**: Hierarchical (single) para **K=2** com **Silhouette Score = 0.9141**.
       - **Pior Resultado**: Bisecting K-Means para **K=6** com **Silhouette Score = 0.2971**.
    
    c) **KAGGLE (Normalizado + PowerTransformer)**:
       - **Melhor Resultado**: Hierarchical (single) para **K=2** com **Silhouette Score = 0.5541**.
       - **Pior Resultado**: Hierarchical (single) para **K=6** com **Silhouette Score = -0.0091**.

- Observamos que para **IRIS**, o melhor desempenho foi com o método **Hierarchical (single)**, que se manteve com um
**Silhouette Score alto** para **K=2** e **K=3**. **K-Means** e **Bisecting K-Means** tiveram desempenhos inferiores, 
especialmente em valores maiores de **K**.

- No **Kaggle (Normalizado)**, o **Hierarchical (single)** obteve os melhores resultados para **K=2** e **K=3**, enquanto 
os outros métodos apresentaram resultados mais baixos, especialmente em valores de **K** maiores.

- Já no **Kaggle (Normalizado + PowerTransformer)**, o método **Hierarchical (single)** também se destacou, 
mas com **K=2** apresentando o melhor desempenho, enquanto os valores de **K** maiores geraram **Silhouette 
Scores muito baixos**, especialmente para o método **single**.

===== IRIS (Sem Normalização) =====
K-Means: Silhouette Score para K = 2: 0.6810
K-Means: Silhouette Score para K = 3: 0.5528
K-Means: Silhouette Score para K = 4: 0.4981
K-Means: Silhouette Score para K = 5: 0.4912
K-Means: Silhouette Score para K = 6: 0.3648

Bisecting K-Means: Silhouette Score para K = 2: 0.6810
Bisecting K-Means: Silhouette Score para K = 3: 0.5417
Bisecting K-Means: Silhouette Score para K = 4: 0.4597
Bisecting K-Means: Silhouette Score para K = 5: 0.3317
Bisecting K-Means: Silhouette Score para K = 6: 0.3351

Hierarchical (ward): Silhouette Score para K = 2: 0.6867
Hierarchical (ward): Silhouette Score para K = 3: 0.5543
Hierarchical (ward): Silhouette Score para K = 4: 0.4890
Hierarchical (ward): Silhouette Score para K = 5: 0.4844
Hierarchical (ward): Silhouette Score para K = 6: 0.3592

Hierarchical (single): Silhouette Score para K = 2: 0.6867
Hierarchical (single): Silhouette Score para K = 3: 0.5121
Hierarchical (single): Silhouette Score para K = 4: 0.2819
Hierarchical (single): Silhouette Score para K = 5: 0.2838
Hierarchical (single): Silhouette Score para K = 6: 0.2214

Hierarchical (complete): Silhouette Score para K = 2: 0.5160
Hierarchical (complete): Silhouette Score para K = 3: 0.5136
Hierarchical (complete): Silhouette Score para K = 4: 0.4998
Hierarchical (complete): Silhouette Score para K = 5: 0.3462
Hierarchical (complete): Silhouette Score para K = 6: 0.3382

Hierarchical (average): Silhouette Score para K = 2: 0.6867
Hierarchical (average): Silhouette Score para K = 3: 0.5542
Hierarchical (average): Silhouette Score para K = 4: 0.4720
Hierarchical (average): Silhouette Score para K = 5: 0.4307
Hierarchical (average): Silhouette Score para K = 6: 0.3420

===== KAGGLE (Apenas Normalizado) =====
K-Means: Silhouette Score para K = 2: 0.6336
K-Means: Silhouette Score para K = 3: 0.3950
K-Means: Silhouette Score para K = 4: 0.3437
K-Means: Silhouette Score para K = 5: 0.3478
K-Means: Silhouette Score para K = 6: 0.3246

Bisecting K-Means: Silhouette Score para K = 2: 0.6336
Bisecting K-Means: Silhouette Score para K = 3: 0.2880
Bisecting K-Means: Silhouette Score para K = 4: 0.2929
Bisecting K-Means: Silhouette Score para K = 5: 0.2974
Bisecting K-Means: Silhouette Score para K = 6: 0.2971

Hierarchical (ward): Silhouette Score para K = 2: 0.7263
Hierarchical (ward): Silhouette Score para K = 3: 0.3064
Hierarchical (ward): Silhouette Score para K = 4: 0.2324
Hierarchical (ward): Silhouette Score para K = 5: 0.2517
Hierarchical (ward): Silhouette Score para K = 6: 0.2542

Hierarchical (single): Silhouette Score para K = 2: 0.9141
Hierarchical (single): Silhouette Score para K = 3: 0.9006
Hierarchical (single): Silhouette Score para K = 4: 0.8777
Hierarchical (single): Silhouette Score para K = 5: 0.8575
Hierarchical (single): Silhouette Score para K = 6: 0.8556

Hierarchical (complete): Silhouette Score para K = 2: 0.8807
Hierarchical (complete): Silhouette Score para K = 3: 0.8454
Hierarchical (complete): Silhouette Score para K = 4: 0.6804
Hierarchical (complete): Silhouette Score para K = 5: 0.6803
Hierarchical (complete): Silhouette Score para K = 6: 0.6372

Hierarchical (average): Silhouette Score para K = 2: 0.8807
Hierarchical (average): Silhouette Score para K = 3: 0.8780
Hierarchical (average): Silhouette Score para K = 4: 0.8577
Hierarchical (average): Silhouette Score para K = 5: 0.8575
Hierarchical (average): Silhouette Score para K = 6: 0.6946

===== KAGGLE (Normalizado + PowerTransformer) =====
K-Means: Silhouette Score para K = 2: 0.4365
K-Means: Silhouette Score para K = 3: 0.3097
K-Means: Silhouette Score para K = 4: 0.2828
K-Means: Silhouette Score para K = 5: 0.2856
K-Means: Silhouette Score para K = 6: 0.2687

Bisecting K-Means: Silhouette Score para K = 2: 0.4364
Bisecting K-Means: Silhouette Score para K = 3: 0.2895
Bisecting K-Means: Silhouette Score para K = 4: 0.2368
Bisecting K-Means: Silhouette Score para K = 5: 0.2273
Bisecting K-Means: Silhouette Score para K = 6: 0.2367

Hierarchical (ward): Silhouette Score para K = 2: 0.3727
Hierarchical (ward): Silhouette Score para K = 3: 0.2898
Hierarchical (ward): Silhouette Score para K = 4: 0.2478
Hierarchical (ward): Silhouette Score para K = 5: 0.2376
Hierarchical (ward): Silhouette Score para K = 6: 0.2062

Hierarchical (single): Silhouette Score para K = 2: 0.5541
Hierarchical (single): Silhouette Score para K = 3: 0.2193
Hierarchical (single): Silhouette Score para K = 4: 0.0625
Hierarchical (single): Silhouette Score para K = 5: 0.0344
Hierarchical (single): Silhouette Score para K = 6: -0.0091

Hierarchical (complete): Silhouette Score para K = 2: 0.5541
Hierarchical (complete): Silhouette Score para K = 3: 0.4118
Hierarchical (complete): Silhouette Score para K = 4: 0.2275
Hierarchical (complete): Silhouette Score para K = 5: 0.2122
Hierarchical (complete): Silhouette Score para K = 6: 0.1844

Hierarchical (average): Silhouette Score para K = 2: 0.5541
Hierarchical (average): Silhouette Score para K = 3: 0.4131
Hierarchical (average): Silhouette Score para K = 4: 0.3321
Hierarchical (average): Silhouette Score para K = 5: 0.2515
Hierarchical (average): Silhouette Score para K = 6: 0.2243

- Observamos a importância da normalização para evitar que variáveis em escalas diferentes dominem o agrupamento. 
  No caso do dataset Kaggle, também aplicamos o PowerTransformer para tratar skewness com intuito de testar. Entretanto, para 
  fins de estudo e por duvida se estava correto, trabalhamos com normalizado e normalizado + transformado.

- A transformação tornou os dados mais Gaussianos e facilitou a visualização, mas acabou prejudicando o agrupamento, pois deixou as 
  distâncias muito homogêneas e próximas da média, o que exige um valor de K diferente para melhor separação.
'''