# 🤖 Projetos de Inteligência Artificial

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Repositório contendo projetos e atividades práticas de Inteligência Artificial desenvolvidos durante o curso. Os projetos abrangem desde algoritmos clássicos de busca até redes neurais e aprendizado de máquina.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Projetos](#projetos)
  - [ATV1 - Busca de Caminho e Limpeza de Dados](#atv1---busca-de-caminho-e-limpeza-de-dados)
  - [ATV2 - Algoritmos de Busca em Grafos](#atv2---algoritmos-de-busca-em-grafos)
  - [ATV3 - Agrupamento e Redes Neurais](#atv3---agrupamento-e-redes-neurais)
  - [ATV4 - Classificação com KNN e Naive Bayes](#atv4---classificação-com-knn-e-naive-bayes)
  - [ATV5 - Redes Neurais MLP e KNN](#atv5---redes-neurais-mlp-e-knn)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Autores](#autores)

## 🎯 Visão Geral

Este repositório contém 5 atividades práticas que exploram diferentes aspectos da Inteligência Artificial:

1. **Busca e Otimização**: Implementação de algoritmos de busca em mapas 2D
2. **Grafos**: Algoritmos de busca em largura, profundidade e profundidade limitada
3. **Clustering**: Técnicas de agrupamento e visualização de dados
4. **Classificação**: Algoritmos KNN e Naive Bayes
5. **Deep Learning**: Redes neurais MLP para classificação

## 🚀 Projetos

### ATV1 - Busca de Caminho e Limpeza de Dados

**Arquivos**: `caça_tesouro.py`, `limp.py`

#### Caça ao Tesouro 🗺️

Implementação de um algoritmo de busca em um mapa 2D (7x7) para encontrar um tesouro.

**Características**:
- Mapa representado como matriz com paredes (#), espaços vazios (.), início (S) e tesouro (T)
- Algoritmo de backtracking para encontrar o caminho até o tesouro
- Rastreamento de posições visitadas para evitar loops
- Contagem de passos e visualização do caminho percorrido

**Autores**: Alex Insel (RA: 21008278), João Pedro Giaretta (RA: 23008717), Thiago Luiz Fossa (RA: 23010116)

**Como executar**:
```bash
cd ATV1
python3 caça_tesouro.py
```

#### Limpeza de Dados 🧹

Função de limpeza de dados para filtragem de datasets de saúde cardiovascular.

**Filtros aplicados**:
- Idade: 10.000 - 30.000 dias
- Altura: 120 - 220 cm
- Peso: 30 - 200 kg
- Pressão arterial sistólica: 90 - 250 mmHg
- Pressão arterial diastólica: 60 - 150 mmHg

---

### ATV2 - Algoritmos de Busca em Grafos

**Arquivo**: `busca_largura_profundidade_limitada.py`

Implementação e comparação de três algoritmos clássicos de busca em grafos:

#### Algoritmos Implementados

1. **Busca em Largura (BFS)** 🌊
   - Explora o grafo nível por nível
   - Garante o caminho mais curto
   - Utiliza estrutura de fila (FIFO)

2. **Busca em Profundidade (DFS)** 🔍
   - Explora o grafo em profundidade primeiro
   - Utiliza estrutura de pilha (LIFO)
   - Pode encontrar caminhos mais longos

3. **Busca em Profundidade Limitada (DLS)** 🎯
   - DFS com limite de profundidade
   - Útil para evitar loops infinitos
   - Balanceamento entre BFS e DFS

**Funcionalidades**:
- Geração aleatória de grafos com parâmetros configuráveis
- Visualização gráfica dos grafos e caminhos encontrados
- Medição de tempo de execução para cada algoritmo
- Comparação de desempenho

**Dependências**:
- `networkx`: Manipulação de grafos
- `matplotlib`: Visualização
- `collections`: Estruturas de dados

**Como executar**:
```bash
cd ATV2
python3 busca_largura_profundidade_limitada.py
```

**Parâmetros configuráveis**:
```python
v = 100  # Número de vértices
k = 3    # Número médio de arestas por nó
limite = 5  # Limite de profundidade para DLS
```

---

### ATV3 - Agrupamento e Redes Neurais

**Arquivos**: `agrupamento.py`, `agrupamento_comentado.py`, `agrupamento_v1.py`, `NN/`

#### Agrupamento de Dados 📊

Análise e agrupamento de dados utilizando múltiplas técnicas:

**Datasets Utilizados**:
- **Iris Dataset**: Dataset clássico de flores
- **TMDB Movies**: Dados de filmes do Kaggle

**Técnicas de Pré-processamento**:
- StandardScaler: Normalização padrão
- PowerTransformer (Yeo-Johnson): Transformação de potência para dados assimétricos

**Algoritmos de Clustering**:
- K-Means
- Hierarchical Clustering (Linkage)
- Bisecting K-Means

**Métricas de Avaliação**:
- Silhouette Score: Avalia a qualidade dos clusters

**Visualizações**:
- Pairplots: Relações entre variáveis
- Dendrogramas: Clustering hierárquico
- Scatter plots: Distribuição dos clusters

#### Rede Neural Interativa 🧠

Aplicação web completa para treinamento e visualização de redes neurais.

**Arquitetura**:
- **Backend** (`NN/backend/nn.py`): FastAPI + PyTorch
- **Frontend** (`NN/frontend/`): HTML + JavaScript + Plotly + Cytoscape

**Funcionalidades**:
- Treinamento de redes neurais para previsão de tokens
- Visualização em tempo real:
  - Gráfico de perda (loss) durante o treinamento
  - Estrutura da rede neural
  - Atualização de pesos
- Interface web interativa
- WebSocket para comunicação em tempo real

**Componentes Técnicos**:
- Tokenização com GPT-2
- Embeddings de palavras
- Camadas ocultas com ativação ReLU
- Otimização com Adam
- CrossEntropyLoss

**Como executar**:
```bash
cd ATV3/NN/backend
pip install torch transformers fastapi uvicorn
python3 nn.py
# Acesse http://localhost:8000 no navegador
```

**Como usar o agrupamento**:
```bash
cd ATV3
pip install pandas matplotlib seaborn scikit-learn scipy kagglehub
python3 agrupamento.py
```

---

### ATV4 - Classificação com KNN e Naive Bayes

**Arquivos**: `knn.ipynb`, `naiveBayes.ipynb`

Implementação e análise de algoritmos de classificação usando o **Forest Cover Type Dataset**.

#### Dataset: Forest Cover Type 🌲

Previsão do tipo de cobertura florestal baseado em variáveis cartográficas.

**Variáveis Contínuas**:
- Elevation (Elevação)
- Aspect (Orientação)
- Slope (Inclinação)
- Horizontal_Distance_To_Hydrology (Distância horizontal à hidrologia)
- Vertical_Distance_To_Hydrology (Distância vertical à hidrologia)
- Horizontal_Distance_To_Roadways (Distância horizontal às estradas)
- Hillshade_9am (Sombreamento às 9h)
- Hillshade_Noon (Sombreamento ao meio-dia)
- Hillshade_3pm (Sombreamento às 15h)
- Horizontal_Distance_To_Fire_Points (Distância horizontal a pontos de incêndio)

**Classes**: 7 tipos diferentes de cobertura florestal

#### K-Nearest Neighbors (KNN) 👥

**Análise Exploratória**:
- Histogramas por classe
- Análise de correlação
- Visualização de distribuições
- PCA para redução de dimensionalidade

**Otimização de Hiperparâmetros**:
- GridSearchCV para encontrar o melhor K
- Teste de diferentes métricas de distância
- Cross-validation

**Métricas de Avaliação**:
- Acurácia
- Precision, Recall, F1-Score
- Matriz de confusão
- Classification Report

#### Naive Bayes 📈

**Implementações**:
- GaussianNB: Para variáveis contínuas
- MultinomialNB: Para dados de contagem
- Pipeline completo com preprocessamento

**Pipeline**:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GaussianNB())
])
```

**Como executar**:
```bash
cd ATV4
jupyter notebook knn.ipynb
# ou
jupyter notebook naiveBayes.ipynb
```

---

### ATV5 - Redes Neurais MLP e KNN

**Arquivos**: `mlp.ipynb`, `knn.ipynb`, `requirements.txt`

Aplicação de redes neurais e KNN ao **Cardiovascular Disease Dataset**.

#### Dataset: Cardiovascular Disease 🫀

Previsão de doenças cardiovasculares baseado em dados clínicos e de estilo de vida.

**Variáveis Contínuas**:
- `age_years`: Idade em anos
- `height`: Altura em cm
- `weight`: Peso em kg
- `ap_hi`: Pressão arterial sistólica
- `ap_lo`: Pressão arterial diastólica

**Variáveis Categóricas/Binárias**:
- `gender`: Gênero (1: feminino, 2: masculino)
- `cholesterol`: Nível de colesterol (1: normal, 2: acima do normal, 3: muito acima do normal)
- `gluc`: Nível de glicose (1: normal, 2: acima do normal, 3: muito acima do normal)
- `smoke`: Fumante (0: não, 1: sim)
- `alco`: Consumo de álcool (0: não, 1: sim)
- `active`: Atividade física (0: não, 1: sim)

**Variável Target**:
- `cardio`: Presença de doença cardiovascular (0: não, 1: sim)

#### Multi-Layer Perceptron (MLP) 🧬

**Arquitetura**:
- Camadas de entrada configuráveis
- Múltiplas camadas ocultas
- Função de ativação ReLU
- Dropout para regularização
- Camada de saída com sigmoid/softmax

**Pré-processamento**:
- Limpeza de dados usando função `limp()` da ATV1
- Normalização com StandardScaler
- Tratamento de valores ausentes
- Balanceamento de classes

**Treinamento**:
- Otimização com Adam ou SGD
- Early stopping
- Validação cruzada
- Análise de overfitting

**Métricas**:
- Acurácia
- ROC-AUC
- Matriz de confusão
- Curva de aprendizado

#### KNN para Cardiovascular 👥

Aplicação do algoritmo KNN ao mesmo dataset com:
- Otimização de K
- Análise de diferentes métricas de distância
- Comparação com MLP

**Como executar**:
```bash
cd ATV5
pip install -r requirements.txt
jupyter notebook mlp.ipynb
# ou
jupyter notebook knn.ipynb
```

---

## 💻 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Jupyter Notebook (para arquivos .ipynb)

### Instalação das Dependências

#### Instalação Global

```bash
# Clone o repositório
git clone https://github.com/JONTK123/Projetos_IA.git
cd Projetos_IA

# Instale as dependências da ATV5 (inclui a maioria das bibliotecas)
pip install -r ATV5/requirements.txt

# Dependências adicionais para ATV2
pip install networkx

# Dependências adicionais para ATV3/NN
pip install torch transformers fastapi uvicorn websockets
```

#### Instalação por Atividade

**ATV1**:
```bash
pip install pandas
```

**ATV2**:
```bash
pip install networkx matplotlib
```

**ATV3**:
```bash
pip install pandas matplotlib seaborn scikit-learn scipy kagglehub
# Para a rede neural:
pip install torch transformers fastapi uvicorn websockets
```

**ATV4 e ATV5**:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn plotly kagglehub
```

### Ambiente Virtual (Recomendado)

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
# No Linux/Mac:
source venv/bin/activate
# No Windows:
venv\Scripts\activate

# Instalar dependências
pip install -r ATV5/requirements.txt
```

---

## 🎮 Como Usar

### Scripts Python (.py)

```bash
# Navegue até o diretório da atividade
cd ATV1

# Execute o script
python3 caça_tesouro.py
```

### Jupyter Notebooks (.ipynb)

```bash
# Inicie o Jupyter Notebook
jupyter notebook

# Ou use JupyterLab
jupyter lab

# Navegue até o arquivo desejado e abra
```

### Aplicação Web (ATV3/NN)

```bash
# Navegue até o backend
cd ATV3/NN/backend

# Execute o servidor
python3 nn.py

# Abra o navegador e acesse
# http://localhost:8000
```

---

## 🛠️ Tecnologias Utilizadas

### Linguagens

- **Python 3.8+**: Linguagem principal
- **JavaScript**: Frontend da aplicação de NN
- **HTML/CSS**: Interface web

### Bibliotecas de Machine Learning

- **scikit-learn**: Algoritmos clássicos de ML
  - KNN, Naive Bayes, K-Means
  - Métricas e validação
  - Pré-processamento
- **PyTorch**: Deep Learning
  - Redes neurais personalizadas
  - Otimização e treinamento
- **Transformers**: Processamento de linguagem natural
  - Tokenização com GPT-2

### Visualização de Dados

- **Matplotlib**: Gráficos básicos
- **Seaborn**: Visualizações estatísticas
- **Plotly**: Gráficos interativos
- **Cytoscape.js**: Visualização de redes

### Manipulação de Dados

- **Pandas**: Manipulação de dataframes
- **NumPy**: Operações numéricas
- **SciPy**: Algoritmos científicos

### Web e API

- **FastAPI**: Framework web moderno
- **Uvicorn**: Servidor ASGI
- **WebSocket**: Comunicação em tempo real

### Grafos e Estruturas

- **NetworkX**: Manipulação e análise de grafos

### Datasets

- **Kaggle Hub**: Download de datasets
- **sklearn.datasets**: Datasets clássicos (Iris, etc.)

---

## 👥 Autores

### ATV1
- **Alex Insel** - RA: 21008278
- **João Pedro Giaretta** - RA: 23008717
- **Thiago Luiz Fossa** - RA: 23010116

### Outras Atividades
Contribuições de estudantes do curso de Inteligência Artificial.

---

## 📝 Estrutura do Repositório

```
Projetos_IA/
│
├── ATV1/
│   ├── caça_tesouro.py          # Algoritmo de busca em mapa 2D
│   └── limp.py                   # Função de limpeza de dados
│
├── ATV2/
│   └── busca_largura_profundidade_limitada.py  # BFS, DFS, DLS
│
├── ATV3/
│   ├── agrupamento.py            # Clustering com múltiplos algoritmos
│   ├── agrupamento_comentado.py  # Versão comentada
│   ├── agrupamento_v1.py         # Versão alternativa
│   ├── Atividade.pdf             # Descrição da atividade
│   └── NN/
│       ├── backend/
│       │   └── nn.py             # Backend FastAPI + PyTorch
│       └── frontend/
│           ├── index.html        # Interface web
│           └── script.js         # Lógica do frontend
│
├── ATV4/
│   ├── knn.ipynb                 # KNN para Forest Cover Type
│   └── naiveBayes.ipynb          # Naive Bayes para classificação
│
├── ATV5/
│   ├── mlp.ipynb                 # MLP para doenças cardiovasculares
│   ├── knn.ipynb                 # KNN para o mesmo dataset
│   └── requirements.txt          # Dependências do projeto
│
└── README.md                     # Este arquivo
```

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto é desenvolvido para fins educacionais como parte de atividades acadêmicas.

---

## 📧 Contato

Para dúvidas ou sugestões sobre o projeto, abra uma issue no repositório.

---

## 🎓 Aprendizados

Este repositório demonstra a aplicação prática de conceitos fundamentais de IA:

- ✅ Algoritmos de busca e otimização
- ✅ Estruturas de dados e grafos
- ✅ Aprendizado de máquina supervisionado e não supervisionado
- ✅ Redes neurais e deep learning
- ✅ Pré-processamento e análise de dados
- ✅ Visualização de dados e resultados
- ✅ Desenvolvimento de aplicações web com IA

---

## 🔗 Links Úteis

- [Documentação Scikit-learn](https://scikit-learn.org/)
- [Documentação PyTorch](https://pytorch.org/docs/)
- [Documentação NetworkX](https://networkx.org/)
- [Documentação FastAPI](https://fastapi.tiangolo.com/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

---

**Desenvolvido com ❤️ para aprendizado de Inteligência Artificial**
