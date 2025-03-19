import networkx as nx
import random
from collections import deque
import time
import matplotlib.pyplot as plt

def gerar_grafo(v, k):
    """
    Gera um grafo não-direcionado com v vértices e em média k conexões por nó.
    """
    G = nx.Graph()
    # Adiciona os nós ao grafo
    for i in range(v):
        G.add_node(i)
    # Conecta os nós de forma aleatória garantindo que cada um tenha pelo menos k arestas
    for i in range(v):
        while len(G[i]) < k:
            destino = random.randint(0, v - 1)
            if destino != i and not G.has_edge(i, destino):
                G.add_edge(i, destino)
    print(f"\n🔹 Grafo gerado com {v} vértices e {G.number_of_edges()} arestas.")
    return G

def busca_largura(G, inicio, objetivo):
    """
    Realiza a busca em largura (BFS) para encontrar um caminho entre `inicio` e `objetivo`.
    Retorna o caminho e o tempo de execução.
    """
    inicio_tempo = time.time()
    fila = deque([inicio])
    visitado = {inicio: None}

    while fila:
        no = fila.popleft()
        if no == objetivo:
            fim_tempo = time.time()
            caminho = reconstruir_caminho(visitado, objetivo)
            return caminho, fim_tempo - inicio_tempo
        for vizinho in G[no]:
            if vizinho not in visitado:
                visitado[vizinho] = no
                fila.append(vizinho)
    return None, time.time() - inicio_tempo

def busca_profundidade(G, inicio, objetivo):
    """
    Realiza a busca em profundidade (DFS) para encontrar um caminho entre `inicio` e `objetivo`.
    Retorna o caminho e o tempo de execução.
    """
    inicio_tempo = time.time()
    pilha = [inicio]
    visitado = {inicio: None}

    while pilha:
        no = pilha.pop()
        if no == objetivo:
            fim_tempo = time.time()
            caminho = reconstruir_caminho(visitado, objetivo)
            return caminho, fim_tempo - inicio_tempo
        for vizinho in G[no]:
            if vizinho not in visitado:
                visitado[vizinho] = no
                pilha.append(vizinho)
    return None, time.time() - inicio_tempo

def busca_profundidade_limitada(G, inicio, objetivo, limite):
    """
    Realiza a busca em profundidade limitada para encontrar um caminho entre `inicio` e `objetivo`
    respeitando o limite máximo de profundidade.
    Retorna o caminho e o tempo de execução.
    """
    inicio_tempo = time.time()
    pilha = [(inicio, 0)]
    visitado = {inicio: None}

    while pilha:
        no, profundidade = pilha.pop()
        if no == objetivo:
            fim_tempo = time.time()
            caminho = reconstruir_caminho(visitado, objetivo)
            return caminho, fim_tempo - inicio_tempo
        if profundidade < limite:
            for vizinho in G[no]:
                if vizinho not in visitado:
                    visitado[vizinho] = no
                    pilha.append((vizinho, profundidade + 1))
    return None, time.time() - inicio_tempo

def reconstruir_caminho(visitado, objetivo):
    """Reconstrói o caminho do objetivo até o início."""
    caminho = []
    atual = objetivo
    while atual is not None:
        caminho.append(atual)
        atual = visitado[atual]
    return caminho[::-1]  # Inverte a lista para que comece em 'inicio'

def desenhar_grafo(G, caminho=None, inicio=None, objetivo=None, titulo="Grafo Gerado e Caminho Encontrado"):
    """
    Desenha o grafo gerado e destaca o caminho encontrado pela busca.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Gera uma posição para os nós

    # Desenha todos os nós e arestas do grafo
    nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color='gray', node_size=300, font_size=8)

    # Se houver um caminho, destacar os nós e arestas desse caminho
    if caminho:
        path_edges = list(zip(caminho, caminho[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=caminho, node_color='red', node_size=400)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=2)

    # Destaca os nós de início e objetivo
    if inicio is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[inicio], node_color='green', node_size=500)
    if objetivo is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[objetivo], node_color='yellow', node_size=500)

    plt.title(titulo)
    plt.legend(["Caminho Percorrido", "Nó Inicial", "Nó Objetivo"])
    plt.show()

def main():
    # Definição dos parâmetros do grafo
    v = 100  # Número de vértices
    k = 3    # Número médio de arestas por nó

    # Geração do grafo
    G = gerar_grafo(v, k)

    # Seleciona aleatoriamente o ponto inicial e o ponto final (garante que sejam diferentes)
    inicio = random.randint(0, v-1)
    objetivo = random.randint(0, v-1)
    while inicio == objetivo:
        objetivo = random.randint(0, v-1)

    print(f"\n🔹 Ponto Inicial: {inicio}")
    print(f"🔹 Ponto Final: {objetivo}")

    # Executa a busca em largura (BFS)
    caminho_bfs, tempo_bfs = busca_largura(G, inicio, objetivo)
    print("\n--- Busca em Largura (BFS) ---")
    if caminho_bfs:
        print(f"Caminho encontrado: {caminho_bfs}")
        print(f"Tempo de execução: {tempo_bfs:.6f} segundos")
        print(f"Tamanho do caminho: {len(caminho_bfs)} nós")
    else:
        print("Nenhum caminho encontrado!")
    desenhar_grafo(G, caminho_bfs, inicio, objetivo, titulo="BFS - Grafo e Caminho")

    # Executa a busca em profundidade (DFS)
    caminho_dfs, tempo_dfs = busca_profundidade(G, inicio, objetivo)
    print("\n--- Busca em Profundidade (DFS) ---")
    if caminho_dfs:
        print(f"Caminho encontrado: {caminho_dfs}")
        print(f"Tempo de execução: {tempo_dfs:.6f} segundos")
        print(f"Tamanho do caminho: {len(caminho_dfs)} nós")
    else:
        print("Nenhum caminho encontrado!")
    desenhar_grafo(G, caminho_dfs, inicio, objetivo, titulo="DFS - Grafo e Caminho")

    # Executa a busca em profundidade limitada (DLS) com um limite definido
    limite = 5  # Pode ser ajustado conforme necessário
    caminho_dls, tempo_dls = busca_profundidade_limitada(G, inicio, objetivo, limite)
    print("\n--- Busca em Profundidade Limitada (DLS) ---")
    if caminho_dls:
        print(f"Caminho encontrado: {caminho_dls}")
        print(f"Tempo de execução: {tempo_dls:.6f} segundos")
        print(f"Tamanho do caminho: {len(caminho_dls)} nós")
    else:
        print("Nenhum caminho encontrado com limite =", limite)
    desenhar_grafo(G, caminho_dls, inicio, objetivo, titulo=f"DLS (Limite {limite}) - Grafo e Caminho")

if __name__ == "__main__":
    main()
