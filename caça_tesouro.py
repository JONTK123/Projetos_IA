# Alex Insel          RA: 21008278
# João Pedro Giaretta RA: 23008717
# Thiago Luiz Fossa   RA: 23010116

# Gere uma grade 2D com tamanho 7x7 representando um mapa que consiste em:
# - Paredes (#): Células intransponíveis.
# - Espaços vazios (.): Células transitáveis.
# - Início (S): A posição inicial do caçador de tesouros.
# - Tesouro (T): O objetivo a ser alcançado.

# Função para criar o mapa
def gerarMatriz():
    # Gerando a matriz
    matriz = [
        ['#', '#', '#', '#', '#', '#', '#'],
        ['#', 'S', '.', '.', '.', '.', '#'],
        ['#', '.', '#', '.', '#', '#', '#'],
        ['#', '.', '.', '.', '#', '.', '#'],
        ['#', '.', '#', '#', '#', '.', '#'],
        ['#', '.', '.', 'T', '.', '.', '#'],
        ['#', '#', '#', '#', '#', '#', '#']
    ]
    return matriz

# Função para imprimir o mapa
def imprimirMatriz(matriz):
    for linha in matriz:
        print(' '.join(linha))

# Função para encontrar o inicio
def encontrarInicio(matriz):
    # loop para encontrar a posição inicial
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            if matriz[i][j] == 'S':
                posicao_inicial = (i, j)

    return posicao_inicial


# Função para encontrar o caminho
def encontrarCaminho(matriz):
    posicao_inicial = encontrarInicio(matriz) # Chama a função para encontrar a posição inicial
    caminho = [posicao_inicial] # Lista para armazenar o caminho percorrido, no final será o caminho até o tesouro
    posicoes_visitadas = [posicao_inicial]  # Lista para armazenar as posições visitadas
    i, j = posicao_inicial # definindo a posição inicial
    count = 0 # Contador de Passos dados

    while matriz[i][j] != 'T': 
        #print(len(matriz[i])-1)
        #print(i) # Enquanto não encontrar o Tesouro
        print(f'Caminho Percorrido: {caminho}') # Imprime o caminho percorrido

        # Verificações para os 4 lados, vendo se já não foi por esse caminho e se a próxima posição é '.' ou 'T'
        # Verifica a posição de cima(i-1)
        if i > 0 and (matriz[i-1][j] == '.' and (i-1, j) not in posicoes_visitadas) or matriz[i-1][j] == 'T':
            i -= 1
            caminho.append((i, j))
            posicoes_visitadas.append((i, j))
            count += 1

        # Verifica a posição de baixo (i+1)
        # Adicionei uma verificação dupla de compimento da matriz no OR, pois quando eu colocava o 'S' na ultima linha da matriz,
        # ele dava erro de tamanho da matriz.
        elif (i < (len(matriz[i]) - 1) and matriz[i+1][j] == '.' and (i+1, j) not in posicoes_visitadas) or (i < (len(matriz[i]) - 1) and matriz[i+1][j] == 'T'):
            i += 1
            caminho.append((i, j))
            posicoes_visitadas.append((i, j))
            count += 1

        # Verifica a posição à esquerda (j-1)
        elif (j > 0 and matriz[i][j-1] == '.' and (i, j-1) not in posicoes_visitadas) or matriz[i][j-1] == 'T':
            j -= 1
            caminho.append((i, j))
            posicoes_visitadas.append((i, j))
            count += 1

        # Verifica a posição à direita (j+1)
        elif (j < len(matriz[i]) - 1 and matriz[i][j+1] == '.' and (i, j+1) not in posicoes_visitadas) or matriz[i][j+1] == 'T':
            j += 1
            caminho.append((i, j))
            posicoes_visitadas.append((i, j))
            count += 1

        else:
            # Volta um passo
            # Caso não tenha mais caminho disponível, ele volta um passo
            # Remove o caminho encurralado da lista de caminho e volta para a posição anterior
            caminho.pop()
            i, j = caminho[-1]
            count += 1

    print("-------------------")
    print("Tesouro Encontrado!")
    print(f"Caminho até o Tesouro: {caminho}")
    print(f"Quantidade de passos: {count}")

if __name__ == '__main__':
    matriz = gerarMatriz()
    imprimirMatriz(matriz)
    encontrarCaminho(matriz)

# Nossa ideia para a implementação da solução foi de que fosse percorrido o caminho disponivel e armazenado em uma lista de caminho
# e a cada passo dado, verificamos se a proxima posição é valida ou o tesouro, caso não seja, ele muda de direção e continua o caminho
# caso não tenha mais caminho disponivel, ele volta um passo e tenta outra direção.
# A lista de caminho foi utilizada primordialmente para que quando ele voltasse um passo, ele não tentasse a mesma direção novamente
# e assim, conseguisse encontrar o tesouro.
