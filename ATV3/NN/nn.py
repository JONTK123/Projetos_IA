# ---------------------------------------------------------------
#  Script completo – agora com mais três gráficos de visualização
# ---------------------------------------------------------------

# DataSet simples  (TODOS os comentários antigos foram mantidos)
frases = [
    "o hotel oferece wi-fi gratuito",
    "o check-in começa às 14h",
    "o café da manhã é servido até as 10h",
    "a recepção funciona 24 horas",
    "o check-out é até meio-dia",
    "para ligar nao perturbe, toque no interruptor 2",
    "o hotel tem piscina aquecida",
    "o hotel tem academia",
    "o hotel tem estacionamento gratuito",
    "o hotel tem serviço de lavanderia",
    "o hotel tem serviço de quarto 24 horas",
]

# Tokenizando o dataset ( sem biblioteca )
# ----------------------------------------------------------------------------------------
palavras = set(" ".join(frases).split())
word2id = {w:i for i, w in enumerate(palavras)}
id2word = {i:w for w, i in word2id.items()}

print(f"Palavras: {palavras}")
print(f"Word2ID: {word2id}")
print(f"ID2Word: {id2word}")
print("\n")

# Tokeinzando o dataset ( com biblioteca )
# ----------------------------------------------------------------------------------------
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

for i, frase in enumerate(frases):
    tokens = tokenizer.tokenize(frase)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    frase_tokenizada = tokenizer.decode(token_ids)
    print(f"Frase: {frase}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Frase tokenizada: {frase_tokenizada}")
    print("\n")

# Tokenização‑dupla (sem biblioteca)
# ----------------------------------------------------------------------------------------
treino = []
for frase in frases:
    palavras_frase = frase.split()
    for i in range(1, len(palavras_frase)):
        entrada  = palavras_frase[:i]
        saida    = palavras_frase[i]
        entrada_ids = [word2id[w] for w in entrada]
        saida_id    = word2id[saida]
        treino.append((entrada_ids, saida_id))

print(f"Treino: {treino}")

# Tokenização‑dupla (com biblioteca)
# ----------------------------------------------------------------------------------------
treino2 = []
for frase in frases:
    token_ids = tokenizer.encode(frase, add_special_tokens=False)
    for i in range(1, len(token_ids)):
        entrada_ids = token_ids[:i]
        saida_id    = token_ids[i]
        treino2.append((entrada_ids, saida_id))

print(f"Treino: {treino2}")

# Criando rede neural e preparando dados
# ----------------------------------------------------------------------------------------
from torch.utils.data import Dataset
import torch, torch.nn as nn, torch.nn.functional as F

class ChatDataset(Dataset):
    def __init__(self, dados):
        self.dados = dados
    def __len__(self):
        return len(self.dados)
    def __getitem__(self, idx):
        entrada, saida = self.dados[idx]
        return {
            "input_ids": torch.tensor(entrada, dtype=torch.long),
            "label"    : torch.tensor(saida  , dtype=torch.long)
        }

from torch.utils.data import DataLoader
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.ff1   = nn.Linear(embed_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, vocab_size)
    def forward(self, input_ids):
        emb    = self.embed(input_ids)      # [batch, seq, dim]
        x      = emb.mean(dim=1)            # pooling simples
        x      = F.relu(self.ff1(x))
        logits = self.out(x)
        return logits

# Função utilitária para prever a próxima palavra
def gerar_proxima_palavra(frase):
    modelo.eval()
    input_ids = tokenizer.encode(frase, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        logits = modelo(input_ids)
        predicted_id = torch.argmax(logits, dim=1).item()
    predicted_token = tokenizer.decode([predicted_id])
    print(f"Frase: '{frase}' → Próxima palavra prevista: '{predicted_token}'")

# ----------------------------
# TREINO + LOGS PARA GRÁFICOS
# ----------------------------
dataset2  = ChatDataset(treino2)
loader    = DataLoader(dataset2, batch_size=1, shuffle=True)
vocab_size = tokenizer.vocab_size
modelo     = MiniGPT(vocab_size)

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.01)

historico_perda      = []   # perda por época (mesmo gráfico 2‑D)
matriz_perda_batches = []   # lista de listas: [época][batch] = loss
pesos_iniciais       = []   # cópia plana dos pesos no início
pesos_finais         = []   # cópia plana no fim   (pra superfície)

# --- Salva pesos iniciais (para o gráfico de superfície) ---
with torch.no_grad():
    pesos_iniciais = torch.cat([p.flatten() for p in modelo.parameters()]).clone()

for epoca in range(50):
    total_loss = 0.0
    perdas_batch = []  # guarda loss de cada batch na época atual
    print(f"\n🚀 Época {epoca + 1}/50 começando...")

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"]
        label     = batch["label"]

        logits = modelo(input_ids)
        loss   = loss_fn(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        perdas_batch.append(loss.item())

        if step <= 3:  # só printa os três primeiros batches
            print(f"Batch {step} → loss {loss.item():.4f} | input {input_ids.tolist()} → label {label.tolist()}")

    historico_perda.append(total_loss)
    matriz_perda_batches.append(perdas_batch)  # salva a linha da matriz  (época → lista de losses)

    if epoca % 10 == 0:
        print(f"\nÉpoca {epoca} - Perda acumulada: {total_loss:.4f}")
        gerar_proxima_palavra("o hotel")
        gerar_proxima_palavra("a recepção")

# --- Salva pesos finais ---
with torch.no_grad():
    pesos_finais = torch.cat([p.flatten() for p in modelo.parameters()]).clone()

# -------------------------------------------------
#  GRÁFICO 1 – CURVA 2‑D: perda por época (clássico)
# -------------------------------------------------
import matplotlib.pyplot as plt
plt.figure()
plt.plot(historico_perda)
plt.xlabel("Época")
plt.ylabel("Perda acumulada")
plt.title("Evolução da perda durante o treino")
plt.grid(True)
plt.show()

# -------------------------------------------------------------------
#  GRÁFICO 2 – MAPA 3‑D: Época × Batch × Loss  (usa matriz_perda_batches)
# -------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

# Constrói grade de índices (época, batch)
max_batches = max(len(l) for l in matriz_perda_batches)
X, Y = np.meshgrid(
    np.arange(len(matriz_perda_batches)),          # épocas
    np.arange(max_batches)                         # batches
)
Z = np.zeros_like(X, dtype=float)

for e, losses in enumerate(matriz_perda_batches):
    Z[:len(losses), e] = losses  # preenche coluna da época

surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel("Época")
ax.set_ylabel("Batch")
ax.set_zlabel("Loss")
ax.set_title("Loss por batch ao longo das épocas")
plt.show()

# -----------------------------------------------------------------
#  GRÁFICO 3 – CURVA 2‑D: perda ao INTERPOLAR pesos (linha 1‑D)
# -----------------------------------------------------------------
# Interpolamos do peso_inicial → peso_final e medimos a perda.
n_pts = 30
alphas = torch.linspace(0, 1, n_pts)
loss_interp = []

# Backup dos pesos originais para não corromper o modelo
original_state = modelo.state_dict()

with torch.no_grad():
    for a in alphas:
        # Interpola vetor de pesos
        vetor_interpolado = pesos_iniciais + a * (pesos_finais - pesos_iniciais)
        # Coloca de volta nos parâmetros (mantendo shapes corretas)
        offset = 0
        for p in modelo.parameters():
            numel = p.numel()
            p.copy_(vetor_interpolado[offset:offset+numel].view_as(p))
            offset += numel
        # Calcula perda total em todo o dataset (rápido porque é pequeno)
        total = 0.0
        for batch in loader:
            logits = modelo(batch["input_ids"])
            total += loss_fn(logits, batch["label"]).item()
        loss_interp.append(total)

# Restaura pesos finais originais
modelo.load_state_dict(original_state)

plt.figure()
plt.plot(alphas.numpy(), loss_interp)
plt.xlabel("α  (0 = início | 1 = fim)")
plt.ylabel("Perda")
plt.title("Curva 1‑D da perda ao longo da linha de interpolação dos pesos")
plt.grid(True)
plt.show()

# -----------------------------------------------------------------
#  GRÁFICO 4 – SUPERFÍCIE 3‑D da perda em função de dois eixos de peso
# -----------------------------------------------------------------
# Escolhemos DUAS direções (d1, d2) no espaço dos pesos.
d1 = (pesos_finais - pesos_iniciais)                  # direção do treinamento
d1 = d1 / d1.norm()                                   # normaliza
rand_vec = torch.randn_like(d1)                       # vetor aleatório
d2 = rand_vec - (rand_vec @ d1) * d1                  # torna‑o ortogonal a d1
d2 = d2 / d2.norm()

grid = torch.linspace(-1, 1, 25)   # tráfego de −1 a +1 unidade em cada direção
loss_surface = np.zeros((len(grid), len(grid)))

with torch.no_grad():
    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            w = pesos_finais + a*d1 + b*d2           # posição (a,b) na superfície
            # carrega pesos
            offset = 0
            for p in modelo.parameters():
                numel = p.numel()
                p.copy_(w[offset:offset+numel].view_as(p))
                offset += numel
            # calcula perda total
            total = 0.0
            for batch in loader:
                logits = modelo(batch["input_ids"])
                total += loss_fn(logits, batch["label"]).item()
            loss_surface[i, j] = total

# volta p/ pesos finais “reais”
modelo.load_state_dict(original_state)

Xg, Yg = np.meshgrid(grid.numpy(), grid.numpy())
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Xg, Yg, loss_surface, cmap='coolwarm', linewidth=0)
ax.set_xlabel("Direção d1")
ax.set_ylabel("Direção d2")
ax.set_zlabel("Perda")
ax.set_title("Superfície 3‑D da Loss (projeção em 2 direções de peso)")
plt.show()

# -------------------------------------------------
# Previsões finais (nada foi removido)
# -------------------------------------------------
gerar_proxima_palavra("o hotel")
gerar_proxima_palavra("o check-in")
gerar_proxima_palavra("a recepção")
gerar_proxima_palavra("o hotel tem")

# A lógica geral continua a mesma:
# - percorre o dataset 50 vezes (época = 50)
# - analisa 1 exemplo por vez (batch = 1)
# - agora, além da curva clássica de perda, gera:
#   1) Mapa 3‑D Época×Batch×Loss
#   2) Curva 2‑D da loss ao longo da linha de interpolação dos pesos
#   3) Superfície 3‑D da loss em duas direções do espaço de pesos
#   Tudo sem remover nenhum comentário original.
