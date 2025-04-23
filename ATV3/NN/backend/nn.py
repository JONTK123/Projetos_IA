import pathlib
import tempfile
import threading
import json
from datetime import datetime as dt
from collections import defaultdict
import anyio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# --------------------  HISTÓRICO AO VIVO  --------------------
historico_buffer: list[dict] = []        # acumula cada passo


def treinar_rede(frases: list[str], out_dir: pathlib.Path):
    """Treina a mini-rede com o conjunto 'frases' fornecido e grava todos os gráficos em PNG dentro de out_dir."""

    palavras = set(" ".join(frases).split())
    word2id = {w: i for i, w in enumerate(palavras)}
    id2word = {i: w for w, i in word2id.items()}

    print(f"Palavras: {palavras}")
    print(f"Word2ID: {word2id}")
    print(f"ID2Word: {id2word}\n")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    for frase in frases:
        tokens = tokenizer.tokenize(frase)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        frase_tokenizada = tokenizer.decode(token_ids)
        print(f"Frase: {frase}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Frase tokenizada: {frase_tokenizada}\n")

    treino = []
    for frase in frases:
        palavras_frase = frase.split()
        for i in range(1, len(palavras_frase)):
            entrada = palavras_frase[:i]
            saida = palavras_frase[i]
            entrada_ids = [word2id[w] for w in entrada]
            saida_id = word2id[saida]
            treino.append((entrada_ids, saida_id))

    print(f"Treino: {treino}")

    treino2 = []
    for frase in frases:
        token_ids = tokenizer.encode(frase, add_special_tokens=False)
        for i in range(1, len(token_ids)):
            entrada_ids = token_ids[:i]
            saida_id = token_ids[i]
            treino2.append((entrada_ids, saida_id))

    print(f"Treino: {treino2}")

    class ChatDataset(Dataset):
        def __init__(self, dados):
            self.dados = dados

        def __len__(self):
            return len(self.dados)

        def __getitem__(self, idx):
            entrada, saida = self.dados[idx]
            return {
                "input_ids": torch.tensor(entrada, dtype=torch.long),
                "label": torch.tensor(saida, dtype=torch.long),
            }

    class MiniGPT(nn.Module):
        def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.ff1 = nn.Linear(embed_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, vocab_size)

        def forward(self, input_ids):
            emb = self.embed(input_ids)
            x = emb.mean(dim=1)
            x = F.relu(self.ff1(x))
            logits = self.out(x)
            return logits

    def gerar_proxima_palavra(frase):
        modelo.eval()
        input_ids = tokenizer.encode(frase, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            logits = modelo(input_ids)
            predicted_id = torch.argmax(logits, dim=1).item()
        predicted_token = tokenizer.decode([predicted_id])
        print(f"Frase: '{frase}' → Próxima palavra prevista: '{predicted_token}'")

    dataset2 = ChatDataset(treino2)
    loader = DataLoader(dataset2, batch_size=1, shuffle=True)
    vocab_size = tokenizer.vocab_size
    modelo = MiniGPT(vocab_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.01)

    historico_perda = []
    historico_acuracia = []
    historico_perplexidade = []
    matriz_perda_batches = []
    pesos_iniciais = []
    pesos_finais = []

    with torch.no_grad():
        pesos_iniciais = torch.cat([p.flatten() for p in modelo.parameters()]).clone()

    for epoca in range(50):
        total_loss = 0.0
        total_acertos = 0
        perdas_batch = []

        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"]
            label = batch["label"]

            logits = modelo(input_ids)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            acertos = (pred == label).sum().item()
            total_acertos += acertos
            perdas_batch.append(loss.item())

            _broadcast_progress(
                {
                    "epoca": epoca,
                    "batch": step,
                    "loss": loss.item(),
                    "weights": modelo.embed.weight.flatten()[:200].tolist(),
                }
            )

        historico_perda.append(total_loss)
        matriz_perda_batches.append(perdas_batch)

        if epoca % 10 == 0:
            print(f"Época {epoca} - Perda acumulada: {total_loss:.4f}")


        acuracia = total_acertos / len(loader)
        historico_acuracia.append(acuracia)
        historico_perplexidade.append(np.exp(total_loss / len(loader)))

    with torch.no_grad():
        pesos_finais = torch.cat([p.flatten() for p in modelo.parameters()]).clone()

    plt.figure()
    plt.plot(historico_perda)
    plt.xlabel("Época")
    plt.ylabel("Perda acumulada")
    plt.title("Evolução da perda durante o treino")
    plt.grid(True)
    plt.savefig(out_dir / "grafico_curva.png")
    plt.close()

    plt.figure()
    plt.plot(historico_acuracia)
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.title("Acurácia por época")
    plt.grid(True)
    plt.savefig(out_dir / "grafico_acuracia.png")
    plt.close()

    plt.figure()
    plt.plot(historico_perplexidade)
    plt.xlabel("Época")
    plt.ylabel("Perplexidade")
    plt.title("Perplexidade por época")
    plt.grid(True)
    plt.savefig(out_dir / "grafico_perplexidade.png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    max_batches = max(len(l) for l in matriz_perda_batches)
    X, Y = np.meshgrid(
        np.arange(len(matriz_perda_batches)),
        np.arange(max_batches),
    )
    Z = np.zeros_like(X, dtype=float)

    for e, losses in enumerate(matriz_perda_batches):
        Z[: len(losses), e] = losses

    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel("Época")
    ax.set_ylabel("Batch")
    ax.set_zlabel("Loss")
    ax.set_title("Loss por batch ao longo das épocas")
    plt.savefig(out_dir / "grafico_mapa3d.png")
    plt.close()

    n_pts = 30
    alphas = torch.linspace(0, 1, n_pts)
    loss_interp = []
    original_state = modelo.state_dict()

    with torch.no_grad():
        for a in alphas:
            vetor_interpolado = pesos_iniciais + a * (pesos_finais - pesos_iniciais)
            offset = 0
            for p in modelo.parameters():
                numel = p.numel()
                p.copy_(vetor_interpolado[offset : offset + numel].view_as(p))
                offset += numel
            total = 0.0
            for batch in loader:
                logits = modelo(batch["input_ids"])
                total += loss_fn(logits, batch["label"]).item()
            loss_interp.append(total)

    modelo.load_state_dict(original_state)

    plt.figure()
    plt.plot(alphas.numpy(), loss_interp)
    plt.xlabel("α (0 = início | 1 = fim)")
    plt.ylabel("Perda")
    plt.title("Curva 1-D da perda ao longo da linha de interpolação dos pesos")
    plt.grid(True)
    plt.savefig(out_dir / "grafico_interpol.png")
    plt.close()

    d1 = (pesos_finais - pesos_iniciais)
    d1 = d1 / d1.norm()
    rand_vec = torch.randn_like(d1)
    d2 = rand_vec - (rand_vec @ d1) * d1
    d2 = d2 / d2.norm()

    grid = torch.linspace(-1, 1, 25)
    loss_surface = np.zeros((len(grid), len(grid)))

    with torch.no_grad():
        for i, a in enumerate(grid):
            for j, b in enumerate(grid):
                w = pesos_finais + a * d1 + b * d2
                offset = 0
                for p in modelo.parameters():
                    numel = p.numel()
                    p.copy_(w[offset : offset + numel].view_as(p))
                    offset += numel
                total = 0.0
                for batch in loader:
                    logits = modelo(batch["input_ids"])
                    total += loss_fn(logits, batch["label"]).item()
                loss_surface[i, j] = total

    modelo.load_state_dict(original_state)

    Xg, Yg = np.meshgrid(grid.numpy(), grid.numpy())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xg, Yg, loss_surface, cmap="coolwarm", linewidth=0)
    ax.set_xlabel("Direção d1")
    ax.set_ylabel("Direção d2")
    ax.set_zlabel("Perda")
    ax.set_title("Superfície 3-D da Loss")
    plt.savefig(out_dir / "grafico_surface.png")
    plt.close()

    for frase_base in frases[-3:]:
        gerar_proxima_palavra(frase_base)

    progresso = {
        "epoca": epoca,
        "batch": step,
        "loss": loss.item(),
        # limite se quiser ↓  ou use todos
        "weights": modelo.embed.weight.flatten()[:200].tolist(),
    }
    historico_buffer.append(progresso)
    _broadcast_progress(progresso)

    hist_path = out_dir / "historico_treino.json"
    with open(hist_path, "w") as f:
        json.dump(historico_buffer, f, indent=2)

    return {
        "loss_final": historico_perda[-1],
        "pngs": {
            "surface_3d": str(out_dir / "grafico_surface.png"),
            "interpol": str(out_dir / "grafico_interpol.png"),  # nome real
            "mapa3d": str(out_dir / "grafico_mapa3d.png"),
            "acuracia": str(out_dir / "grafico_acuracia.png"),
            "perplexidade": str(out_dir / "grafico_perplexidade.png"),
        },
    "hist_json": str(hist_path),          #  ← nova chave

    }


app = FastAPI(title="Mini NN API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

estado = {"treinando": False, "loss_final": None, "pngs": {}}
subscribers: set[WebSocket] = set()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    subscribers.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        subscribers.remove(ws)


def _broadcast_progress(payload: dict):
    txt = json.dumps(payload, default=float)
    for ws in set(subscribers):
        try:
            anyio.from_thread.run(ws.send_text, txt)
        except RuntimeError:
            subscribers.discard(ws)


class FrasesInput(BaseModel):
    frases: list[str]


def _treino_bg(frases: list[str]):
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    estado["treinando"] = True
    resultado = treinar_rede(frases, tmp_dir)
    estado.update(resultado)
    estado["treinando"] = False
    _broadcast_progress({"done": True, **resultado})


@app.post("/treinar")
def treinar(req: FrasesInput, bg: BackgroundTasks):
    if estado.get("treinando"):
        return {"msg": "Já existe treino em andamento."}
    bg.add_task(_treino_bg, req.frases)
    return {"msg": "Treino iniciado!"}


@app.get("/status")
def status():
    return estado


@app.get("/static")
def static_file(path: str):
    p = pathlib.Path(path)
    if not p.exists():
        raise HTTPException(404, f"Arquivo {p.name} ainda não foi gerado.")
    return FileResponse(p)

@app.get("/historico")
def historico(path: str):
    p = pathlib.Path(path)
    if not p.exists():
        raise HTTPException(404, f"Histórico {p.name} não encontrado.")
    return FileResponse(p, media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
