//----------------------------------------------------
//  End‑points
//----------------------------------------------------
const API = "http://127.0.0.1:8000";

//----------------------------------------------------
//  Botão “Treinar!”
//----------------------------------------------------
document.getElementById("bt").onclick = async () => {
  const frases = document.getElementById("txt").value
    .split("\n").filter(l => l.trim());
  if (frases.length < 5) {
    alert("Digite pelo menos 5 linhas");
    return;
  }
  await fetch(API + "/treinar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frases })
  });
};

//----------------------------------------------------
//  WebSocket – stream ao vivo
//----------------------------------------------------
let ws;
connectStream();

function connectStream() {
  ws = new WebSocket("ws://127.0.0.1:8000/ws");
  ws.onopen = () => console.log("WS conectado");

  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);

    if (msg.done) {
      // ✅ Recarrega gráfico com histórico completo salvo
      fetch(`/historico?path=${encodeURIComponent(msg.hist_json)}`)
        .then(r => r.json())
        .then(hist => repaintLiveLoss(hist))
        .then(() => showPngs(msg.pngs));
    } else {
      updateCharts(msg);
      updateGraph(msg);
      showStatus(msg);
    }
  };

  ws.onclose = () => setTimeout(connectStream, 2000);
}

//----------------------------------------------------
//  Plotly – linha de perda
//----------------------------------------------------
const trace = { x: [], y: [], mode: 'lines', name: 'loss' };
Plotly.newPlot('liveLoss', [trace], {
  margin: { t: 30 },
  xaxis: { title: 'passo' },
  yaxis: { title: 'loss' }
});

// 🟢 Atualização ao vivo durante o treino
function updateCharts({ loss, batch, epoca }) {
  const passoGlobal = batch + epoca * 1e4;
  Plotly.extendTraces('liveLoss', {
    x: [[passoGlobal]],
    y: [[loss]]
  }, [0], 500);
}

// ✅ Replotagem completa após o fim do treino
function repaintLiveLoss(hist) {
  const xs = hist.map(p => p.batch + p.epoca * 1e4);
  const ys = hist.map(p => p.loss);
  Plotly.newPlot('liveLoss', [{
    x: xs,
    y: ys,
    mode: 'lines',
    name: 'loss'
  }], {
    margin: { t: 30 },
    xaxis: { title: 'passo' },
    yaxis: { title: 'loss' }
  });
}

//----------------------------------------------------
//  Cytoscape – pequena rede
//----------------------------------------------------
const cy = cytoscape({
  container: document.querySelector('#liveNet'),
  elements: buildMiniGraph(),
  layout: { name: 'grid' },
  style: [
    { selector: 'edge', style: { 'width': 2, 'line-color': '#888' } },
    {
      selector: 'node',
      style: {
        'background-color': '#222', 'label': 'data(id)',
        'color': '#fff', 'font-size': 8, 'text-valign': 'center'
      }
    }
  ]
});

function buildMiniGraph() {
  const els = [];
  els.push({ data: { id: 'E' } });
  for (let i = 0; i < 4; i++) {
    els.push({ data: { id: 'H' + i } });
    els.push({ data: { id: 'eh' + i, source: 'E', target: 'H' + i } });
    els.push({ data: { id: 'h' + i + 'o', source: 'H' + i, target: 'Out' } });
  }
  els.push({ data: { id: 'Out' } });
  return els;
}

function updateGraph({ weights = [] }) {
  weights.forEach((w, i) => {
    const e = cy.edges()[i];
    if (!e) return;
    const v = Math.tanh(w);
    e.style('width', 1 + 5 * Math.abs(v));
    e.style('line-color', v > 0 ? 'red' : 'blue');
  });
}

//----------------------------------------------------
//  Exibe JSON simplificado no <pre>
//----------------------------------------------------
function showStatus(obj) {
  document.getElementById("status").textContent =
    JSON.stringify(obj, null, 2);
}

//----------------------------------------------------
//  PNGs finais
//----------------------------------------------------
function showPngs(pngs) {
  const div = document.getElementById('finalImgs');
  div.innerHTML = "";
  for (const [nome, url] of Object.entries(pngs)) {
    const img = document.createElement('img');
    img.src = API + "/static?path=" + encodeURIComponent(url);
    img.alt = nome;
    div.appendChild(img);
  }
}
