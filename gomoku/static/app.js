const $ = (id) => document.getElementById(id);
const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const sessionKey = "gomoku-session";
let game = null;
let busy = false;
let human = 1;
let availableModels = [];
let messageTimer;

const coordinate = (action, size = game.size) => {
  const row = Math.floor(action / size), col = action % size;
  return `${letters[col] || col + 1}${row + 1}`;
};
const percent = (p) => `${(p * 100).toFixed(2)}%`;

function message(text) {
  clearTimeout(messageTimer);
  $("message").textContent = text;
  $("message").hidden = !text;
  if (text) messageTimer = setTimeout(() => { $("message").hidden = true; }, 7000);
}

async function api(path, data) {
  const response = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败，请检查本地服务。");
  return result;
}

function accept(result) {
  game = result;
  sessionStorage.setItem(sessionKey, game.session);
}

function controls() {
  for (const id of ["model", "mode", "seed", "refresh", "choose-black", "choose-white", "search-depth", "search-time", "search-top-p"])
    $(id).disabled = busy;
  $("new-game").disabled = busy || !availableModels.length;
  $("undo").disabled = busy || !game?.can_undo;
  $("export").disabled = busy || !game?.history.length;
}

async function run(task, status = "模型思考中…") {
  if (busy) return;
  busy = true;
  controls();
  drawBoard();
  $("status").textContent = status;
  message("");
  try { await task(); }
  catch (error) {
    message(error instanceof TypeError ? "连接失败，请确认本地服务正在运行。" : error.message);
  }
  finally { busy = false; render(); }
}

async function refreshModels() {
  const response = await fetch("/api/models");
  if (!response.ok) throw new Error("无法读取模型列表");
  const result = await response.json();
  const previous = $("model").value;
  availableModels = result.models;
  $("model").replaceChildren();
  for (const name of availableModels) {
    const label = name === "residual-history.pt" ? "历史池 PPO · residual-history.pt"
      : name === "residual-distillation-best.pt" ? "蒸馏 · residual-distillation-best.pt" : name;
    $("model").add(new Option(label, name));
  }
  if (!availableModels.length) {
    $("model").add(new Option("还没有可用模型", ""));
    $("settings-note").textContent = "请先训练模型，或在启动时指定模型目录，再刷新列表。";
  } else {
    $("model").value = availableModels.includes(previous) ? previous
      : availableModels.includes("residual-history.pt") ? "residual-history.pt" : availableModels[0];
    $("settings-note").textContent = "设置在开始新对局时生效。";
  }
}

async function newGame() {
  const seed = Number($("seed").value);
  if (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 32)
    throw new Error("随机种子须为 0 到 4294967295 的整数");
  accept(await api("/api/new", {
    session: game?.session, model: $("model").value,
    human, mode: $("mode").value, seed,
    ...($("mode").value === "minimax" ? {search: {
      depth: Number($("search-depth").value), top_p: Number($("search-top-p").value) / 100,
      time_limit: Number($("search-time").value),
    }} : {}),
  }));
}

function selectSide(side) {
  human = side;
  for (const [id, value] of [["choose-black", 1], ["choose-white", -1]]) {
    $(id).classList.toggle("selected", value === human);
    $(id).setAttribute("aria-pressed", String(value === human));
  }
}

function drawBoard() {
  renderGomokuBoard($("board"), game, {busy, interactive:true, heatmap:$("heatmap").checked});
  $("placeholder").hidden = !!game;
}

function render() {
  controls();
  drawBoard();
  if (!game) {
    $("status").textContent = "准备开始一局";
    return;
  }
  const humanBlack = game.human === 1;
  $("black-label").textContent = `黑棋 · ${humanBlack ? "你" : "AI"}`;
  $("white-label").textContent = `白棋 · ${humanBlack ? "AI" : "你"}`;
  $("move-count").textContent = `第 ${game.history.length} 手`;
  $("history-count").textContent = String(game.history.length).padStart(2, "0");
  $("turn-badge").textContent = game.done ? "本局结束" : `你执${humanBlack ? "黑" : "白"}`;
  $("status").textContent = game.done ? game.winner === 0 ? "和棋，势均力敌"
    : game.winner === game.human ? "你赢了，漂亮的一局" : "AI 获胜，再试一种下法"
    : "轮到你落子";
  const modeLabel = game.mode === "minimax" ? `Minimax + αβ · ${game.search.depth} 层 · Top-p ${(game.search.top_p * 100).toFixed(0)}%${game.search.top_p < 1 ? " · 近似搜索" : ""}`
    : game.mode === "greedy" ? "最大概率落子" : "概率采样";
  $("instruction").textContent = `${game.model} · ${modeLabel} · ${game.size} × ${game.size}${game.done ? " · 可以导出棋谱复盘" : " · 点击交叉点落子"}`;
  $("heatmap-note").textContent = $("heatmap").checked
    ? game.done ? "对局结束，已隐藏概率热力图。"
      : `热力图：当前行棋方（${game.to_play === 1 ? "黑棋" : "白棋"} / 你）的策略头偏好，非搜索评分；颜色越深，概率越高。`
    : "连续五子或以上获胜，无禁手。末手以绿色圆环标记。";
  const aiMove = game.history.findLast((move) => move.decision);
  $("analysis-empty").hidden = !!aiMove;
  $("analysis-content").hidden = !aiMove;
  if (aiMove) {
    const decision = aiMove.decision;
    $("ai-coordinate").textContent = coordinate(aiMove.action);
    const search = decision.search;
    $("ai-score-label").textContent = search ? "搜索评分" : "落子概率";
    $("ai-probability").textContent = search ? search.score.toFixed(3) : percent(decision.probability);
    $("search-summary").hidden = !search;
    if (search) $("search-summary").textContent = `完成 ${search.depth}/${search.requested_depth} 层 · ${search.nodes} 节点 · ${search.cutoffs} 次剪枝 · ${search.evaluations} 次网络评估${search.timed_out ? " · 达到时限，采用上一完整深度" : ""}`;
    $("candidate-label").textContent = search ? "搜索前的策略偏好（非搜索排名）" : "当时最偏好的落点";
    $("analysis-note").textContent = search
      ? "评分属于刚落子的 AI：越高越有利，±2 表示当前搜索树的胜负结果；近似搜索可能遗漏分支。价值头评分在 [-1, 1]，不代表胜率。"
      : "概率表示模型偏好，不代表获胜概率。";
    $("ai-time").textContent = `${decision.milliseconds.toFixed(1)} ms`;
    const max = decision.top[0].probability;
    $("candidates").innerHTML = decision.top.map((item, i) => `<li><span class="candidate-rank">${i + 1}</span>
      <span class="candidate-coord">${coordinate(item.action)}</span><div class="bar-track"><div class="bar-fill" style="width:${item.probability / max * 100}%"></div></div>
      <span class="candidate-prob">${percent(item.probability)}</span></li>`).join("");
  }
  $("history").innerHTML = game.history.length ? game.history.map((move, i) => `<div class="history-entry"><span class="history-index">${String(i + 1).padStart(2, "0")}</span>
    <i class="stone ${move.player === 1 ? "black" : "white"}"></i><span class="history-who">${move.player === game.human ? "你" : "AI"}</span>
    <span class="history-coordinate">${coordinate(move.action)}</span></div>`).join("")
    : '<div class="empty-state">棋盘上的每一次选择，都会留在这里。</div>';
  $("history").scrollTop = $("history").scrollHeight;
}

function playAt(cell) {
  if (!cell || busy || !game || cell.getAttribute("aria-disabled") === "true") return;
  run(async () => accept(await api("/api/move", {
    session: game.session, version: game.version, action: Number(cell.dataset.action),
  })));
}

$("board").addEventListener("click", (event) => playAt(event.target.closest(".cell")));
$("board").addEventListener("keydown", (event) => {
  const cell = event.target.closest(".cell");
  if (!cell || !game) return;
  if (["Enter", " "].includes(event.key)) { event.preventDefault(); playAt(cell); }
  const offsets = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -game.size, ArrowDown: game.size };
  if (event.key in offsets) {
    event.preventDefault();
    const action = Number(cell.dataset.action) + offsets[event.key];
    $("board").querySelector(`[data-action="${action}"]`)?.focus();
  }
});
$("choose-black").onclick = () => selectSide(1);
$("choose-white").onclick = () => selectSide(-1);
$("mode").onchange = () => {
  $("seed-field").hidden = $("mode").value !== "sample";
  $("search-fields").hidden = $("mode").value !== "minimax";
};
$("heatmap").onchange = render;
$("refresh").onclick = () => run(refreshModels, "正在读取模型…");
$("new-game").onclick = () => run(newGame, "正在准备棋局…");
$("undo").onclick = () => run(async () => accept(await api("/api/undo", {
  session: game.session, version: game.version,
})), "正在恢复棋局…");
$("export").onclick = () => run(async () => {
  // 由服务直接返回附件，兼容不能下载临时 Blob 地址的内嵌浏览器。
  const link = document.createElement("a");
  link.href = `/api/export?session=${encodeURIComponent(game.session)}`;
  link.download = `gomoku-${new Date().toISOString().replaceAll(":", "-")}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
}, "正在导出棋谱…");

run(async () => {
  await refreshModels();
  const session = sessionStorage.getItem(sessionKey);
  if (session) {
    try {
      accept(await api("/api/state", { session }));
      selectSide(game.human);
      $("mode").value = game.mode;
      $("seed").value = game.seed;
      $("mode").onchange();
      if (game.search) {
        $("search-depth").value = game.search.depth;
        $("search-time").value = game.search.time_limit;
        $("search-top-p").value = String(game.search.top_p * 100);
      }
      if (availableModels.includes(game.model)) $("model").value = game.model;
      return;
    } catch { sessionStorage.removeItem(sessionKey); }
  }
  if (availableModels.length) await newGame();
}, "正在加载模型…");
