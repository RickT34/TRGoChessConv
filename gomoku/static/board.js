function renderGomokuBoard(element, game, options = {}) {
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const size = game?.size || 19;
  const busy = !!options.busy;
  const percent = (p) => `${(p * 100).toFixed(2)}%`;
  const margin = 34, step = 572 / (size - 1), radius = Math.min(step * 0.43, 28);
  const x = (col) => margin + col * step;
  const y = (row) => margin + row * step;
  const lastMove = game?.history.at(-1)?.action;
  const canPlay = options.interactive && game && !game.done && !busy && game.to_play === game.human;
  const numbers = new Map((game?.history || []).map((m,i) => [m.action, i+1]));
  const probs = game?.probabilities;
  const maxProb = probs ? Math.max(...probs) : 0;
  let svg = `<defs>
    <radialGradient id="black-stone" cx="32%" cy="24%" r="80%"><stop offset="0%" stop-color="#535550"/><stop offset="100%" stop-color="#191e1b"/></radialGradient>
    <radialGradient id="white-stone" cx="32%" cy="24%" r="80%"><stop offset="0%" stop-color="#ffffff"/><stop offset="100%" stop-color="#e9e6db"/></radialGradient>
    <filter id="stone-shadow" x="-30%" y="-30%" width="160%" height="170%"><feDropShadow dx="0.5" dy="1.5" stdDeviation="1" flood-opacity=".2"/></filter>
  </defs>`;
  for (let i = 0; i < size; i++) {
    svg += `<line class="grid-line" x1="${x(i)}" y1="34" x2="${x(i)}" y2="606"/>
      <line class="grid-line" x1="34" y1="${y(i)}" x2="606" y2="${y(i)}"/>
      <text class="coordinate" x="${x(i)}" y="14">${letters[i] || i + 1}</text>
      <text class="coordinate" x="14" y="${y(i)}">${i + 1}</text>`;
  }
  const stars = size === 19 ? [3, 9, 15] : size >= 9 ? [2, Math.floor(size / 2), size - 3] : [];
  for (const row of stars) for (const col of stars)
    svg += `<circle cx="${x(col)}" cy="${y(row)}" r="2.4" fill="#856438"/>`;
  for (let row = 0; row < size; row++) for (let col = 0; col < size; col++) {
    const action = row * size + col, stone = game?.board[row][col] || 0;
    const legal = canPlay && !stone;
    const name = `第 ${row + 1} 行，第 ${col + 1} 列，${stone === 1 ? "黑棋" : stone === -1 ? "白棋" : "空位"}`;
    svg += `<g class="cell" data-action="${action}" data-empty="${!stone}" role="button"
      aria-label="${name}" aria-disabled="${!legal}" tabindex="${legal ? 0 : -1}">
      <title>${name}${!stone && probs ? ` · 当前方落子概率 ${percent(probs[action])}` : ""}</title>
      <circle cx="${x(col)}" cy="${y(row)}" r="${step / 2}" fill="transparent"/>`;
    if (!stone && options.heatmap && probs && maxProb > 0)
      svg += `<circle cx="${x(col)}" cy="${y(row)}" r="${radius * .86}" fill="#2b7655" opacity="${.05 + .5 * probs[action] / maxProb}" pointer-events="none"/>`;
    if (stone) {
      svg += `<circle cx="${x(col)}" cy="${y(row)}" r="${radius}" fill="url(#${stone === 1 ? "black" : "white"}-stone)" stroke="${stone === 1 ? "#20251f" : "#c8c3b4"}" stroke-width=".6" filter="url(#stone-shadow)"/>`;
      if (action === lastMove)
        svg += `<circle cx="${x(col)}" cy="${y(row)}" r="${radius * .34}" fill="none" stroke="${stone === 1 ? "#aed4a2" : "#438058"}" stroke-width="1.8"/>`;
    } else {
      svg += `<circle class="hover-stone" cx="${x(col)}" cy="${y(row)}" r="${radius}" fill="${game?.human === -1 ? "#fff" : "#202921"}"/>
        <circle class="focus-ring" cx="${x(col)}" cy="${y(row)}" r="${radius + 1}" fill="none"/>`;
    }
    if (stone && options.numbers) {
      svg += `<text x="${x(col)}" y="${y(row)}" text-anchor="middle" dominant-baseline="central" fill="${stone === 1 ? '#fff' : '#263c30'}" font-size="${Math.max(8,radius*.8)}" pointer-events="none">${numbers.get(action) || ''}</text>`;
    }
    svg += "</g>";
  }
  element.innerHTML = svg;
  element.setAttribute("aria-label", `${size}×${size} 五子棋棋盘`);
}
