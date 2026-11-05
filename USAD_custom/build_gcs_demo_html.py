# -*- coding: utf-8 -*-
"""
Professional Multi-Pane Telemetry Terminal (Bloomberg / TradingView / Grafana Style)
- Replaces AI-card aesthetic with authentic 4-quadrant professional financial/telemetry terminal layout
- Crisp 1px pane dividers, dark theme (#131722, #1e222d), canvas-rendered high-fps time-series curves
- Complete 16-channel real-time contribution watchlist table with in-cell delta bars
- No emojis, no oversized badges, high data density
"""

import json
from pathlib import Path

CUSTOM_DIR = Path(__file__).resolve().parent
JSON_PATH = CUSTOM_DIR / "results" / "web_demo_telemetry_16ch.json"
HTML_OUTPUT = CUSTOM_DIR / "results" / "uav_realtime_gcs_demo.html"


def generate_terminal_html():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        telemetry_data = json.load(f)

    telemetry_json_str = json.dumps(telemetry_data, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ALFA Telemetry Terminal - USAD Anomaly Detection</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {{
      --bg-root: #0d1017;
      --bg-pane: #131722;
      --bg-header: #181c27;
      --bg-hover: #1e2330;
      --border-dark: #202634;
      --border-light: #2a3142;
      --text-main: #d1d4dc;
      --text-muted: #787b86;
      --text-dim: #4c525e;
      --green: #26a69a;
      --red: #ef5350;
      --cyan: #29b6f6;
      --amber: #ffa726;
      --purple: #ab47bc;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      background: var(--bg-root);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, sans-serif;
      font-size: 12px;
      line-height: 1.4;
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}

    .mono {{
      font-family: 'Roboto Mono', monospace;
    }}

    /* Top Navigation Toolbar */
    .top-bar {{
      height: 40px;
      background: var(--bg-header);
      border-bottom: 1px solid var(--border-dark);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 12px;
      gap: 12px;
      flex-shrink: 0;
    }}

    .top-left, .top-center, .top-right {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .brand {{
      font-weight: 700;
      color: #fff;
      font-size: 13px;
      letter-spacing: -0.3px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .brand-tag {{
      background: #2a2e39;
      color: var(--text-muted);
      font-size: 10px;
      padding: 2px 5px;
      border-radius: 2px;
      font-weight: 500;
    }}

    select, button {{
      background: #1e222d;
      border: 1px solid var(--border-dark);
      color: var(--text-main);
      font-size: 11px;
      padding: 4px 8px;
      border-radius: 3px;
      cursor: pointer;
      outline: none;
      font-family: inherit;
    }}

    select:focus, button:hover {{
      border-color: #3b4254;
      background: #252b38;
    }}

    .btn-active {{
      background: #2962ff !important;
      color: #fff !important;
      border-color: #2962ff !important;
    }}

    /* Status Pill */
    .status-pill {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 3px 8px;
      border-radius: 3px;
      font-size: 11px;
      font-weight: 600;
      background: #142823;
      border: 1px solid #1c473c;
      color: var(--green);
    }}

    .status-pill.alarm {{
      background: #2e1719;
      border-color: #632327;
      color: var(--red);
    }}

    .status-dot {{
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
    }}

    /* Time Scrubber */
    .scrubber-wrap {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 1;
      max-width: 420px;
    }}

    input[type=range] {{
      flex: 1;
      height: 3px;
      background: #2a2e39;
      outline: none;
      -webkit-appearance: none;
      border-radius: 2px;
    }}

    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 10px;
      height: 10px;
      background: #2962ff;
      border-radius: 50%;
      cursor: pointer;
    }}

    /* Grid Layout: 4 Panes (2x2 Grid) */
    .workspace-grid {{
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr;
      gap: 2px;
      background: var(--bg-root);
      padding: 2px;
      min-height: 0;
    }}

    .pane {{
      background: var(--bg-pane);
      border: 1px solid var(--border-dark);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      position: relative;
    }}

    .pane-header {{
      height: 28px;
      background: var(--bg-header);
      border-bottom: 1px solid var(--border-dark);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 10px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
      letter-spacing: 0.2px;
      flex-shrink: 0;
    }}

    .pane-title {{
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--text-main);
    }}

    .pane-body {{
      flex: 1;
      position: relative;
      overflow: hidden;
    }}

    /* Table styling for Pane 4 (All 16 channels) */
    .telemetry-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
    }}

    .telemetry-table th {{
      background: #181c27;
      color: var(--text-muted);
      font-weight: 500;
      text-align: left;
      padding: 5px 8px;
      border-bottom: 1px solid var(--border-dark);
      position: sticky;
      top: 0;
      z-index: 2;
    }}

    .telemetry-table td {{
      padding: 4px 8px;
      border-bottom: 1px solid #1a1f2c;
      color: var(--text-main);
      white-space: nowrap;
    }}

    .telemetry-table tr:hover {{
      background: var(--bg-hover);
    }}

    .bar-bg {{
      background: #202634;
      height: 6px;
      border-radius: 1px;
      overflow: hidden;
      width: 100%;
      min-width: 60px;
    }}

    .bar-fill {{
      height: 100%;
      background: #3b82f6;
      transition: width 0.1s linear;
    }}

    .bar-fill.danger {{
      background: var(--red);
    }}

    .bar-fill.warn {{
      background: var(--amber);
    }}

    /* Canvas full size */
    canvas {{
      display: block;
      width: 100%;
      height: 100%;
    }}

    /* Bottom status bar */
    .bottom-bar {{
      height: 24px;
      background: var(--bg-header);
      border-top: 1px solid var(--border-dark);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 10px;
      font-size: 10px;
      color: var(--text-dim);
      flex-shrink: 0;
    }}
  </style>
</head>
<body>

  <!-- 1. SLIM PROFESSIONAL TOP NAVIGATION (TRADINGVIEW / BLOOMBERG STYLE) -->
  <header class="top-bar">
    <div class="top-left">
      <div class="brand">
        <span>ALFA TERMINAL</span>
        <span class="brand-tag mono">USAD ADVERSARIAL FDI</span>
      </div>

      <div style="height: 14px; width: 1px; background: var(--border-light);"></div>

      <select id="flightSelect" class="mono">
      </select>
    </div>

    <!-- Central Playback Controls -->
    <div class="top-center">
      <button id="btnPlay" class="mono">⏸ PAUSE</button>
      <button id="btnReset" class="mono">RESET</button>
      
      <div style="display: flex; gap: 2px;">
        <button class="spd-btn mono btn-active" data-spd="1">1X</button>
        <button class="spd-btn mono" data-spd="2">2X</button>
        <button class="spd-btn mono" data-spd="5">5X</button>
      </div>

      <div class="scrubber-wrap">
        <span id="lblTime" class="mono text-muted" style="min-width: 48px;">000.0s</span>
        <input id="timeSlider" type="range" min="0" max="100" value="0" step="0.1">
      </div>
    </div>

    <div class="top-right">
      <div id="masterStatusPill" class="status-pill mono">
        <span class="status-dot"></span>
        <span id="masterStatusText">NOMINAL (10 Hz)</span>
      </div>
    </div>
  </header>

  <!-- 2. 4-QUADRANT FULL-BLEED WORKSPACE (REFERENCING UPLOADED REAL TERMINAL IMAGES) -->
  <main class="workspace-grid">

    <!-- PANE 1: COMPOSITE ANOMALY SCORE OSCILLOSCOPE (TOP-LEFT) -->
    <div class="pane">
      <div class="pane-header">
        <div class="pane-title mono">
          <span>PANE 1: SYSTEM ANOMALY SCORE [α·MSE(AE₁) + β·MSE(AE₂)]</span>
        </div>
        <div class="mono" style="font-size: 11px;">
          THRESHOLD: <span style="color: var(--red);">0.1953</span> &nbsp;|&nbsp; 
          SCORE: <span id="hdrScore" style="color: #fff; font-weight: 700;">0.0000</span>
        </div>
      </div>
      <div class="pane-body">
        <canvas id="canvasScore"></canvas>
      </div>
    </div>

    <!-- PANE 2: PRIMARY ROOT CAUSE TRACKING (PITCH COMMAND vs ACTUAL vs RECON) (TOP-RIGHT) -->
    <div class="pane">
      <div class="pane-header">
        <div class="pane-title mono">
          <span>PANE 2: PRIMARY ACTUATOR TRACKING [pitch_cmd vs pitch_meas vs RECON]</span>
        </div>
        <div class="mono" style="font-size: 11px; display: flex; gap: 10px;">
          <span style="color: var(--cyan);">■ CMD</span>
          <span style="color: #ffffff;">■ ACTUAL</span>
          <span style="color: var(--amber);">╌╌ RECON</span>
        </div>
      </div>
      <div class="pane-body">
        <canvas id="canvasPitch"></canvas>
      </div>
    </div>

    <!-- PANE 3: COUPLED ROTATIONAL DYNAMICS (PITCH & ROLL RATES) (BOTTOM-LEFT) -->
    <div class="pane">
      <div class="pane-header">
        <div class="pane-title mono">
          <span>PANE 3: MULTI-AXIS ANGULAR DYNAMICS [gyro_y (pitch rate) & gyro_x (roll rate)]</span>
        </div>
        <div class="mono" style="font-size: 11px; display: flex; gap: 10px;">
          <span style="color: var(--green);">■ gyro_y (rad/s)</span>
          <span style="color: var(--purple);">■ gyro_x (rad/s)</span>
        </div>
      </div>
      <div class="pane-body">
        <canvas id="canvasGyro"></canvas>
      </div>
    </div>

    <!-- PANE 4: 16-CHANNEL REAL-TIME SENSOR ATTRIBUTION WATCHLIST (BOTTOM-RIGHT) -->
    <div class="pane">
      <div class="pane-header">
        <div class="pane-title mono">
          <span>PANE 4: 16-VARIABLE REAL-TIME ATTRIBUTION WATCHLIST</span>
        </div>
        <div class="mono" style="font-size: 10px;">
          SORT: <span style="color: var(--cyan); cursor: pointer;" id="btnSortMode">CONTRIBUTION % (DESC)</span>
        </div>
      </div>
      <div class="pane-body" style="overflow-y: auto;">
        <table class="telemetry-table mono">
          <thead>
            <tr>
              <th style="width: 25px;">#</th>
              <th>VARIABLE</th>
              <th style="text-align: right;">MEASURED</th>
              <th style="text-align: right;">RECON</th>
              <th style="text-align: right;">ERROR (Δ²)</th>
              <th style="width: 130px;">CONTRIBUTION (%)</th>
            </tr>
          </thead>
          <tbody id="telemetryTableBody">
            <!-- Dynamically populated at 10Hz -->
          </tbody>
        </table>
      </div>
    </div>

  </main>

  <!-- 3. SLIM FOOTER TELEMETRY AUDIT STRIP -->
  <footer class="bottom-bar mono">
    <div>
      <span>FLIGHT: <span id="ftSequence" style="color: var(--text-main);">-</span></span> &nbsp;|&nbsp;
      <span>ONSET: <span id="ftOnset" style="color: var(--amber);">-</span></span> &nbsp;|&nbsp;
      <span>DIAGNOSIS: <span id="ftDiagnosis" style="color: #fff;">MONITORING</span></span>
    </div>
    <div>
      <span>LATENCY: &lt; 0.8ms</span> &nbsp;|&nbsp;
      <span>SAMPLING: 10.0 Hz</span> &nbsp;|&nbsp;
      <span>CHANNELS: 16 PHYSICAL</span>
    </div>
  </footer>

  <!-- 4. RUNTIME DATA ENGINE -->
  <script>
    const PAYLOAD = {telemetry_json_str};

    let flightKeys = Object.keys(PAYLOAD.flights);
    let currentFlightKey = flightKeys[0];
    let flightData = PAYLOAD.flights[currentFlightKey];
    let currentIndex = 0;
    let isPlaying = true;
    let playSpeed = 1.0;
    let sortRanked = true;

    const channelNames = PAYLOAD.channels;

    // DOM References
    const flightSelect = document.getElementById('flightSelect');
    const btnPlay = document.getElementById('btnPlay');
    const btnReset = document.getElementById('btnReset');
    const timeSlider = document.getElementById('timeSlider');
    const lblTime = document.getElementById('lblTime');
    const masterStatusPill = document.getElementById('masterStatusPill');
    const masterStatusText = document.getElementById('masterStatusText');
    const hdrScore = document.getElementById('hdrScore');
    const telemetryTableBody = document.getElementById('telemetryTableBody');
    const btnSortMode = document.getElementById('btnSortMode');
    const ftSequence = document.getElementById('ftSequence');
    const ftOnset = document.getElementById('ftOnset');
    const ftDiagnosis = document.getElementById('ftDiagnosis');

    // Canvases
    const cScore = document.getElementById('canvasScore');
    const ctxScore = cScore.getContext('2d');
    const cPitch = document.getElementById('canvasPitch');
    const ctxPitch = cPitch.getContext('2d');
    const cGyro = document.getElementById('canvasGyro');
    const ctxGyro = cGyro.getContext('2d');

    // Populate Flight Selector
    flightKeys.forEach(k => {{
      const opt = document.createElement('option');
      opt.value = k;
      opt.textContent = k;
      flightSelect.appendChild(opt);
    }});

    function loadFlight(k) {{
      currentFlightKey = k;
      flightData = PAYLOAD.flights[k];
      currentIndex = 0;
      timeSlider.max = flightData.steps.length - 1;
      ftSequence.textContent = flightData.sequence;
      ftOnset.textContent = flightData.onset_sec ? flightData.onset_sec.toFixed(1) + 's' : 'NONE';
    }}

    flightSelect.addEventListener('change', e => loadFlight(e.target.value));

    // Sort Toggle
    btnSortMode.addEventListener('click', () => {{
      sortRanked = !sortRanked;
      btnSortMode.textContent = sortRanked ? 'CONTRIBUTION % (DESC)' : 'CHANNEL INDEX (1-16)';
      render(Math.floor(currentIndex));
    }});

    // Speed buttons
    document.querySelectorAll('.spd-btn').forEach(btn => {{
      btn.addEventListener('click', e => {{
        document.querySelectorAll('.spd-btn').forEach(b => b.classList.remove('btn-active'));
        e.target.classList.add('btn-active');
        playSpeed = parseFloat(e.target.dataset.spd);
      }});
    }});

    btnPlay.addEventListener('click', () => {{
      isPlaying = !isPlaying;
      btnPlay.textContent = isPlaying ? '⏸ PAUSE' : '▶ PLAY';
    }});

    btnReset.addEventListener('click', () => currentIndex = 0);

    timeSlider.addEventListener('input', e => {{
      currentIndex = parseInt(e.target.value);
      render(currentIndex);
    }});

    // Core 10Hz Render Method
    function render(idx) {{
      const step = flightData.steps[idx];
      if (!step) return;

      timeSlider.value = idx;
      lblTime.textContent = step.t.toFixed(1) + 's';
      hdrScore.textContent = step.score.toFixed(4);

      const th = flightData.threshold;
      const isAlarm = step.alarm;

      // Status Pill
      if (isAlarm) {{
        masterStatusPill.className = "status-pill alarm mono";
        masterStatusText.textContent = `FAULT DETECTED (+${{(step.score - th).toFixed(4)}})`;
        hdrScore.style.color = "var(--red)";
      }} else {{
        masterStatusPill.className = "status-pill mono";
        masterStatusText.textContent = "NOMINAL (10 Hz)";
        hdrScore.style.color = "#ffffff";
      }}

      // Diagnosis String
      if (isAlarm) {{
        const top = step.top1_sensor;
        const pct = step.top1_pct;
        if (top.includes('pitch') || top.includes('gyro_y')) {{
          ftDiagnosis.innerHTML = `<span style="color: var(--red);">ELEVATOR SURFACE JAM / PITCH DECOUPLING (${{top}} ${{pct}}%)</span>`;
        }} else if (top.includes('roll') || top.includes('gyro_x')) {{
          ftDiagnosis.innerHTML = `<span style="color: var(--red);">AILERON DEFLECTION JAM (${{top}} ${{pct}}%)</span>`;
        }} else if (top.includes('speed') || top.includes('throttle') || top.includes('climb')) {{
          ftDiagnosis.innerHTML = `<span style="color: var(--red);">PROPULSION POWER LOSS (${{top}} ${{pct}}%)</span>`;
        }} else {{
          ftDiagnosis.innerHTML = `<span style="color: var(--red);">ANOMALOUS RESIDUAL SPIKE (${{top}} ${{pct}}%)</span>`;
        }}
      }} else {{
        ftDiagnosis.innerHTML = `<span style="color: var(--green);">SYSTEM NORMAL (ALL CHANNELS BOUNDED)</span>`;
      }}

      // Render Table: 16 Channels
      renderTable(step);

      // Render 3 Oscilloscope Canvases
      renderCanvases(idx);
    }}

    function renderTable(step) {{
      let rows = channelNames.map((name, i) => ({{
        idx: i + 1,
        name: name,
        act: step.all_actual[i],
        rec: step.all_recon[i],
        err: step.all_err[i],
        pct: step.all_pcts[i]
      }}));

      if (sortRanked) {{
        rows.sort((a, b) => b.pct - a.pct);
      }}

      let html = '';
      rows.forEach((r, rankIdx) => {{
        const isRoot = step.alarm && rankIdx === 0;
        const isHigh = step.alarm && r.pct > 15.0;

        let rowStyle = "";
        let barClass = "bar-fill";
        let valColor = "var(--text-main)";

        if (isRoot) {{
          rowStyle = 'style="background: rgba(239, 83, 80, 0.12);"';
          barClass = 'bar-fill danger';
          valColor = 'var(--red)';
        }} else if (isHigh) {{
          rowStyle = 'style="background: rgba(255, 167, 38, 0.08);"';
          barClass = 'bar-fill warn';
          valColor = 'var(--amber)';
        }}

        html += `
          <tr ${{rowStyle}}>
            <td style="color: var(--text-dim);">${{String(r.idx).padStart(2, '0')}}</td>
            <td style="font-weight: ${{isRoot ? 700 : 500}}; color: ${{valColor}};">${{r.name}}</td>
            <td style="text-align: right; color: var(--text-muted);">${{r.act.toFixed(3)}}</td>
            <td style="text-align: right; color: var(--text-muted);">${{r.rec.toFixed(3)}}</td>
            <td style="text-align: right; color: ${{isRoot ? 'var(--red)' : 'var(--text-muted)'}};">${{r.err.toFixed(4)}}</td>
            <td>
              <div style="display: flex; align-items: center; gap: 6px;">
                <div class="bar-bg">
                  <div class="${{barClass}}" style="width: ${{Math.min(100, r.pct)}}%;"></div>
                </div>
                <span style="font-size: 10px; min-width: 38px; text-align: right; color: ${{isRoot ? 'var(--red)' : 'var(--text-main)'}};">
                  ${{r.pct.toFixed(1)}}%
                </span>
              </div>
            </td>
          </tr>
        `;
      }});
      telemetryTableBody.innerHTML = html;
    }}

    function renderCanvases(currentIdx) {{
      const wPts = 100;
      const startIdx = Math.max(0, currentIdx - wPts);
      const slice = flightData.steps.slice(startIdx, currentIdx + 1);
      if (slice.length < 2) return;

      // 1. Score Canvas
      drawSinglePlot(cScore, ctxScore, wPts, startIdx, currentIdx, slice, [
        {{ color: '#ef5350', getVal: s => flightData.threshold, isDashed: true, minVal: 0, maxVal: 0.6 }},
        {{ color: '#29b6f6', getVal: s => s.score, minVal: 0, maxVal: 0.6, fill: 'rgba(41, 182, 246, 0.08)' }}
      ]);

      // 2. Pitch Canvas (Cmd vs Meas vs Recon)
      const pCmdIdx = channelNames.indexOf("pitch_cmd");
      const pMeasIdx = channelNames.indexOf("pitch_meas");
      drawSinglePlot(cPitch, ctxPitch, wPts, startIdx, currentIdx, slice, [
        {{ color: '#29b6f6', getVal: s => s.all_actual[pCmdIdx], minVal: 0, maxVal: 1.0 }},
        {{ color: '#ffffff', getVal: s => s.all_actual[pMeasIdx], minVal: 0, maxVal: 1.0 }},
        {{ color: '#ffa726', getVal: s => s.all_recon[pMeasIdx], isDashed: true, minVal: 0, maxVal: 1.0 }}
      ]);

      // 3. Gyro Canvas (gyro_y vs gyro_x)
      const gyIdx = channelNames.indexOf("gyro_y");
      const gxIdx = channelNames.indexOf("gyro_x");
      drawSinglePlot(cGyro, ctxGyro, wPts, startIdx, currentIdx, slice, [
        {{ color: '#26a69a', getVal: s => s.all_actual[gyIdx], minVal: 0, maxVal: 1.0 }},
        {{ color: '#ab47bc', getVal: s => s.all_actual[gxIdx], minVal: 0, maxVal: 1.0 }}
      ]);
    }}

    function drawSinglePlot(c, ctx, wPts, startIdx, currentIdx, slice, curves) {{
      const w = c.width = c.offsetWidth;
      const h = c.height = c.offsetHeight;
      ctx.clearRect(0, 0, w, h);

      // Grid
      ctx.strokeStyle = '#1a1f2b';
      ctx.lineWidth = 1;
      for (let y = 0; y < h; y += 24) {{
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }}

      const stepX = w / (wPts - 1);

      // Fault Onset Zone
      if (flightData.onset_sec !== null) {{
        const onsetIdx = flightData.steps.findIndex(s => s.t >= flightData.onset_sec);
        if (onsetIdx !== -1 && currentIdx >= onsetIdx) {{
          const faultX = Math.max(0, (onsetIdx - startIdx) * stepX);
          ctx.fillStyle = 'rgba(239, 83, 80, 0.06)';
          ctx.fillRect(faultX, 0, w - faultX, h);

          ctx.strokeStyle = 'rgba(239, 83, 80, 0.5)';
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(faultX, 0);
          ctx.lineTo(faultX, h);
          ctx.stroke();
          ctx.setLineDash([]);
        }}
      }}

      curves.forEach(curve => {{
        ctx.strokeStyle = curve.color;
        ctx.lineWidth = 1.3;
        if (curve.isDashed) ctx.setLineDash([3, 3]); else ctx.setLineDash([]);
        ctx.beginPath();

        const range = (curve.maxVal - curve.minVal) || 1.0;
        slice.forEach((s, i) => {{
          const x = (slice.length < wPts ? (wPts - slice.length + i) : i) * stepX;
          const val = curve.getVal(s);
          const normY = (val - curve.minVal) / range;
          const y = h - (normY * (h * 0.8) + (h * 0.1));
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        ctx.stroke();

        if (curve.fill) {{
          ctx.lineTo(w, h);
          const startX = (slice.length < wPts ? (wPts - slice.length) : 0) * stepX;
          ctx.lineTo(startX, h);
          ctx.closePath();
          ctx.fillStyle = curve.fill;
          ctx.fill();
        }}
      }});
    }}

    // Animation Loop
    let lastTime = performance.now();
    function tick(now) {{
      const delta = (now - lastTime) / 1000;
      lastTime = now;

      if (isPlaying && flightData) {{
        const stepAdv = delta * 10 * playSpeed;
        currentIndex = Math.min(flightData.steps.length - 1, currentIndex + stepAdv);
        render(Math.floor(currentIndex));

        if (currentIndex >= flightData.steps.length - 1) {{
          isPlaying = false;
          btnPlay.textContent = '▶ PLAY';
        }}
      }}
      requestAnimationFrame(tick);
    }}

    loadFlight(currentFlightKey);
    requestAnimationFrame(tick);
    window.addEventListener('resize', () => render(Math.floor(currentIndex)));
  </script>
</body>
</html>
"""

    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[SUCCESS] Built Authentic Terminal GCS HTML: {HTML_OUTPUT}")


if __name__ == "__main__":
    generate_terminal_html()
