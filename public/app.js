/**
 * WALL-E & EVE: Autonomous Mission Control Web Application
 * Handles live 2D grid rendering, Bayesian heatmap overlays,
 * telemetry updates, playback controls, and audio effects.
 */

// Global State
const state = {
  scenario: "sample_mission",
  simData: null,
  currentStep: 0,
  isPlaying: false,
  playTimer: null,
  speed: 800,
  soundEnabled: true,
  showHeatmap: true,
  showGroundTruth: false,
};

// Web Audio API Synthesizer
let audioCtx = null;

function initAudio() {
  if (!audioCtx) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (AudioContext) {
      audioCtx = new AudioContext();
    }
  }
}

function playTone(freq, type = "sine", duration = 0.1, gainVal = 0.05) {
  if (!state.soundEnabled) return;
  try {
    initAudio();
    if (!audioCtx) return;
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(gainVal, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {
    // Ignore audio error
  }
}

function playActionSound(action) {
  if (action.startsWith("MOVE")) {
    playTone(280, "triangle", 0.08, 0.04);
  } else if (action === "SCAN") {
    playTone(880, "sine", 0.25, 0.06);
    setTimeout(() => playTone(1200, "sine", 0.2, 0.05), 100);
  } else if (action === "RECHARGE") {
    playTone(440, "sine", 0.15, 0.04);
    setTimeout(() => playTone(660, "sine", 0.15, 0.04), 80);
    setTimeout(() => playTone(880, "sine", 0.2, 0.05), 160);
  } else if (action === "REPORT_FOUND") {
    playTone(523.25, "triangle", 0.15, 0.08);
    setTimeout(() => playTone(659.25, "triangle", 0.15, 0.08), 120);
    setTimeout(() => playTone(783.99, "triangle", 0.2, 0.08), 240);
    setTimeout(() => playTone(1046.50, "triangle", 0.4, 0.1), 360);
  }
}

// DOM Elements
const elements = {
  scenarioSelect: document.getElementById("scenarioSelect"),
  seedInput: document.getElementById("seedInput"),
  runSimBtn: document.getElementById("runSimBtn"),
  openBenchmarkBtn: document.getElementById("openBenchmarkBtn"),
  soundToggleBtn: document.getElementById("soundToggleBtn"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  stepBackBtn: document.getElementById("stepBackBtn"),
  stepFwdBtn: document.getElementById("stepFwdBtn"),
  resetBtn: document.getElementById("resetBtn"),
  timelineSlider: document.getElementById("timelineSlider"),
  currentStepText: document.getElementById("currentStepText"),
  totalStepsText: document.getElementById("totalStepsText"),
  stepActionText: document.getElementById("stepActionText"),
  speedBtns: document.querySelectorAll(".speed-btn"),
  batteryValText: document.getElementById("batteryValText"),
  batteryFill: document.getElementById("batteryFill"),
  timeValText: document.getElementById("timeValText"),
  timeFill: document.getElementById("timeFill"),
  entropyValText: document.getElementById("entropyValText"),
  entropyFill: document.getElementById("entropyFill"),
  bearingNeedle: document.getElementById("bearingNeedle"),
  signalStrengthText: document.getElementById("signalStrengthText"),
  signalBearingText: document.getElementById("signalBearingText"),
  signalDistText: document.getElementById("signalDistText"),
  totalScansText: document.getElementById("totalScansText"),
  inferredCoordText: document.getElementById("inferredCoordText"),
  inferredCoordProb: document.getElementById("inferredCoordProb"),
  inferredStatusBadge: document.getElementById("inferredStatusBadge"),
  inferredStatusProb: document.getElementById("inferredStatusProb"),
  gridDimText: document.getElementById("gridDimText"),
  heatmapToggle: document.getElementById("heatmapToggle"),
  groundTruthToggle: document.getElementById("groundTruthToggle"),
  gridContainer: document.getElementById("gridContainer"),
  gridViewport: document.getElementById("gridViewport"),
  tileTooltip: document.getElementById("tileTooltip"),
  bannerActionType: document.getElementById("bannerActionType"),
  bannerPosition: document.getElementById("bannerPosition"),
  bannerExplanation: document.getElementById("bannerExplanation"),
  reasoningFeed: document.getElementById("reasoningFeed"),
  cluesFeed: document.getElementById("cluesFeed"),
  cluesCountBadge: document.getElementById("cluesCountBadge"),
  reportContainer: document.getElementById("reportContainer"),
  benchmarkModal: document.getElementById("benchmarkModal"),
  closeBenchmarkBtn: document.getElementById("closeBenchmarkBtn"),
  benchmarkTableBody: document.getElementById("benchmarkTableBody"),
  tabBtns: document.querySelectorAll(".tab-btn"),
  tabContents: document.querySelectorAll(".tab-content"),
  liveStatusText: document.getElementById("liveStatusText"),
  livePill: document.getElementById("livePill"),
};

// Initialization
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadScenario(state.scenario);
});

function setupEventListeners() {
  elements.scenarioSelect.addEventListener("change", (e) => {
    state.scenario = e.target.value;
    loadScenario(state.scenario);
  });

  elements.runSimBtn.addEventListener("click", () => {
    runCustomSimulation();
  });

  elements.playPauseBtn.addEventListener("click", togglePlayPause);
  elements.stepBackBtn.addEventListener("click", () => stepTo(state.currentStep - 1));
  elements.stepFwdBtn.addEventListener("click", () => stepTo(state.currentStep + 1));
  elements.resetBtn.addEventListener("click", () => stepTo(0));

  elements.timelineSlider.addEventListener("input", (e) => {
    stepTo(parseInt(e.target.value, 10));
  });

  elements.speedBtns.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      elements.speedBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.speed = parseInt(btn.dataset.speed, 10);
      if (state.isPlaying) {
        clearInterval(state.playTimer);
        state.playTimer = setInterval(advancePlayStep, state.speed);
      }
    });
  });

  elements.soundToggleBtn.addEventListener("click", () => {
    state.soundEnabled = !state.soundEnabled;
    elements.soundToggleBtn.textContent = state.soundEnabled ? "🔊" : "🔇";
    initAudio();
  });

  elements.heatmapToggle.addEventListener("change", (e) => {
    state.showHeatmap = e.target.checked;
    renderGridCurrentStep();
  });

  elements.groundTruthToggle.addEventListener("change", (e) => {
    state.showGroundTruth = e.target.checked;
    renderGridCurrentStep();
  });

  // Tab switching
  elements.tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      elements.tabBtns.forEach((b) => b.classList.remove("active"));
      elements.tabContents.forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });

  // Benchmark Modal
  elements.openBenchmarkBtn.addEventListener("click", openBenchmarkModal);
  elements.closeBenchmarkBtn.addEventListener("click", () => {
    elements.benchmarkModal.style.display = "none";
  });
  elements.benchmarkModal.addEventListener("click", (e) => {
    if (e.target === elements.benchmarkModal) {
      elements.benchmarkModal.style.display = "none";
    }
  });

  // Keyboard navigation
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      togglePlayPause();
    } else if (e.code === "ArrowRight") {
      stepTo(state.currentStep + 1);
    } else if (e.code === "ArrowLeft") {
      stepTo(state.currentStep - 1);
    }
  });
}

// Data Fetching
async function loadScenario(scenarioName) {
  pause();
  setLiveStatus("LOADING...", "#3b82f6");
  try {
    let res = await fetch(`/api/load_scenario?name=${encodeURIComponent(scenarioName)}`).catch(() => null);
    if (!res || !res.ok) {
      res = await fetch(`data/${scenarioName}.json`);
    }
    if (!res.ok) throw new Error("Failed to load scenario data");
    state.simData = await res.json();
    onDataLoaded();
  } catch (err) {
    console.error("Error loading scenario:", err);
    setLiveStatus("ERROR", "#ef4444");
  }
}

async function runCustomSimulation() {
  pause();
  setLiveStatus("SIMULATING...", "#f59e0b");
  const seed = parseInt(elements.seedInput.value, 10) || 42;
  const payload = {
    scenario: state.scenario,
    seed: seed,
  };

  try {
    let res = await fetch("/api/run_simulation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch(() => null);

    if (!res || !res.ok) {
      res = await fetch(`data/${state.scenario}.json`);
    }
    if (!res.ok) throw new Error("Simulation failed");
    state.simData = await res.json();
    onDataLoaded();
  } catch (err) {
    console.error("Error running simulation:", err);
    setLiveStatus("ERROR", "#ef4444");
  }
}

function onDataLoaded() {
  const { config, snapshots, report } = state.simData;
  const total = snapshots.length - 1;

  elements.timelineSlider.max = total;
  elements.timelineSlider.value = 0;
  elements.totalStepsText.textContent = total;
  elements.gridDimText.textContent = `${config.grid_width} x ${config.grid_height} GRID`;

  // Render reasoning cards list
  renderReasoningCards(snapshots);

  // Render clues feed
  renderCluesFeed();

  // Render mission report
  renderMissionReport(report);

  // Set step to 0
  stepTo(0);
  setLiveStatus("READY", "#10b981");
}

function setLiveStatus(text, color) {
  elements.liveStatusText.textContent = text;
  elements.livePill.style.color = color;
  elements.livePill.style.borderColor = color;
}

// Playback Logic
function togglePlayPause() {
  if (state.isPlaying) {
    pause();
  } else {
    play();
  }
}

function play() {
  if (!state.simData) return;
  initAudio();
  state.isPlaying = true;
  elements.playPauseBtn.textContent = "⏸";
  setLiveStatus("ACTIVE EXPLORATION", "#f59e0b");

  if (state.currentStep >= state.simData.snapshots.length - 1) {
    state.currentStep = 0;
  }

  state.playTimer = setInterval(advancePlayStep, state.speed);
}

function pause() {
  state.isPlaying = false;
  elements.playPauseBtn.textContent = "▶";
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
  setLiveStatus("PAUSED", "#60a5fa");
}

function advancePlayStep() {
  if (!state.simData) return;
  const maxStep = state.simData.snapshots.length - 1;
  if (state.currentStep < maxStep) {
    stepTo(state.currentStep + 1);
  } else {
    pause();
    setLiveStatus("MISSION COMPLETE", "#10b981");
    playTone(1046.50, "sine", 0.5, 0.1);
  }
}

function stepTo(stepIndex) {
  if (!state.simData) return;
  const maxStep = state.simData.snapshots.length - 1;
  const target = Math.max(0, Math.min(stepIndex, maxStep));
  state.currentStep = target;

  elements.timelineSlider.value = target;
  elements.currentStepText.textContent = target;

  const currentSnapshot = state.simData.snapshots[target];
  elements.stepActionText.textContent = currentSnapshot.action;

  // Audio effect
  if (target > 0) {
    playActionSound(currentSnapshot.action);
  }

  // Update telemetry
  updateTelemetry(currentSnapshot);

  // Render Grid
  renderGridCurrentStep();

  // Update Reasoning highlights
  highlightReasoningCard(target);

  // Update Clues
  updateCluesDiscovered(target);
}

// Telemetry Updates
function updateTelemetry(snap) {
  const cfg = state.simData.config;

  // Battery
  const battPct = Math.max(0, Math.min(100, (snap.battery / cfg.battery_capacity) * 100));
  elements.batteryValText.textContent = `${snap.battery.toFixed(1)} / ${cfg.battery_capacity.toFixed(0)} (${battPct.toFixed(0)}%)`;
  elements.batteryFill.style.width = `${battPct}%`;
  if (battPct < 25) {
    elements.batteryFill.style.background = "#ef4444";
  } else if (battPct < 55) {
    elements.batteryFill.style.background = "#f59e0b";
  } else {
    elements.batteryFill.style.background = "linear-gradient(90deg, #ef4444, #f59e0b 40%, #10b981 100%)";
  }

  // Time
  const timePct = Math.max(0, Math.min(100, (snap.time / cfg.time_budget) * 100));
  elements.timeValText.textContent = `${snap.time} / ${cfg.time_budget} ticks`;
  elements.timeFill.style.width = `${timePct}%`;

  // Entropy
  elements.entropyValText.textContent = `${snap.entropy.toFixed(2)} bits`;
  // Normalize entropy roughly for width (initial entropy usually ~ 4-5 bits)
  const entropyPct = Math.min(100, (snap.entropy / 4.5) * 100);
  elements.entropyFill.style.width = `${entropyPct}%`;

  // Signal & Radar
  if (snap.signal && snap.signal.bearing !== null) {
    elements.bearingNeedle.style.transform = `rotate(${snap.signal.bearing}deg)`;
    elements.signalStrengthText.textContent = snap.signal.strength !== null ? snap.signal.strength.toFixed(3) : "--";
    elements.signalBearingText.textContent = `${snap.signal.bearing.toFixed(1)}°`;
    elements.signalDistText.textContent = snap.signal.distance !== null ? `Dist: ${snap.signal.distance.toFixed(1)}m` : "--";
  } else {
    elements.signalStrengthText.textContent = "--";
    elements.signalBearingText.textContent = "--";
    elements.signalDistText.textContent = "--";
  }

  // Total scans up to current step
  const scansUpToNow = state.simData.snapshots.slice(0, state.currentStep + 1).filter(s => s.action === "SCAN").length;
  elements.totalScansText.textContent = scansUpToNow;

  // Inferred Target
  elements.inferredCoordText.textContent = `(${snap.top_candidate[0]}, ${snap.top_candidate[1]})`;
  elements.inferredCoordProb.textContent = `P = ${(snap.top_probability * 100).toFixed(1)}%`;
  elements.inferredStatusBadge.textContent = snap.inferred_status;
  elements.inferredStatusBadge.className = `status-badge ${snap.inferred_status.toLowerCase()}`;
  elements.inferredStatusProb.textContent = `Confidence: ${(snap.status_confidence * 100).toFixed(1)}%`;

  // Banner
  elements.bannerActionType.textContent = snap.action;
  elements.bannerPosition.textContent = `Pos: (${snap.position[0]}, ${snap.position[1]})`;
  elements.bannerExplanation.textContent = snap.explanation;
}

// 2D Grid Visualizer
function renderGridCurrentStep() {
  if (!state.simData) return;
  const { config, snapshots, ground_truth } = state.simData;
  const currentSnap = snapshots[state.currentStep];
  const grid = currentSnap.belief_grid;

  const w = config.grid_width;
  const h = config.grid_height;

  // Trajectory set up to current step
  const trajectoryCoords = new Set();
  for (let i = 0; i <= state.currentStep; i++) {
    trajectoryCoords.add(`${snapshots[i].position[0]},${snapshots[i].position[1]}`);
  }

  const obstacleSet = new Set(config.obstacles.map(([x, y]) => `${x},${y}`));
  const rechargeSet = new Set(config.recharge_stations.map(([x, y]) => `${x},${y}`));
  
  // Discovered clues up to current step
  const discoveredClueCoords = new Set();
  for (let i = 0; i <= state.currentStep; i++) {
    snapshots[i].clues_found.forEach(clue => {
      // Find internal coordinate for visual marking
      const internal = config.clues.find(c => c.clue_id === clue.clue_id);
      if (internal && internal.coordinates) {
        discoveredClueCoords.add(`${internal.coordinates[0]},${internal.coordinates[1]}`);
      }
    });
  }

  elements.gridContainer.style.gridTemplateColumns = `repeat(${w}, 38px)`;
  elements.gridContainer.innerHTML = "";

  // Render from top row (y = h - 1) down to bottom (y = 0)
  for (let y = h - 1; y >= 0; y--) {
    for (let x = 0; x < w; x++) {
      const coordKey = `${x},${y}`;
      const cell = document.createElement("div");
      cell.className = "grid-cell";
      cell.dataset.x = x;
      cell.dataset.y = y;

      const prob = (grid && grid[y] && grid[y][x] !== undefined) ? grid[y][x] : 0.0;
      cell.dataset.prob = prob;

      const isWalle = (currentSnap.position[0] === x && currentSnap.position[1] === y);
      const isObstacle = obstacleSet.has(coordKey);
      const isRecharge = rechargeSet.has(coordKey);
      const isTrail = trajectoryCoords.has(coordKey) && !isWalle;
      const isClue = discoveredClueCoords.has(coordKey);
      const isPeak = (currentSnap.top_candidate[0] === x && currentSnap.top_candidate[1] === y);
      const isEveTrue = (ground_truth.eve_location[0] === x && ground_truth.eve_location[1] === y);

      // Classes
      if (isObstacle) cell.classList.add("obstacle");
      if (isRecharge) cell.classList.add("recharge-station");
      if (isTrail) cell.classList.add("trail");
      if (isClue) cell.classList.add("clue-discovered");
      if (isPeak) cell.classList.add("peak-belief");
      if (isWalle) cell.classList.add("walle-active");
      if (state.showGroundTruth && isEveTrue) cell.classList.add("eve-ground-truth");

      // Heatmap Background Color
      if (state.showHeatmap && !isObstacle) {
        const heatmapColor = getHeatmapColor(prob);
        cell.style.backgroundColor = heatmapColor;
      }

      // Inner Content Icon
      if (isWalle) {
        cell.innerHTML = `<span title="WALL-E" style="filter: drop-shadow(0 0 6px #f59e0b);">🤖</span>`;
      } else if (state.showGroundTruth && isEveTrue) {
        cell.innerHTML = `<span title="EVE Ground Truth" style="filter: drop-shadow(0 0 6px #ec4899);">💫</span>`;
      } else if (isClue) {
        cell.innerHTML = `🔍`;
      } else if (isRecharge) {
        cell.innerHTML = `⚡`;
      } else if (isPeak && prob > 0.15) {
        cell.innerHTML = `<span style="font-size: 0.8rem; color: #10b981;">★</span>`;
      }

      // Tooltip Listeners
      cell.addEventListener("mouseenter", (e) => showTooltip(e, x, y, prob, isObstacle, isRecharge, isClue, isWalle, isEveTrue));
      cell.addEventListener("mouseleave", hideTooltip);

      elements.gridContainer.appendChild(cell);
    }
  }
}

function getHeatmapColor(prob) {
  if (prob < 0.001) return "rgba(30, 41, 59, 0.4)";
  // Color scale from dark purple/blue to vibrant cyan, emerald, amber, and hot pink/white
  if (prob < 0.05) return `rgba(37, 99, 235, ${0.15 + prob * 5})`;
  if (prob < 0.20) return `rgba(6, 182, 212, ${0.3 + prob * 2})`;
  if (prob < 0.50) return `rgba(16, 185, 129, ${0.5 + prob})`;
  if (prob < 0.80) return `rgba(245, 158, 11, ${0.7 + prob * 0.3})`;
  return `rgba(236, 72, 153, 0.95)`;
}

function showTooltip(e, x, y, prob, isObs, isRecharge, isClue, isWalle, isEve) {
  const tt = elements.tileTooltip;
  let statusStr = "Navigable Sector";
  if (isObs) statusStr = "IMPASSABLE OBSTACLE";
  else if (isWalle) statusStr = "WALL-E CURRENT POSITION";
  else if (isEve && state.showGroundTruth) statusStr = `EVE GROUND TRUTH (${state.simData.ground_truth.eve_status})`;
  else if (isClue) statusStr = "DISCOVERED CLUE SITE";
  else if (isRecharge) statusStr = "RECHARGE BASE STATION";

  tt.innerHTML = `
    <div style="font-weight: 700; color: #fff;">TILE (${x}, ${y})</div>
    <div style="color: #94a3b8; font-size: 0.72rem;">${statusStr}</div>
    <div style="color: #06b6d4; margin-top: 4px;">P(EVE at cell) = ${(prob * 100).toFixed(2)}%</div>
  `;
  tt.style.display = "block";

  const rect = e.target.getBoundingClientRect();
  const vpRect = elements.gridViewport.getBoundingClientRect();
  tt.style.left = `${rect.left - vpRect.left + 42}px`;
  tt.style.top = `${rect.top - vpRect.top - 10}px`;
}

function hideTooltip() {
  elements.tileTooltip.style.display = "none";
}

// Reasoning Cards
function renderReasoningCards(snapshots) {
  elements.reasoningFeed.innerHTML = "";
  snapshots.forEach((snap, idx) => {
    const card = document.createElement("div");
    card.className = "reasoning-card";
    card.id = `reasoning-step-${idx}`;
    card.innerHTML = `
      <div class="reasoning-card-header">
        <span class="step-badge">STEP ${String(snap.step_number).padStart(3, "0")}</span>
        <span class="reasoning-action">${snap.action}</span>
      </div>
      <div class="reasoning-body">${snap.explanation}</div>
      <div class="reasoning-footer">
        <span>Pos: (${snap.position[0]}, ${snap.position[1]})</span>
        <span>Batt: ${snap.battery.toFixed(1)}</span>
        <span>H(L): ${snap.entropy.toFixed(2)}</span>
      </div>
    `;
    card.addEventListener("click", () => stepTo(idx));
    elements.reasoningFeed.appendChild(card);
  });
}

function highlightReasoningCard(idx) {
  const cards = elements.reasoningFeed.querySelectorAll(".reasoning-card");
  cards.forEach(c => c.classList.remove("active-step"));
  const activeCard = document.getElementById(`reasoning-step-${idx}`);
  if (activeCard) {
    activeCard.classList.add("active-step");
    activeCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

// Clues Feed
function renderCluesFeed() {
  updateCluesDiscovered(state.currentStep);
}

function updateCluesDiscovered(currentStep) {
  if (!state.simData) return;
  const snapshots = state.simData.snapshots;
  const discoveredMap = new Map();

  for (let i = 0; i <= currentStep; i++) {
    snapshots[i].clues_found.forEach(clue => {
      if (!discoveredMap.has(clue.clue_id)) {
        discoveredMap.set(clue.clue_id, { clue, step: i });
      }
    });
  }

  elements.cluesCountBadge.textContent = discoveredMap.size;

  if (discoveredMap.size === 0) {
    elements.cluesFeed.innerHTML = `<div class="empty-state">No environmental clues discovered yet. Explore or scan nearby sectors.</div>`;
    return;
  }

  elements.cluesFeed.innerHTML = "";
  discoveredMap.forEach(({ clue, step }) => {
    const card = document.createElement("div");
    card.className = "clue-item-card";
    card.innerHTML = `
      <div class="clue-icon-box">🔍</div>
      <div class="clue-details">
        <span class="clue-type-title">${clue.clue_type}</span>
        <p class="clue-desc">${clue.description}</p>
        <div class="clue-meta">Discovered at Step ${step}</div>
      </div>
    `;
    elements.cluesFeed.appendChild(card);
  });
}

// Mission Report
function renderMissionReport(report) {
  if (!report) return;
  const isSuccess = report.eve_found;
  const boxClass = isSuccess ? "success" : "failed";
  const icon = isSuccess ? "🏆" : "❌";

  elements.reportContainer.innerHTML = `
    <div class="report-outcome-box ${boxClass}">
      <span class="outcome-icon">${icon}</span>
      <div>
        <h3 class="outcome-title">${report.mission_status}</h3>
        <p class="outcome-sub">${report.termination_reason}</p>
      </div>
    </div>

    <table class="report-table">
      <tr><td>Total Execution Steps:</td><td>${report.total_steps}</td></tr>
      <tr><td>Active Sensor Scans:</td><td>${report.scans_performed}</td></tr>
      <tr><td>Battery Consumed:</td><td>${report.battery_consumed.toFixed(1)}</td></tr>
      <tr><td>Battery Remaining:</td><td>${report.battery_remaining.toFixed(1)}</td></tr>
      <tr><td>Inferred Coordinates:</td><td>(${report.final_inferred_location[0]}, ${report.final_inferred_location[1]})</td></tr>
      <tr><td>Ground Truth Coordinates:</td><td>(${report.ground_truth_location[0]}, ${report.ground_truth_location[1]})</td></tr>
      <tr><td>Coordinate Error Distance:</td><td>${report.location_error_distance.toFixed(2)} tiles</td></tr>
      <tr><td>Inferred Status:</td><td>${report.final_inferred_status} (${(report.status_confidence * 100).toFixed(1)}%)</td></tr>
      <tr><td>Ground Truth Status:</td><td>${report.ground_truth_status}</td></tr>
    </table>

    <div class="recommendation-box">
      <strong>RECOMMENDED NEXT ACTION</strong>
      ${report.recommended_next_action}
    </div>
  `;
}

// Benchmark Modal
async function openBenchmarkModal() {
  elements.benchmarkModal.style.display = "flex";
  try {
    let res = await fetch("/api/benchmark").catch(() => null);
    if (!res || !res.ok) {
      res = await fetch("data/benchmark_results.json");
    }
    if (!res.ok) throw new Error("Failed to load benchmark results");
    const data = await res.json();
    renderBenchmarkTable(data.summary);
  } catch (err) {
    elements.benchmarkTableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: #ef4444;">Benchmark data unavailable.</td></tr>`;
  }
}

function renderBenchmarkTable(summary) {
  elements.benchmarkTableBody.innerHTML = "";
  for (const [policy, s] of Object.entries(summary)) {
    const row = document.createElement("tr");
    if (policy === "BayesianAutonomous") {
      row.className = "highlight";
    }
    row.innerHTML = `
      <td>${policy === "BayesianAutonomous" ? "⭐ " + policy : policy}</td>
      <td>${s.success_rate_pct.toFixed(1)}%</td>
      <td>${s.budget_exhaustion_rate_pct.toFixed(1)}%</td>
      <td>${s.avg_steps.toFixed(1)}</td>
      <td>${s.avg_scans.toFixed(1)}</td>
      <td>${s.avg_battery_consumed.toFixed(1)}</td>
      <td>${s.avg_location_error.toFixed(2)}</td>
    `;
    elements.benchmarkTableBody.appendChild(row);
  }
}
