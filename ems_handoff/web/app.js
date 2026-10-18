/*
  app.js — HANDOFF.AI screen logic.

  The page holds no business logic. It
    1. calls the local API (/api/...) when a button is pressed, and
    2. follows one Server-Sent Events stream (/api/events) to decide what to show.

  Server events: status, state, listening, caption, level, finished, stage,
  transcript, result, error. See ems_handoff/server.py for the producers.
*/
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SCREENS = ["home", "live", "working", "report", "error"];
  const ui = {
    status: null,
    samples: [],
    scriptId: "",
    stageStart: 0,
    workTimer: null,
    lastStageAt: 0,
  };

  // ---------------------------------------------------------------- helpers
  function esc(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function has(v) {
    if (v == null) return false;
    if (Array.isArray(v)) return v.length > 0;
    const s = String(v).trim().toLowerCase();
    return s !== "" && s !== "null" && s !== "unknown" && s !== "none";
  }
  function clock(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }
  function show(name) {
    SCREENS.forEach((s) => { $(`screen-${s}`).hidden = s !== name; });
    document.body.dataset.screen = name;
  }
  async function api(path, body) {
    const opts = body === undefined ? { method: "GET" }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch(path, opts);
    let data = {};
    try { data = await r.json(); } catch (_) { /* empty body */ }
    if (!r.ok || data.ok === false) throw new Error(data.message || `Request failed (${r.status})`);
    return data;
  }
  function showError(message) {
    $("errorText").textContent = message;
    stopWorkTimer();
    show("error");
  }

  // ---------------------------------------------------------------- status chip
  function applyStatus(st) {
    ui.status = st;
    const s = st.settings;
    $("providerText").textContent = `${s.provider} · ${s.model}`;
    const dot = $("providerDot");
    dot.className = "dot " + (s.keys[s.provider] ? "warn" : "bad");
    dot.title = s.keys[s.provider] ? "Not checked yet" : "No API key set";
    const phrase = `“${s.stop_phrase}”`;
    $("phraseHint").textContent = phrase;
    $("phraseTip").textContent = phrase;
    $("engineLine").textContent = `Transcript: ${st.transcriber}. Mic: ${s.mic_device}.`;
    $("lastReportLine").hidden = !st.has_result;
  }
  async function checkProvider() {
    const dot = $("providerDot");
    try {
      const r = await fetch("/api/check", { method: "POST" }).then((x) => x.json());
      dot.className = "dot " + (r.ok ? "ok" : "bad");
      dot.title = r.message;
      return r;
    } catch (e) {
      dot.className = "dot bad";
      dot.title = String(e);
      return { ok: false, message: String(e) };
    }
  }

  // ---------------------------------------------------------------- home
  async function loadSamples() {
    ui.samples = await api("/api/samples");
    const sel = $("scriptSelect");
    sel.innerHTML = '<option value="">No script, I will speak freely</option>' +
      ui.samples.map((s) => `<option value="${esc(s.id)}">${esc(s.title)}</option>`).join("");
    sel.value = "quick";
    $("sampleRow").innerHTML = ui.samples
      .map((s) => `<button class="btn ghost" type="button" data-sample="${esc(s.id)}">${esc(s.title.split(":")[0])}</button>`)
      .join("");
    $("sampleRow").addEventListener("click", (ev) => {
      const id = ev.target.closest("button")?.dataset.sample;
      const sample = ui.samples.find((s) => s.id === id);
      if (sample) $("reportText").value = sample.text;
    });
  }

  async function startRecording() {
    ui.scriptId = $("scriptSelect").value;
    const sample = ui.samples.find((s) => s.id === ui.scriptId);
    $("scriptText").textContent = sample ? sample.text : "";
    $("scriptPanel").hidden = !sample;
    $("liveGrid").classList.toggle("solo", !sample);
    $("liveTranscript").innerHTML = '<span class="placeholder">Listening… start talking.</span>';
    $("recTime").textContent = "0:00";
    $("meterFill").style.width = "0%";
    $("doneBtn").disabled = false;
    $("startBtn").disabled = true;
    try {
      await api("/api/live/start", {});
      show("live");
    } catch (e) {
      showError(e.message);
    } finally {
      $("startBtn").disabled = false;
    }
  }

  async function makeTextReport() {
    const text = $("reportText").value.trim();
    if (!text) { $("reportText").focus(); return; }
    try {
      await api("/api/report/text", { text, source: "text" });
    } catch (e) {
      showError(e.message);
    }
  }

  // ---------------------------------------------------------------- live
  function renderCaption(ev) {
    const box = $("liveTranscript");
    const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
    if (!ev.final && !ev.partial) return;
    box.innerHTML = `<span class="final">${esc(ev.final)}</span>` +
      (ev.partial ? ` <span class="partial">${esc(ev.partial)}</span>` : "");
    if (nearBottom) box.scrollTop = box.scrollHeight;
  }
  function renderLevel(ev) {
    // speech RMS is roughly 0.01-0.2; map it with a log scale so quiet speech still moves the bar
    const pct = ev.rms <= 0 ? 0 : Math.min(100, Math.max(0, (Math.log10(ev.rms) + 3) / 2.3 * 100));
    $("meterFill").style.width = `${pct.toFixed(0)}%`;
    $("recTime").textContent = clock(ev.seconds);
  }
  async function done() {
    $("doneBtn").disabled = true;
    try { await api("/api/live/done", {}); } catch (e) { showError(e.message); }
  }
  async function cancel() {
    try { await api("/api/live/cancel", {}); } catch (_) { /* already stopped */ }
    show("home");
  }

  // ---------------------------------------------------------------- working
  function startWorking(title) {
    $("workingTitle").textContent = title;
    $("stageList").innerHTML = "";
    $("finalTranscriptPanel").hidden = true;
    ui.stageStart = Date.now();
    ui.lastStageAt = Date.now();
    stopWorkTimer();
    ui.workTimer = setInterval(() => {
      $("workTime").textContent = clock((Date.now() - ui.stageStart) / 1000);
    }, 500);
    $("workTime").textContent = "0:00";
    show("working");
  }
  function stopWorkTimer() {
    if (ui.workTimer) clearInterval(ui.workTimer);
    ui.workTimer = null;
  }
  function addStage(text) {
    const list = $("stageList");
    const prev = list.lastElementChild;
    const now = Date.now();
    if (prev) prev.querySelector("span").textContent = `${((now - ui.lastStageAt) / 1000).toFixed(1)} s`;
    ui.lastStageAt = now;
    const li = document.createElement("li");
    li.innerHTML = `${esc(text)}<span></span>`;
    list.appendChild(li);
  }

  // ---------------------------------------------------------------- report
  const row = (k, v) => has(v) ? `<div class="kv"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>` : "";
  const list = (arr, cls) => has(arr) ? `<ul>${arr.map((x) => `<li${cls ? ` class="${cls}"` : ""}>${esc(typeof x === "object" ? JSON.stringify(x) : x)}</li>`).join("")}</ul>` : "";
  const card = (title, inner, opt = {}) => inner ? `<div class="dcard${opt.full ? " full" : ""}${opt.crit ? " crit" : ""}"><h3 class="${opt.crit ? "crit" : ""}">${esc(title)}</h3>${inner}</div>` : "";
  const join = (v) => Array.isArray(v) ? v.filter(has).join(", ") : v;

  function renderReport(d) {
    stopWorkTimer();
    d = d || {};
    const level = has(d.alert_level) ? String(d.alert_level).toUpperCase() : "UNKNOWN";
    const s = d.handoff_summary || {}, p = d.patient || {}, v = d.vital_signs || {};
    const a = d.assessment_findings || {}, h = d.history || {}, hr = d.hospital_recommendations || {};
    const q = d.data_quality || {}, meta = d._meta || {};

    let html = `<div class="dash-banner level-${esc(level)}">
      <h2>${esc(d.alert_type || "UNKNOWN")} ALERT · ${esc(level)} PRIORITY</h2>
      <div class="eta">ETA ${esc(has(d.eta_minutes) ? d.eta_minutes : "?")} min</div>
      ${has(s.one_liner) ? `<div class="one-liner">“${esc(s.one_liner)}”</div>` : ""}
    </div><div class="dash-grid">`;

    html += card("Patient",
      row("Age", has(p.age) ? `${p.age} ${p.age_unit || "years"}` : "") + row("Sex", p.sex) +
      row("Chief complaint", d.chief_complaint) + row("Onset / mechanism", d.mechanism_or_onset));
    html += card("Vital signs",
      row("BP", v.blood_pressure) + row("HR", v.heart_rate) + row("RR", v.respiratory_rate) +
      row("SpO2", v.spo2) + row("Temp", v.temperature) + row("Glucose", v.blood_glucose) + row("GCS", v.gcs));
    html += card("Assessment",
      row("LOC", a.level_of_consciousness) + row("Airway", a.airway) + row("Breathing", a.breathing) +
      row("Circulation", a.circulation) + row("Neuro", a.neuro) + list(a.other_findings));
    if (has(d.ems_interventions)) {
      html += card("EMS interventions", d.ems_interventions.map((it) =>
        `<div class="intv"><strong>${esc(it.intervention || "")}</strong>` +
        (has(it.details) ? ` <span class="detail">· ${esc(it.details)}</span>` : "") +
        (has(it.response) ? `<div class="resp">→ ${esc(it.response)}</div>` : "") + "</div>").join(""));
    }
    html += card("History",
      row("Medical history", join(h.past_medical)) + row("Medications", join(h.medications)) +
      row("Allergies", join(h.allergies)) + row("Code status", h.code_status));
    let prep = row("Activation", hr.activation_requested) + row("Destination", hr.suggested_destination);
    if (has(hr.resources_to_prepare)) prep += `<h3>Resources</h3>${list(hr.resources_to_prepare)}`;
    if (has(hr.time_critical_actions)) prep += `<h3 class="crit">Time-critical</h3>${list(hr.time_critical_actions, "crit")}`;
    html += card("Hospital preparation", prep, { full: true, crit: true });
    let cc = "";
    if (has(s.key_concerns)) cc += `<h3>Key concerns</h3>${list(s.key_concerns, "concern")}`;
    if (has(s.pending_needs)) cc += `<h3>Needed on arrival</h3>${list(s.pending_needs)}`;
    if (cc) html += `<div class="dcard full">${cc}</div>`;
    if (has(d.clinical_guidance)) {
      html += card("Suggested actions (textbook-grounded)", d.clinical_guidance.map((g) =>
        `<div class="guide"><strong>${esc(g.action || "")}</strong>` +
        (has(g.rationale) ? `<div class="muted">${esc(g.rationale)}</div>` : "") +
        (has(g.source) ? `<div class="src">${esc(g.source)}</div>` : "") + "</div>").join(""), { full: true });
    }
    let dq = has(q.confidence) ? `<div class="kv"><span class="k">Confidence</span><span class="v"><span class="pill ${esc(q.confidence)}">${esc(q.confidence)}</span></span></div>` : "";
    dq += row("Missing", join(q.missing_info));
    if (has(q.assumptions)) dq += `<h3>Assumptions</h3>${list(q.assumptions)}`;
    html += card("Data quality", dq, { full: true });
    if (has(meta.transcript)) {
      html += `<div class="dcard full"><details class="transcript-details"><summary>Transcript used for this report</summary><div>${esc(meta.transcript)}</div></details></div>`;
    }
    html += "</div>";
    const t = meta.timings || {};
    const parts = [];
    if (t.transcribe_s != null) parts.push(`transcribe ${t.transcribe_s}s`);
    if (t.extract_s != null) parts.push(`report ${t.extract_s}s`);
    if (t.guidance_s != null) parts.push(`guidance ${t.guidance_s}s`);
    html += `<p class="meta">Decision support only; clinical judgment required. ${esc(meta.provider || "")} · ${esc(meta.model || "")}${parts.length ? " · " + esc(parts.join(", ")) : ""}. Saved to output/handoff_report.json.</p>`;
    $("dashboard").innerHTML = html;
    show("report");
    window.scrollTo(0, 0);
  }

  // ---------------------------------------------------------------- events
  function handle(ev) {
    switch (ev.type) {
      case "status":
        applyStatus(ev.status);
        syncScreen(ev.status.state);
        break;
      case "state":
        if (ev.state === "recording") show("live");
        else if (ev.state === "transcribing") {
          startWorking("Transcribing your report");
          const why = { phrase: `Heard “${ui.status?.settings.stop_phrase || "stop phrase"}”`,
                        button: "Done pressed", timeout: "Time limit reached" }[ui.finishReason];
          if (why) addStage(why);
          ui.finishReason = null;
        }
        else if (ev.state === "processing") {
          if (document.body.dataset.screen !== "working") startWorking("Making the handoff report");
          else $("workingTitle").textContent = "Making the handoff report";
        } else if (ev.state === "idle" && document.body.dataset.screen === "live") show("home");
        break;
      case "listening":
        $("captionEngine").textContent = ev.captions ? `· ${ev.captions}` : "";
        $("micLine").textContent = ev.mic ? `Mic: ${ev.mic}` : "";
        if (ev.problems && ev.problems.length) {
          $("liveTranscript").innerHTML = `<span class="placeholder">${esc(ev.problems.join(" "))} Recording continues.</span>`;
        }
        break;
      case "caption": renderCaption(ev); break;
      case "level": renderLevel(ev); break;
      case "finished":
        ui.finishReason = ev.reason;
        break;
      case "stage": addStage(ev.text); break;
      case "transcript":
        $("finalTranscript").textContent = ev.text;
        $("finalTranscriptPanel").hidden = false;
        break;
      case "result": renderReport(ev.report); break;
      case "error": showError(ev.message); break;
      default: break;
    }
  }

  async function syncScreen(state) {
    if (state === "recording") show("live");
    else if (state === "transcribing" || state === "processing") {
      if (document.body.dataset.screen !== "working") startWorking("Making the handoff report");
    } else if (!document.body.dataset.screen) show("home");
  }

  function connect() {
    const es = new EventSource("/api/events");
    es.onmessage = (m) => { try { handle(JSON.parse(m.data)); } catch (e) { console.error(e); } };
    es.onerror = () => { $("providerDot").className = "dot bad"; $("providerDot").title = "Lost connection to the app server"; };
    es.onopen = () => { checkProvider(); };
  }

  // ---------------------------------------------------------------- settings
  async function openSettings() {
    const st = ui.status || await api("/api/status");
    const s = st.settings;
    const names = { groq: "Groq (cloud, fastest)", ollama: "Ollama (on this device, offline)", openrouter: "OpenRouter (cloud)" };
    $("providerOptions").innerHTML = Object.keys(names).map((p) =>
      `<label class="radio"><input type="radio" name="provider" value="${p}" ${p === s.provider ? "checked" : ""}>
        ${esc(names[p])}${s.keys[p] ? "" : ' <span class="fine">(no API key in .env)</span>'}</label>`).join("");
    $("checkResult").textContent = "";
    $("settingsInfo").textContent = `Transcript: ${st.transcriber}. Live captions: ${st.captions}. Settings are saved for next time.`;
    const mics = await api("/api/mics").catch((e) => ({ devices: [], selected: "auto", error: e.message }));
    $("micSelect").innerHTML = `<option value="">Auto (prefer USB)</option>` +
      mics.devices.map((d) => `<option value="${esc(d.name)}">${esc(d.index)}: ${esc(d.name)}</option>`).join("");
    $("micSelect").value = s.mic_device === "auto" ? "" : s.mic_device;
    $("settingsDialog").showModal();
  }
  async function saveSettings(checkAfter) {
    const provider = document.querySelector('input[name="provider"]:checked')?.value;
    const r = await api("/api/settings", { provider, mic_device: $("micSelect").value });
    applyStatus(r.status);
    if (checkAfter) {
      $("checkResult").textContent = "Checking…";
      const c = await checkProvider();
      $("checkResult").textContent = c.message;
    } else {
      $("settingsDialog").close();
      checkProvider();
    }
  }

  // ---------------------------------------------------------------- wire up
  $("startBtn").addEventListener("click", startRecording);
  $("againBtn").addEventListener("click", startRecording);
  $("textBtn").addEventListener("click", makeTextReport);
  $("doneBtn").addEventListener("click", done);
  $("cancelBtn").addEventListener("click", cancel);
  $("homeBtn").addEventListener("click", () => show("home"));
  $("errorBackBtn").addEventListener("click", () => show("home"));
  $("settingsBtn").addEventListener("click", openSettings);
  $("providerChip").addEventListener("click", openSettings);
  $("checkBtn").addEventListener("click", () => saveSettings(true).catch((e) => { $("checkResult").textContent = e.message; }));
  $("saveSettingsBtn").addEventListener("click", () => saveSettings(false).catch((e) => { $("checkResult").textContent = e.message; }));
  $("showLastBtn").addEventListener("click", async () => {
    const r = await api("/api/result");
    if (r.report) renderReport(r.report);
  });
  $("quitBtn").addEventListener("click", async () => {
    try { await api("/api/quit", {}); document.body.innerHTML = '<main><div class="card error-card"><h1>HANDOFF.AI closed</h1><p>You can close this window.</p></div></main>'; }
    catch (e) { showError(e.message); }
  });
  document.addEventListener("keydown", (e) => {
    // Enter on the live screen = DONE, as a keyboard shortcut
    if (document.body.dataset.screen === "live" && (e.key === "Enter") && !$("doneBtn").disabled) done();
  });

  show("home");
  loadSamples().catch((e) => showError(`Could not load samples: ${e.message}`));
  connect();
})();
