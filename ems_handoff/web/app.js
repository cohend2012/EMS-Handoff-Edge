/*
  app.js — HANDOFF.AI screen logic.

  The page holds no business logic. It
    1. calls the local API (/api/...) when a button is pressed, and
    2. follows one Server-Sent Events stream (/api/events) to decide what to show.

  Server events: status, state, listening, caption, level, finished, stage,
  transcript, result, error. See ems_handoff/server.py for the producers.
  Replay mode adds "replay" (the driver) and "incident" (the session) events,
  and tags caption / level / transcript events with replay_burst; those go to
  the replay screen and never touch the live screen's handlers.
*/
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SCREENS = ["home", "live", "working", "report", "error", "replay"];
  const ui = {
    status: null,
    samples: [],
    scriptId: "",
    stageStart: 0,
    workTimer: null,
    lastStageAt: 0,
    suspect: new Set(),       // field paths to mark on the dashboard (replay attribution); empty on the live path
    replay: { callsetsLoaded: false, startedAt: 0, timer: null, bursts: 0, lastChange: "", flashTimer: null },
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
    if (st.version) $("appVersion").textContent = `new app v${st.version}`;
    $("providerText").textContent = `${s.provider} · ${s.model}`;
    const dot = $("providerDot");
    dot.className = "dot " + (s.keys[s.provider] ? "warn" : "bad");
    dot.title = s.keys[s.provider] ? "Not checked yet" : "No API key set";
    const phrase = `“${s.stop_phrase}”`;
    $("phraseHint").textContent = phrase;
    $("phraseTip").textContent = phrase;
    $("engineLine").textContent = `Transcript: ${st.transcriber}. Mic: ${s.mic_device}.`;
    $("lastReportLine").hidden = !st.has_result;
    if (s.replay_mode) {
      $("replayCard").hidden = false;
      $("rpCommit").textContent = st.commit || "";
      if (!ui.replay.callsetsLoaded) loadCallsets();
    }
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
    fixedSeen.delete("liveTranscript");
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
  // Captions with repair: rough words (Vosk, now) are muted; repaired words (Whisper, a few
  // seconds later) are normal and flash green once when they replace the rough ones.
  const fixedSeen = new Map();          // box id -> Set of segment indexes already shown repaired
  function captionHtml(ev, boxId) {
    if (!ev.segments) {
      return `<span class="final">${esc(ev.final)}</span>` + (ev.partial ? ` <span class="partial">${esc(ev.partial)}</span>` : "");
    }
    if (!fixedSeen.has(boxId)) fixedSeen.set(boxId, new Set());
    const seen = fixedSeen.get(boxId);
    const html = ev.segments.filter((s) => s.text).map((s) => {
      const fresh = s.fixed && !seen.has(s.i);
      if (s.fixed) seen.add(s.i);
      return `<span class="cap-seg ${s.fixed ? "fixed" : "rough"}${fresh ? " just-fixed" : ""}">${esc(s.text)}</span>`;
    }).join(" ");
    return html + (ev.partial ? ` <span class="partial">${esc(ev.partial)}</span>` : "");
  }
  function renderCaption(ev) {
    const box = $("liveTranscript");
    const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
    if (!ev.final && !ev.partial && !(ev.segments && ev.segments.length)) return;
    box.innerHTML = captionHtml(ev, "liveTranscript");
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
  const row = (k, v, path) => has(v) ? `<div class="kv${path && ui.suspect.has(path) ? " suspect" : ""}"${path && ui.suspect.has(path) ? ' title="suspect: not stated in the cited burst, or no burst given"' : ""}><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>` : "";
  const list = (arr, cls) => has(arr) ? `<ul>${arr.map((x) => `<li${cls ? ` class="${cls}"` : ""}>${esc(typeof x === "object" ? JSON.stringify(x) : x)}</li>`).join("")}</ul>` : "";
  const card = (title, inner, opt = {}) => inner ? `<div class="dcard${opt.full ? " full" : ""}${opt.crit ? " crit" : ""}${opt.suspect ? " suspect" : ""}"><h3 class="${opt.crit ? "crit" : ""}">${esc(title)}</h3>${inner}</div>` : "";
  const join = (v) => Array.isArray(v) ? v.filter(has).join(", ") : v;

  function renderReport(d) {
    stopWorkTimer();
    ui.suspect = new Set();
    $("dashboard").innerHTML = reportHtml(d);
    show("report");
    window.scrollTo(0, 0);
  }
  function reportHtml(d) {
    d = d || {};
    const level = has(d.alert_level) ? String(d.alert_level).toUpperCase() : "UNKNOWN";
    const s = d.handoff_summary || {}, p = d.patient || {}, v = d.vital_signs || {};
    const a = d.assessment_findings || {}, h = d.history || {}, hr = d.hospital_recommendations || {};
    const q = d.data_quality || {}, meta = d._meta || {};

    let html = `<div class="dash-banner level-${esc(level)}">
      <h2>${esc(d.alert_type || "UNKNOWN")} ALERT · ${esc(level)} PRIORITY</h2>
      <div class="eta${ui.suspect.has("eta_minutes") ? " suspect" : ""}">ETA ${esc(has(d.eta_minutes) ? d.eta_minutes : "?")} min</div>
      ${has(s.one_liner) ? `<div class="one-liner">“${esc(s.one_liner)}”</div>` : ""}
    </div><div class="dash-grid">`;

    html += card("Patient",
      row("Age", has(p.age) ? `${p.age} ${p.age_unit || "years"}` : "", "patient.age") + row("Sex", p.sex, "patient.sex") +
      row("Chief complaint", d.chief_complaint, "chief_complaint") + row("Onset / mechanism", d.mechanism_or_onset, "mechanism_or_onset"));
    html += card("Vital signs",
      row("BP", v.blood_pressure, "vital_signs.blood_pressure") + row("HR", v.heart_rate, "vital_signs.heart_rate") +
      row("RR", v.respiratory_rate, "vital_signs.respiratory_rate") + row("SpO2", v.spo2, "vital_signs.spo2") +
      row("Temp", v.temperature, "vital_signs.temperature") + row("Glucose", v.blood_glucose, "vital_signs.blood_glucose") +
      row("GCS", v.gcs, "vital_signs.gcs"));
    html += card("Assessment",
      row("LOC", a.level_of_consciousness, "assessment_findings.level_of_consciousness") + row("Airway", a.airway, "assessment_findings.airway") +
      row("Breathing", a.breathing, "assessment_findings.breathing") + row("Circulation", a.circulation, "assessment_findings.circulation") +
      row("Neuro", a.neuro, "assessment_findings.neuro") + list(a.other_findings));
    if (has(d.ems_interventions)) {
      html += card("EMS interventions", d.ems_interventions.map((it) =>
        `<div class="intv"><strong>${esc(it.intervention || "")}</strong>` +
        (has(it.details) ? ` <span class="detail">· ${esc(it.details)}</span>` : "") +
        (has(it.response) ? `<div class="resp">→ ${esc(it.response)}</div>` : "") + "</div>").join(""),
        { suspect: ui.suspect.has("ems_interventions") });
    }
    html += card("History",
      row("Medical history", join(h.past_medical), "history.past_medical") + row("Medications", join(h.medications), "history.medications") +
      row("Allergies", join(h.allergies), "history.allergies") + row("Code status", h.code_status, "history.code_status"));
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
    return html;
  }

  // ---------------------------------------------------------------- replay (replay mode only)
  async function loadCallsets() {
    ui.replay.callsetsLoaded = true;
    try {
      const r = await api("/api/replay/callsets");
      const sel = $("callsetSelect");
      sel.innerHTML = r.callsets.map((c) => c.error
        ? `<option value="${esc(c.id)}" disabled>${esc(c.id)} · ${esc(c.error)}</option>`
        : `<option value="${esc(c.id)}">${esc(c.id)} · ${esc(c.bursts)} burst${c.bursts === 1 ? "" : "s"} · ${esc(c.words)} words · ${esc(c.units || "?")} → ${esc(c.hospital || "?")} · ${esc(c.ages || "")} ${esc(c.sex || "")} · ${esc(c.keywords || "")}</option>`).join("");
      $("replayInfo").textContent = `${r.callsets.length} call-sets in ${r.folder}. Bursts play at real time; only the gaps are divided by the speed.`;
    } catch (e) {
      $("replayInfo").textContent = e.message;
    }
  }
  function resetReplayScreen(info) {
    ui.suspect = new Set();
    ui.replay.bursts = 0;
    ui.replay.lastChange = "";
    $("rpCallset").textContent = info?.callset || "";
    $("rpCallsetInfo").textContent = info?.callset_info ? `${info.callset_info.units || "?"} → ${info.callset_info.hospital || "?"} · ${info.callset_info.first_burst_utc || ""} UTC` : "";
    $("rpProvider").textContent = info?.provider || "";
    $("rpModel").textContent = info?.model || "";
    $("rpCommit").textContent = info?.commit || ui.status?.commit || "";
    $("rpBurst").textContent = "0";
    $("rpOf").textContent = info?.driver?.of ?? "0";
    $("rpPhase").textContent = "Starting…";
    $("rpNote").hidden = true;
    $("rpFallback").hidden = true;
    $("rpRecording").hidden = true;
    $("rpCaptions").innerHTML = '<span class="placeholder">Waiting for the first burst…</span>';
    $("rpBurstNow").textContent = "";
    $("rpMeter").style.width = "0%";
    $("rpTranscript").innerHTML = '<span class="placeholder">Bursts appear here as they are transcribed.</span>';
    $("rpSessionLine").textContent = "";
    $("rpUpdating").hidden = true;
    $("rpAsOf").hidden = true;
    $("rpChange").hidden = true;
    $("rpStage").textContent = "";
    $("rpDashboard").innerHTML = '<div class="placeholder-box">No report yet. The first report comes after the quiet period.</div>';
    $("rpDashboard").classList.remove("updating");
    $("rpBackBtn").hidden = true;
    $("rpStopBtn").hidden = false;
    $("rpNewPatientBtn").disabled = false;
    $("rpMergeBtn").disabled = false;
    if (ui.replay.timer) clearInterval(ui.replay.timer);
    ui.replay.startedAt = Date.now();
    ui.replay.timer = setInterval(() => { $("rpElapsed").textContent = clock((Date.now() - ui.replay.startedAt) / 1000); }, 500);
    show("replay");
  }
  // While the recording's tail runs: hold on the final report, then scroll everything that can
  // scroll (the page, the report column) to the bottom at an easy pace, and hold there.
  function tourReport(tailS) {
    const hold = 2.5, settle = 3;
    const dur = Math.max(2, (tailS || 8) - hold - settle);
    const targets = [document.scrollingElement, $("rpDashboard")].filter(el => el && el.scrollHeight - el.clientHeight > 4);
    if (!targets.length) return;
    const starts = targets.map(el => el.scrollTop), maxes = targets.map(el => el.scrollHeight - el.clientHeight);
    setTimeout(() => {
      const t0 = performance.now();
      const step = (now) => {
        const p = Math.min(1, (now - t0) / (dur * 1000));
        const e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
        targets.forEach((el, i) => { el.scrollTop = starts[i] + (maxes[i] - starts[i]) * e; });
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }, hold * 1000);
  }
  window.handoffTourReport = tourReport;
  function endReplayScreen() {
    if (ui.replay.timer) clearInterval(ui.replay.timer);
    ui.replay.timer = null;
    $("rpBackBtn").hidden = false;
    $("rpStopBtn").hidden = true;
  }
  async function startReplay() {
    const callset = $("callsetSelect").value;
    if (!callset) return;
    $("replayBtn").disabled = true;
    try {
      const r = await api("/api/replay/start", { callset, provider: $("replayProvider").value, speed: Number($("replaySpeed").value) || 1,
                                                 record: $("replayRecord").checked });
      resetReplayScreen(r.replay);
    } catch (e) {
      $("replayInfo").textContent = e.message;
    } finally {
      $("replayBtn").disabled = false;
    }
  }
  async function stopReplay() {
    try { await api("/api/replay/stop", {}); } catch (_) { /* already stopped */ }
    endReplayScreen();
    $("rpPhase").textContent = "Replay stopped";
  }
  function flash() {
    // The sync marker goes white in step with the click in the sink audio. 250 ms is about four
    // frames at 15 fps: a two-frame flash vanished when the Jetson's capture dropped a frame
    // (55 of 250 frames in one run under an over-current throttle). Only the onset is measured.
    const m = $("syncMarker");
    m.classList.add("on");
    if (ui.replay.flashTimer) clearTimeout(ui.replay.flashTimer);
    ui.replay.flashTimer = setTimeout(() => m.classList.remove("on"), 250);
  }
  function renderReplay(ev) {
    if (document.body.dataset.screen !== "replay") return;
    if (ev.callset && $("rpCallset").textContent !== ev.callset) { syncReplayScreen(); return; }   // a different run
    $("rpBurst").textContent = ev.burst;
    $("rpOf").textContent = ev.of;
    if (ev.provider) $("rpProvider").textContent = ev.provider;
    if (ev.model) $("rpModel").textContent = ev.model;
    if (ev.elapsed_s != null) ui.replay.startedAt = Date.now() - ev.elapsed_s * 1000;
    if (ev.flash) flash();
    const note = $("rpNote");
    if (ev.note) { note.textContent = ev.note; note.hidden = false; } else if (ev.phase !== "burst") note.hidden = true;
    if (ev.fallback) {
      $("rpFallback").textContent = `${ev.fallback.from} failed (${ev.fallback.reason}); reports from local ${ev.fallback.to}`;
      $("rpFallback").hidden = false;
    }
    if (ev.recording) {
      const rec = ev.recording;
      const sync = rec.sync ? ` · sync ${rec.sync.pass ? "pass" : "FAIL"}${rec.sync.max_abs_offset_ms != null ? ` (worst ${rec.sync.max_abs_offset_ms.toFixed(0)} ms)` : ""}` : "";
      $("rpRecording").textContent = rec.state === "recording" ? `● REC ${rec.path}`
        : rec.state === "done" ? `Recorded ${rec.path}${rec.bytes ? ` · ${(rec.bytes / 1048576).toFixed(1)} MB` : ""}${rec.duration_s ? ` · ${clock(rec.duration_s)}` : ""}${sync}`
        : rec.state === "failed" ? `Recording failed: ${rec.error}` : `Recording ready: ${rec.path}`;
      $("rpRecording").hidden = false;
    }
    const phase = $("rpPhase");
    if (ev.phase === "gap") {
      phase.textContent = ev.burst === 1 && ev.seconds_left <= 0 ? "Starting the first burst…"
        : `Next burst (${ev.burst} of ${ev.of}${ev.next_feed ? ", feed " + ev.next_feed : ""}) in ${ev.seconds_left.toFixed(1)} s`;
    } else if (ev.phase === "burst") {
      phase.textContent = `Burst ${ev.burst} of ${ev.of} playing`;
      $("rpBurstNow").textContent = `· burst ${ev.burst}`;
      $("rpCaptions").innerHTML = '<span class="placeholder">Listening…</span>';
      fixedSeen.delete("rpCaptions");
    } else if (ev.phase === "waiting") {
      phase.textContent = "All bursts played · waiting for the report";
    } else if (ev.phase === "done") {
      phase.textContent = `Replay finished · ${clock(ev.elapsed_s || 0)}`;
      endReplayScreen();
      if (ev.recording && ev.recording.state === "recording") tourReport(ev.recording.tail_s);   // a person would scroll down
    } else if (ev.phase === "error") {
      phase.textContent = `Replay failed: ${ev.error || "unknown error"}`;
      endReplayScreen();
    }
  }
  function renderIncident(ev) {
    if (document.body.dataset.screen !== "replay") return;
    const s = ev.session;
    if (!s) return;
    // the growing transcript, one block per burst
    if (s.bursts.length !== ui.replay.bursts || ev.event === "merged") {
      const box = $("rpTranscript");
      box.innerHTML = s.bursts.map((b, i) => {
        const t = new Date(b.time * 1000);
        const stamp = `${String(t.getUTCHours()).padStart(2, "0")}:${String(t.getUTCMinutes()).padStart(2, "0")}:${String(t.getUTCSeconds()).padStart(2, "0")}`;
        return `<div class="burst-block${i === s.bursts.length - 1 ? " new" : ""}"><div class="burst-head">[Burst ${b.number} · ${stamp} UTC${b.feed ? " · feed " + esc(b.feed) : ""}${b.source_id ? " · " + esc(b.source_id) : ""}] <span class="words">${b.words} words</span></div><div>${esc(b.text) || '<span class="placeholder">(no words)</span>'}</div></div>`;
      }).join("");
      box.scrollTop = box.scrollHeight;
      ui.replay.bursts = s.bursts.length;
    }
    $("rpSessionLine").textContent = `· session ${s.id} · ${s.of} burst${s.of === 1 ? "" : "s"}${s.closed ? " · closed" : ""}`;
    // the report, with the updating marker and the burst count it covers
    const updating = $("rpUpdating");
    updating.hidden = !s.updating;
    if (s.updating) $("rpUpdatingText").textContent = s.report ? `updating · report as of burst ${s.covered} of ${s.of}` : `writing the first report (${s.of} burst${s.of === 1 ? "" : "s"})`;
    $("rpDashboard").classList.toggle("updating", !!s.updating);
    if (s.report) {
      const asof = $("rpAsOf");
      asof.textContent = `Report as of burst ${s.covered} of ${s.of}${s.pending ? " · another run queued" : ""}`;
      asof.hidden = false;
      if (ev.event === "report" || ev.event === "merged") {
        const att = s.report.attribution || {};
        ui.suspect = new Set([...(att.suspect || []), ...(att.unattributed || [])]);
        $("rpDashboard").innerHTML = reportHtml(s.report);
        $("rpDashboard").scrollTop = 0;
      }
    }
    renderState(s.state, s.of);
    const line = s.change_lines[s.change_lines.length - 1];
    if (line && line !== ui.replay.lastChange) {
      const el = $("rpChange");
      el.textContent = line;
      el.hidden = false;
      el.classList.remove("new");
      void el.offsetWidth;
      el.classList.add("new");
      ui.replay.lastChange = line;
    }
    if (ev.event === "error") $("rpStage").textContent = `Extraction failed: ${ev.message}`;
  }
  // The patient state (ems_handoff/state.py): flags with their published rule and source burst, what the
  // declared alert type still lacks, conflicts between bursts, and vital-sign threshold crossings.
  function renderState(st, bursts) {
    const el = $("rpState");
    if (!st || st.error || !bursts) { el.hidden = true; return; }
    const rows = [];
    for (const f of st.flags || []) rows.push(`<div class="st-flag">⚑ ${esc(f.text)} (${esc(f.source)}) <span class="rule">${esc(f.rule)}</span></div>`);
    for (const s of st.series || []) for (const c of s.crossings || []) rows.push(`<div class="st-cross">↕ ${esc(c)}</div>`);
    if ((st.missing || []).length) rows.push(`<div class="st-missing">? ${esc(st.alert_type || "call")}: not stated: ${esc(st.missing.join(", "))}</div>`);
    for (const c of st.conflicts || []) rows.push(`<div class="st-conflict">≠ ${esc(c)}</div>`);
    if (st.news2 && st.news2.partial === false) rows.push(`<div class="st-ok">NEWS2 ${st.news2.score} (${esc(st.news2.band)})</div>`);
    if (!rows.length) rows.push(`<div class="st-ok">state: no flags, nothing missing for ${esc(st.alert_type || "this call")}</div>`);
    el.innerHTML = rows.join("");
    el.hidden = false;
  }

  function renderReplayCaption(ev) {
    const box = $("rpCaptions");
    if (!ev.final && !ev.partial && !(ev.segments && ev.segments.length)) return;
    const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
    box.innerHTML = captionHtml(ev, "rpCaptions");
    if (nearBottom) box.scrollTop = box.scrollHeight;
  }
  function renderReplayLevel(ev) {
    const pct = ev.rms <= 0 ? 0 : Math.min(100, Math.max(0, (Math.log10(ev.rms) + 3) / 2.3 * 100));
    $("rpMeter").style.width = `${pct.toFixed(0)}%`;
    $("rpPhase").textContent = `Burst ${$("rpBurst").textContent} of ${$("rpOf").textContent} playing · ${clock(ev.seconds)}`;
  }
  async function syncReplayScreen() {
    try {
      const r = await api("/api/replay/status");
      if (!r.replay) return;
      resetReplayScreen(r.replay);
      if (r.replay.session) renderIncident({ type: "incident", event: "report", session: r.replay.session });
      renderReplay({ ...r.replay.driver, provider: r.replay.provider, model: r.replay.model, fallback: r.replay.fallback });
    } catch (e) { console.error(e); }
  }

  // ---------------------------------------------------------------- events
  function handle(ev) {
    if (ev.replay_burst) {                 // a replayed burst's listener events: replay screen only
      if (ev.type === "caption") renderReplayCaption(ev);
      else if (ev.type === "level") renderReplayLevel(ev);
      else if (ev.type === "listening") $("rpCaptionEngine").textContent = ev.captions ? `· ${ev.captions}${/repair/i.test(ev.captions) ? " · grey = rough, white = repaired" : ""}` : "";
      else if (ev.type === "transcript") $("rpStage").textContent = `Burst ${ev.replay_burst} transcribed by ${ev.transcriber} in ${ev.transcribe_s} s · ${ev.words} words${ev.notes && ev.notes.length ? " · " + ev.notes.join("; ") : ""}`;
      return;
    }
    if (ev.replay) {                       // a replay stage or error: shown in the replay screen, never the error screen
      if (ev.type === "stage") $("rpStage").textContent = ev.text;
      else if (ev.type === "error") $("rpStage").textContent = ev.message;
      return;
    }
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
          const title = ev.detail ? `Making the handoff report · ${ev.detail}` : "Making the handoff report";
          if (document.body.dataset.screen !== "working") startWorking(title);
          else $("workingTitle").textContent = title;
        } else if (ev.state === "idle" && document.body.dataset.screen === "live") show("home");
        else if (ev.state === "replaying") syncReplayScreen();   // every new run resets the screen, even on the same window
        break;
      case "replay": renderReplay(ev); break;
      case "incident": renderIncident(ev); break;
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
    else if (state === "replaying") syncReplayScreen();
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
    await fillOllamaModels(s);
    $("settingsDialog").showModal();
  }
  // Pulled Ollama models with what each needs; one that does not fit is shown but cannot be saved.
  async function fillOllamaModels(s) {
    const sel = $("ollamaModelSelect");
    sel.dataset.initial = s.provider === "ollama" ? s.model : "";
    try {
      const inv = await api("/api/ollama/models");
      const free = inv.free_gb == null ? "" : ` · ${inv.free_gb.toFixed(1)} GB free`;
      sel.innerHTML = inv.models.map((m) =>
        `<option value="${esc(m.name)}" ${m.name === inv.selected ? "selected" : ""}>${esc(m.name)} · needs ~${m.need_gb.toFixed(1)} GB${m.fits ? "" : " · does not fit"}</option>`).join("");
      sel.dataset.initial = inv.selected;
      sel.title = `Pulled models${free}`;
      sel.disabled = false;
    } catch (e) {
      sel.innerHTML = `<option value="">${esc(e.message)}</option>`;
      sel.disabled = true;
    }
  }
  async function saveSettings(checkAfter) {
    const provider = document.querySelector('input[name="provider"]:checked')?.value;
    const body = { provider, mic_device: $("micSelect").value };
    const model = $("ollamaModelSelect");
    if (!model.disabled && model.value && model.value !== model.dataset.initial) body.ollama_model = model.value;
    const r = await api("/api/settings", body);
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
  $("replayBtn").addEventListener("click", startReplay);
  $("rpStopBtn").addEventListener("click", stopReplay);
  $("rpBackBtn").addEventListener("click", () => show("home"));
  $("rpNewPatientBtn").addEventListener("click", () => api("/api/replay/new_patient", {}).catch((e) => { $("rpStage").textContent = e.message; }));
  $("rpMergeBtn").addEventListener("click", () => api("/api/replay/merge", {}).catch((e) => { $("rpStage").textContent = e.message; }));
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
