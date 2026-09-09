import time
import threading
import webbrowser
import tempfile
import os
import shutil
import atexit
import logging 
import html
import json

from typing import Optional, Callable
from difflib import SequenceMatcher
 
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
sd.default.device = (24, None)   # input index, output unchanged
from flask import Flask, jsonify
from flask_cors import CORS

from .samples import SAMPLE_STEMI, SAMPLE_TRAUMA, SAMPLE_SEPSIS

_QUICK_DEMO = """
Medic 21 to General Hospital.
We are inbound with a 45-year-old male found down.
GCS is 14. History of Seizures.
Vitals: BP 140 over 90, Heart Rate 110, SpO2 98%.
We have established an IV and given 5mg Versed.
ETA is 5 minutes.
"""

# Dropdown options for the live-recording page: label -> script text.
SAMPLE_SCRIPTS = {
    "Quick demo (short)":   _QUICK_DEMO,
    "STEMI (Heart Attack)": SAMPLE_STEMI,
    "Trauma (MVC)":         SAMPLE_TRAUMA,
    "Sepsis (Medical)":     SAMPLE_SEPSIS,
}

def _script_to_html(text: str) -> str:
    """Render one sample as safe, line-broken HTML for the script box."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return "<br>".join(html.escape(ln) for ln in lines)

_INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EMS Handoff - Live Mode</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; display: flex; justify-content: center; align-items: flex-start; min-height: 100vh; margin: 0; padding: 2rem 1rem; box-sizing: border-box; overflow-y: auto; transition: background-color 0.5s; }
        .container { background-color: #1e1e1e; padding: 2rem; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.5); width: 80%; max-width: 800px; text-align: center; border: 1px solid #333; position: relative; z-index: 10; }
        h1 { color: #fff; margin-bottom: 0.5rem; font-weight: 300; letter-spacing: 1px; }
        .status-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: bold; margin-bottom: 1.5rem; text-transform: uppercase; letter-spacing: 1px; }
        .status-ready { background-color: #333; color: #aaa; }
        .status-recording { background-color: #cf222e; color: white; animation: pulse 1.5s infinite; }
        .status-confirmed { background-color: #238636; color: white; }
        
        .sample-picker { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; text-align: left; }
        .sample-picker label { color: #8b949e; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; white-space: nowrap; }
        .sample-picker select { flex: 1; background-color: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 0.5rem 0.75rem; padding-right: 2.75rem; font-size: 1rem; font-family: inherit; cursor: pointer; -webkit-appearance: none; -moz-appearance: none; appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' fill='none' stroke='%238b949e' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 1rem center; }
        .sample-picker select:focus { outline: none; border-color: #58a6ff; }
        .script-box { background-color: #252525; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #58a6ff; text-align: left; margin-bottom: 1.5rem; font-size: 1.1rem; line-height: 1.6; color: #fff; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2); max-height: 40vh; overflow-y: auto; }
        .transcript-box { background-color: #0d1117; padding: 1.5rem; border-radius: 8px; border: 1px solid #30363d; max-height: 40vh; overflow-y: auto; text-align: left; font-family: 'Consolas', monospace; color: #8b949e; margin-bottom: 2rem; transition: all 0.3s; }
        .transcript-box.active { border-color: #58a6ff; color: #c9d1d9; }
        .transcript-box .new-text { color: #fff; font-weight: bold; }
        
        .controls { display: flex; gap: 1rem; justify-content: center; }
        button { padding: 0.8rem 2rem; border: none; border-radius: 6px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        .btn-record { background-color: #238636; color: white; }
        .btn-record:hover { background-color: #2ea043; transform: translateY(-2px); }
        .btn-stop { background-color: #da3633; color: white; }
        .btn-stop:hover { background-color: #f85149; }
        
        .instruction { color: #8b949e; font-size: 0.9rem; margin-top: 1rem; }
        .magic-word { color: #58a6ff; font-weight: bold; }

        /* Flash Animation for Confirmation */
        @keyframes flashRed {
            0% { background-color: #121212; }
            50% { background-color: #8a1c1c; }
            100% { background-color: #121212; }
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(207, 34, 46, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(207, 34, 46, 0); }
            100% { box-shadow: 0 0 0 0 rgba(207, 34, 46, 0); }
        }
        
        .flashing { animation: flashRed 2s ease-in-out infinite; }
        .confirmed-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(35, 134, 54, 0.95); display: flex; flex-direction: column; justify-content: center; align-items: center; z-index: 100; opacity: 0; pointer-events: none; transition: opacity 0.5s; }
        .confirmed-overlay.show { opacity: 1; pointer-events: auto; }
        .confirmed-text { color: white; font-size: 3rem; font-weight: bold; margin-bottom: 1rem; text-shadow: 0 4px 8px rgba(0,0,0,0.3); }
        .spinner { width: 44px; height: 44px; margin-top: 1.25rem; border: 4px solid rgba(255,255,255,0.35); border-top-color: #fff; border-radius: 50%; animation: spin 0.9s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .final-transcript { margin-top: 1.5rem; max-width: 640px; width: 90%; max-height: 40vh; overflow-y: auto; background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.25); border-radius: 8px; padding: 1rem 1.25rem; text-align: left; }
        .final-transcript .ft-label { color: #fff; opacity: 0.8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin-bottom: 0.5rem; }
        .final-transcript .ft-body { color: #fff; font-family: 'Consolas', monospace; font-size: 0.98rem; line-height: 1.5; white-space: pre-wrap; }

        /* ---- Hospital dashboard (shown after the report is generated) ---- */
        .hidden { display: none !important; }
        .dashboard { width: 100%; max-width: 1080px; margin: 0 auto; text-align: left; }
        .dash-banner { border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1.25rem; color: #fff; box-shadow: 0 8px 24px rgba(0,0,0,0.45); }
        .dash-banner h2 { margin: 0 0 0.25rem; font-size: 1.9rem; font-weight: 700; letter-spacing: 1px; }
        .dash-banner .eta { font-size: 1.05rem; opacity: 0.95; font-weight: 600; }
        .dash-banner .one-liner { margin-top: 0.85rem; font-size: 1.05rem; font-style: italic; opacity: 0.97; }
        .level-RED { background: linear-gradient(135deg, #cf222e, #7d1620); }
        .level-YELLOW { background: linear-gradient(135deg, #d29922, #6f4a00); }
        .level-GREEN { background: linear-gradient(135deg, #238636, #12481f); }
        .level-UNKNOWN { background: linear-gradient(135deg, #444, #222); }
        .dash-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
        .card { background: #1e1e1e; border: 1px solid #333; border-radius: 10px; padding: 1.2rem 1.4rem; }
        .card.full { grid-column: 1 / -1; }
        .card h3 { margin: 0 0 0.7rem; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px; color: #58a6ff; font-weight: 700; }
        .card h3.crit { color: #f85149; }
        .row { display: flex; justify-content: space-between; gap: 1rem; padding: 0.28rem 0; border-bottom: 1px solid #262626; }
        .row:last-child { border-bottom: none; }
        .row .k { color: #8b949e; }
        .row .v { color: #fff; text-align: right; font-weight: 600; }
        .card ul { margin: 0.3rem 0 0; padding-left: 1.15rem; }
        .card li { margin: 0.25rem 0; color: #e0e0e0; }
        .card li.crit { color: #f85149; font-weight: 600; }
        .card li.concern { color: #f0b429; }
        .confidence-pill { display: inline-block; padding: 0.15rem 0.7rem; border-radius: 12px; font-size: 0.8rem; font-weight: 700; }
        .conf-HIGH { background: #238636; color: #fff; }
        .conf-MEDIUM { background: #d29922; color: #1a1a1a; }
        .conf-LOW { background: #cf222e; color: #fff; }
        .intv { padding: 0.4rem 0; border-bottom: 1px solid #262626; }
        .intv:last-child { border-bottom: none; }
        .intv .name { color: #fff; font-weight: 600; }
        .intv .detail { color: #8b949e; font-size: 0.92rem; }
        .intv .resp { color: #3fb950; font-size: 0.92rem; }
        .guidance-item { padding: 0.6rem 0; border-bottom: 1px solid #262626; }
        .guidance-item:last-child { border-bottom: none; }
        .guidance-item .action { color: #fff; font-weight: 600; }
        .guidance-item .rationale { color: #c9d1d9; font-size: 0.92rem; margin-top: 0.2rem; }
        .guidance-item .source { color: #58a6ff; font-size: 0.82rem; margin-top: 0.2rem; }
        .transcript-readout { max-height: 260px; overflow-y: auto; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.25rem; color: #c9d1d9; font-family: 'Consolas', monospace; font-size: 0.95rem; line-height: 1.55; white-space: pre-wrap; }
        .dash-footer { margin-top: 1.25rem; text-align: center; color: #6e7681; font-size: 0.85rem; }
    </style>
</head>
<body>
    <div id="flash-layer"></div>
    <div class="confirmed-overlay" id="confirmOverlay">
        <div class="confirmed-text">REPORT CONFIRMED</div>
        <div id="genStage" style="color: #fff; font-size: 1.2rem;">Generating Handoff...</div>
        <div class="spinner"></div>
        <div id="genTimer" style="color: #fff; opacity: 0.85; font-size: 0.95rem; margin-top: 0.9rem;"></div>
        <div id="finalTranscript" class="final-transcript hidden"></div>
        <div style="color: #fff; opacity: 0.65; font-size: 0.85rem; margin-top: 0.35rem;">Local models can take a few minutes.</div>
    </div>

    <div class="container" id="recordUI">
        <h1>EMS HANDOFF AI</h1>
        <div id="status-badge" class="status-badge status-ready">Ready</div>
        
        <div class="sample-picker">
            <label for="sampleSelect">Sample report:</label>
            <select id="sampleSelect" onchange="updateScript()"></select>
        </div>

        <div class="script-box" id="scriptBox">
            (Choose a sample report above to see the script.)
        </div>
        
        <div class="transcript-box active" id="transcript">
            (Transcript will appear here...)
        </div>
        
        <div class="controls">
            <button id="btnAction" class="btn-record" onclick="toggleRecording()">START RECORDING</button>
        </div>
        
        <p class="instruction">
            Say <span class="magic-word">"CONFIRM REPORT READY"</span> to send automatically.
        </p>
    </div>

    <div class="dashboard hidden" id="dashboard"></div>

    <script>
        // Injected server-side from SAMPLE_SCRIPTS: {label: pre-rendered HTML}
        const SAMPLE_SCRIPTS = __SAMPLE_SCRIPTS_JSON__;

        function updateScript() {
            const sel = document.getElementById('sampleSelect');
            const box = document.getElementById('scriptBox');
            const body = SAMPLE_SCRIPTS[sel.value] || "";
            box.innerHTML = '<strong>📄 READ THIS SCRIPT:</strong><br><br>"' + body + '"';
        }

        function populateSamples() {
            const sel = document.getElementById('sampleSelect');
            Object.keys(SAMPLE_SCRIPTS).forEach((name) => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                sel.appendChild(opt);
            });
            updateScript();
        }

        window.addEventListener('DOMContentLoaded', populateSamples);

        let isRecording = false;
        let pollInterval;

        async function toggleRecording() {
            const btn = document.getElementById('btnAction');
            const status = document.getElementById('status-badge');
            
            if (!isRecording) {
                // Start
                await fetch('/start_recording');
                isRecording = true;
                btn.textContent = "STOP RECORDING";
                btn.className = "btn-stop";
                status.textContent = "🔴 RECORDING";
                status.className = "status-badge status-recording";
                startPolling();
            } else {
                // Stop manually
                await fetch('/stop_recording');
                isRecording = false;
                btn.textContent = "START RECORDING";
                btn.className = "btn-record";
                status.textContent = "Ready";
                status.className = "status-badge status-ready";
                stopPolling();
            }
        }

        function startPolling() {
            pollInterval = setInterval(async () => {
                const response = await fetch('/get_state');
                const data = await response.json();
                
                // Update transcript
                const textBox = document.getElementById('transcript');
                if(data.transcript) {
                    textBox.innerText = data.transcript;
                    textBox.scrollTop = textBox.scrollHeight;
                }
                
                // Check status
                if (data.status === 'confirmed') {
                    stopPolling();
                    isRecording = false;
                    document.getElementById('confirmOverlay').className = "confirmed-overlay show";
                    pollResult();  // wait for main.py to finish extraction, then render
                }
            }, 1000);
        }

        function stopPolling() {
            clearInterval(pollInterval);
        }

        // ---- Wait for the finished handoff, then swap in the dashboard --------
        let resultInterval;
        let resultElapsed = 0;   // seconds since we started waiting (display only)
        let resultFailures = 0;  // consecutive failed polls (server gone?)
        let finalTranscriptText = "";
        function pollResult() {
            resultElapsed = 0;
            resultFailures = 0;
            resultInterval = setInterval(async () => {
                resultElapsed += 1;
                const t = document.getElementById('genTimer');
                if (t) {
                    const m = Math.floor(resultElapsed / 60);
                    const s = String(resultElapsed % 60).padStart(2, '0');
                    t.textContent = 'Elapsed: ' + m + ':' + s;
                }
                try {
                    const r = await fetch('/get_result');
                    if (!r.ok) throw new Error('bad status ' + r.status);
                    const data = await r.json();
                    resultFailures = 0;  // server is alive and working
                    if (data.transcript) {
                        finalTranscriptText = data.transcript;
                        const ft = document.getElementById('finalTranscript');
                        if (ft && ft.classList.contains('hidden')) {
                            ft.innerHTML = '<div class="ft-label">Final transcription</div>' +
                                        '<div class="ft-body">' + esc(data.transcript) + '</div>';
                            ft.classList.remove('hidden');
                        }
                    }
                    if (data.stage) {
                        const st = document.getElementById('genStage');
                        if (st) st.textContent = data.stage;
                    }
                    if (data.ready) {
                        clearInterval(resultInterval);
                        if (data.result && data.result.error) {
                            showError(data.result.error);
                        } else {
                            renderDashboard(data.result);
                        }
                    }
                    // else: extraction still running. Local LLMs can take several
                    // minutes, so we keep waiting as long as the server answers —
                    // elapsed time alone never ends the wait.
                } catch (e) {
                    // Only give up if the server truly stops responding (e.g. the
                    // program was closed), detected as a run of failed polls.
                    resultFailures += 1;
                    if (resultFailures >= 15) {
                        clearInterval(resultInterval);
                        showError("Lost connection to the local server. Check the terminal for the report.");
                    }
                }
            }, 1000);
        }

        function showError(msg) {
            document.getElementById('confirmOverlay').className = "confirmed-overlay";
            const dash = document.getElementById('dashboard');
            document.getElementById('recordUI').className = "container hidden";
            dash.className = "dashboard";
            dash.innerHTML =
                '<div class="dash-banner level-RED"><h2>Report unavailable</h2>' +
                '<div class="one-liner">' + esc(msg) + '</div></div>';
        }

        // ---- Rendering helpers ----------------------------------------------
        function esc(s) {
            return String(s == null ? "" : s)
                .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
        }
        function has(v) {
            if (v == null) return false;
            if (Array.isArray(v)) return v.length > 0;
            const s = String(v).trim();
            return s !== "" && s.toLowerCase() !== "null" && s.toLowerCase() !== "unknown";
        }
        function row(k, v) {
            return has(v) ? '<div class="row"><span class="k">' + esc(k) +
                '</span><span class="v">' + esc(v) + '</span></div>' : "";
        }
        function list(arr, cls) {
            if (!has(arr)) return "";
            const c = cls ? ' class="' + cls + '"' : "";
            return '<ul>' + arr.map(x => '<li' + c + '>' + esc(x) + '</li>').join("") + '</ul>';
        }
        function card(title, inner, opts) {
            if (!inner) return "";
            opts = opts || {};
            const full = opts.full ? " full" : "";
            const critH = opts.crit ? " crit" : "";
            return '<div class="card' + full + '"><h3 class="' + critH.trim() + '">' +
                esc(title) + '</h3>' + inner + '</div>';
        }

        function renderDashboard(d) {
            d = d || {};
            const overlay = document.getElementById('confirmOverlay');
            overlay.className = "confirmed-overlay";
            document.body.className = "";
            document.getElementById('recordUI').className = "container hidden";

            const level = (d.alert_level || "UNKNOWN");
            const s = d.handoff_summary || {};
            const p = d.patient || {};
            const v = d.vital_signs || {};
            const a = d.assessment_findings || {};
            const h = d.history || {};
            const hr = d.hospital_recommendations || {};
            const q = d.data_quality || {};

            // Banner
            let html = '<div class="dash-banner level-' + esc(level) + '">' +
                '<h2>' + esc(d.alert_type || "UNKNOWN") + ' ALERT — ' + esc(level) + ' PRIORITY</h2>' +
                '<div class="eta">ETA: ' + esc(has(d.eta_minutes) ? d.eta_minutes : "?") + ' minutes</div>';
            if (has(s.one_liner)) html += '<div class="one-liner">"' + esc(s.one_liner) + '"</div>';
            html += '</div>';

            html += '<div class="dash-grid">';

            // Patient
            html += card("Patient",
                row("Age", has(p.age) ? (p.age + " " + (p.age_unit || "years")) : "") +
                row("Sex", p.sex) +
                row("Chief complaint", d.chief_complaint) +
                row("Onset / mechanism", d.mechanism_or_onset));

            // Vitals
            html += card("Vital Signs",
                row("BP", v.blood_pressure) + row("HR", v.heart_rate) +
                row("RR", v.respiratory_rate) + row("SpO2", v.spo2) +
                row("Temp", v.temperature) + row("BGL", v.blood_glucose) +
                row("GCS", v.gcs));

            // Assessment
            let assess = row("LOC", a.level_of_consciousness) + row("Airway", a.airway) +
                row("Breathing", a.breathing) + row("Circulation", a.circulation) +
                row("Neuro", a.neuro);
            if (has(a.other_findings)) assess += list(a.other_findings);
            html += card("Assessment", assess);

            // Interventions
            if (has(d.ems_interventions)) {
                const items = d.ems_interventions.map(it => {
                    let s2 = '<div class="intv"><span class="name">' + esc(it.intervention || "") + '</span>';
                    if (has(it.details)) s2 += ' <span class="detail">— ' + esc(it.details) + '</span>';
                    if (has(it.response)) s2 += '<div class="resp">→ ' + esc(it.response) + '</div>';
                    return s2 + '</div>';
                }).join("");
                html += card("EMS Interventions", items);
            }

            // History
            html += card("History",
                row("PMH", (h.past_medical || []).join(", ")) +
                row("Meds", (h.medications || []).join(", ")) +
                row("Allergies", (h.allergies || []).join(", ")) +
                row("Code status", h.code_status));

            // Hospital prep
            let prep = row("Activation", hr.activation_requested) +
                row("Destination", hr.suggested_destination);
            if (has(hr.resources_to_prepare)) prep += '<h3 style="margin-top:0.8rem;">Resources</h3>' + list(hr.resources_to_prepare);
            if (has(hr.time_critical_actions)) prep += '<h3 class="crit" style="margin-top:0.8rem;">Time-critical</h3>' + list(hr.time_critical_actions, "crit");
            html += card("Hospital Preparation", prep, {full: true, crit: true});

            // Concerns / pending
            let cc = "";
            if (has(s.key_concerns)) cc += '<h3>Key concerns</h3>' + list(s.key_concerns, "concern");
            if (has(s.pending_needs)) cc += '<h3 style="margin-top:0.8rem;">Pending on arrival</h3>' + list(s.pending_needs);
            if (cc) html += '<div class="card full">' + cc + '</div>';

            // Clinical guidance (optional)
            if (has(d.clinical_guidance)) {
                const g = d.clinical_guidance.map(x => {
                    if (typeof x !== "object") return '<div class="guidance-item"><div class="action">' + esc(x) + '</div></div>';
                    let s3 = '<div class="guidance-item"><div class="action">' + esc(x.action || "") + '</div>';
                    if (has(x.rationale)) s3 += '<div class="rationale">' + esc(x.rationale) + '</div>';
                    if (has(x.source)) s3 += '<div class="source">' + esc(x.source) + '</div>';
                    return s3 + '</div>';
                }).join("");
                html += card("Suggested Courses of Action (textbook-grounded)", g, {full: true});
            }

            // Data quality
            let dq = "";
            if (has(q.confidence)) dq += '<div class="row"><span class="k">Confidence</span>' +
                '<span class="v"><span class="confidence-pill conf-' + esc(q.confidence) + '">' + esc(q.confidence) + '</span></span></div>';
            dq += row("Missing", (q.missing_info || []).join(", "));
            if (has(q.assumptions)) dq += '<h3 style="margin-top:0.8rem;">Assumptions</h3>' + list(q.assumptions);
            html += card("Data Quality", dq, {full: true});

            // Final transcript (raw full-audio transcription, for review)
            if (finalTranscriptText) {
                html += '<div class="card full"><h3>Final Transcript</h3>' +
                    '<div class="transcript-readout">' + esc(finalTranscriptText) + '</div></div>';
            }

            html += '</div>';  // .dash-grid
            html += '<div class="dash-footer">Decision support only — clinical judgment required. ' +
                    'Full JSON saved to output/handoff_report.json<br><br>' +
                    '<button class="btn-record" onclick="recordAnother()">RECORD ANOTHER REPORT</button>' +
                    '<button class="btn-stop" style="margin-left:0.75rem;" onclick="finishSession()">FINISH</button></div>';

            const dash = document.getElementById('dashboard');
            dash.innerHTML = html;
            dash.className = "dashboard";
            window.scrollTo(0, 0);
        }

        // Client-side reset back to the record screen for another recording.
        // No reload: the server keeps looping in run_live_session(), and the
        // next START RECORDING clears the previous result server-side.
        function recordAnother() {
            clearInterval(resultInterval);
            const dash = document.getElementById('dashboard');
            dash.className = "dashboard hidden";
            dash.innerHTML = "";
            document.getElementById('confirmOverlay').className = "confirmed-overlay";
            document.getElementById('recordUI').className = "container";
            const tb = document.getElementById('transcript');
            tb.innerText = "(Transcript will appear here...)";
            const badge = document.getElementById('status-badge');
            badge.textContent = "Ready"; badge.className = "status-badge status-ready";
            const btn = document.getElementById('btnAction');
            btn.textContent = "START RECORDING"; btn.className = "btn-record";
            isRecording = false;
            const t = document.getElementById('genTimer'); if (t) t.textContent = "";
            const ft = document.getElementById('finalTranscript');
            if (ft) { ft.className = "final-transcript hidden"; ft.innerHTML = ""; }
            finalTranscriptText = "";
            const st = document.getElementById('genStage');
            if (st) st.textContent = "Generating handoff…";
            window.scrollTo(0, 0);
        }

        async function finishSession() {
            clearInterval(resultInterval);
            try { await fetch('/finish'); } catch (e) { /* ignore */ }
            document.getElementById('confirmOverlay').className = "confirmed-overlay";
            document.getElementById('recordUI').className = "container hidden";
            const dash = document.getElementById('dashboard');
            dash.className = "dashboard";
            dash.innerHTML =
                '<div class="dash-banner level-GREEN"><h2>Session complete</h2>' +
                '<div class="one-liner">You can close this tab and return to the terminal.</div></div>';
        }
    </script>
</body>
</html>
"""

# Pre-render {label: html} once, then inject it into the page as a JSON blob the dropdown reads. 
_SAMPLE_SCRIPTS_HTML = {name: _script_to_html(txt) for name, txt in SAMPLE_SCRIPTS.items()}

def _render_index() -> str:
    """Return the page HTML with the sample scripts injected for the dropdown."""
    payload = json.dumps(_SAMPLE_SCRIPTS_HTML).replace("</", "<\\/")
    return _INDEX_HTML.replace("__SAMPLE_SCRIPTS_JSON__", payload)

TIMEOUT_SECONDS = 10 * 60
WARNING_THRESHOLD = 3 * 60
SAMPLE_RATE = 48000
CHUNK_DURATION = 3.0
MIN_AUDIO_LEVEL = 0.01
KEYWORD_WINDOW_SECONDS = 6.0

logger = logging.getLogger(__name__)

class RecordingManager:
    """Manages audio recording with timeout and cleanup."""
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="handoff_")
        self.files_to_cleanup = []
        logger.info(f"Created temp directory: {self.temp_dir}")
        atexit.register(self.cleanup)
    
    def create_temp_file(self, suffix: str = ".wav") -> str:
        """Create a temp file path."""
        filepath = os.path.join(self.temp_dir, f"recording{suffix}")
        self.files_to_cleanup.append(filepath)
        return filepath
    
    def save_final(self, final_file: Optional[str]) -> None:
        """Save final recording to output folder."""
        if not final_file or not os.path.exists(final_file):
            return
        
        try:
            # Copy the temp recording into the persistent output folder
            os.makedirs("output", exist_ok=True)
            output_filename = "output/final_recording.wav"
            shutil.copy(final_file, output_filename)
            logger.info(f"Saved final recording to: {output_filename}")
            print(f"[SAVE] Recording: {output_filename}")
        except Exception as e:
            logger.warning(f"Could not save to output/: {e}")
    
    def cleanup(self, final_file: Optional[str] = None) -> None: 
        """Clean up all temp files except the final recording."""
        for filepath in self.files_to_cleanup: 
            if final_file and filepath == final_file:
                continue
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Cleaned up: {filepath}")
            except Exception as e: 
                logger.warning(f"Could not clean up {filepath}: {e}")

        try: 
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e: 
            logger.warning(f"Could not clean up temp directory: {e}")

class KeywordDetector: 
    """Detects confirmation keyword with fuzzy matching."""

    def __init__(self, keyword: str = "confirm report ready", threshold: float = 0.85):
        """
        Initialize keyword detector. 

        Args: 
            keyword: phrase to detect
            threshold: similarity threshold 
        """
        self.keywords = [
            keyword.lower(),
            "confirm report",
            "report ready"
        ]
        self.threshold = threshold

    def _matches(self, phrase: str, text: str) -> bool:
        """Check if phrase matches text (exact or fuzzy)."""
        if phrase in text:
            return True
        similarity = SequenceMatcher(None, phrase, text).ratio()
        return similarity > self.threshold

    def detect(self, text: str) -> bool: 
        """
        Detect if text contains confirmation keyword (with fuzzy matching).

        Args: 
            text: transcribed text to check 

        Returns: 
            True or False 
        """
        text = text.lower()
        
        for keyword in self.keywords:
            if self._matches(keyword, text):
                logger.info(f"Keyword detected: {keyword}")
                return True
        
        return False
    
class AudioValidator: 
    """Validates audio quality during recording."""
    @staticmethod
    def get_rms_level(audio_chunk: np.ndarray) -> float:
        """Calculate RMS (root mean square) level of audio chunk."""
        return float(np.sqrt(np.mean(audio_chunk**2)))
    
    @staticmethod
    def is_silent(audio_chunk: np.ndarray, threshold: float = MIN_AUDIO_LEVEL) -> bool: 
        """Check if audio chunk is essentially silent."""
        rms = AudioValidator.get_rms_level(audio_chunk)
        return rms < threshold

recording_manager = RecordingManager()

class RecordingState:
    """Manages state of audio recording session."""
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear all fields back to a fresh pre-recording state.

        Called at the start of every start_web_ui() run so a second case
        does not inherit 'confirmed' status or the previous final_file.
        """
        self.is_recording = False
        self.transcript = ""
        self.status = "ready"  # ready, recording, confirmed
        self.final_file = None
        self.audio_level = 0.0
        self.start_time = None
        self.result = None     # handoff dict pushed in after extraction; None until ready
        self.final_transcript = None
        self.stage = None
        self.session_active = False  
        self.recording_seq = 0  # bumped once per confirmed recording (loop discriminator)

def _run_recording_worker(state, transcribe_chunk_fn) -> None:
    """
    Background thread: record continuously, transcribe periodically.
 
    Audio capture and transcription are decoupled. The InputStream callback
    appends to `frames` on PortAudio's thread; the loop below snapshots that
    buffer every CHUNK_DURATION seconds to check for the stop keyword. A slow
    transcription delays the next *check*, never the *recording*.
    """
    frames = []                     
    lock = threading.Lock()
    silent_checks = 0
    mic_issue_warned = False
 
    def on_audio(indata, n_frames, time_info, status):
        if status:
            logger.debug(f"Audio stream status: {status}")
        with lock:
            frames.append(indata.copy())
 
    def snapshot() -> np.ndarray:
        with lock:
            if not frames:
                return np.array([], dtype=np.float32)
            return np.concatenate(frames).flatten()
 
    try:
        logger.info("Recording started (continuous stream)")
        keyword_detector = KeywordDetector()         
 
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=on_audio,
        ):
            next_check = time.time() + CHUNK_DURATION
 
            while state.is_recording:
                # Poll cheaply so a stop request is honored promptly
                time.sleep(0.1)
 
                if state.start_time and time.time() - state.start_time > TIMEOUT_SECONDS:
                    logger.warning("Recording timeout - auto-stopping")
                    state.is_recording = False
                    break
 
                if time.time() < next_check:
                    continue
                next_check = time.time() + CHUNK_DURATION
 
                audio = snapshot()
                if audio.size == 0:
                    continue
 
                recent = audio[-int(CHUNK_DURATION * SAMPLE_RATE):]
                level = AudioValidator.get_rms_level(recent) if recent.size else 0.0
                state.audio_level = level
                if level < MIN_AUDIO_LEVEL: 
                    silent_checks += 1
                    if silent_checks > 2 and not mic_issue_warned:
                        logger.warning("Audio level very low - check microphone")
                        mic_issue_warned = True
                else:
                    silent_checks = 0
 
                # keyword detection on a trailing window 
                window = int(KEYWORD_WINDOW_SECONDS * SAMPLE_RATE)
                check_data = audio[-window:] if audio.size > window else audio
 
                temp_check = recording_manager.create_temp_file("_check.wav")  
                wav.write(temp_check, SAMPLE_RATE, check_data)
 
                new_text = transcribe_chunk_fn(temp_check)   # stream keeps running
                if not new_text:
                    continue
 
                if len(state.transcript) == 0:
                    state.transcript = new_text
                elif len(new_text) > 3:
                    state.transcript += f" {new_text}"
 
                if keyword_detector.detect(new_text):
                    logger.info("Confirmation keyword detected - ending recording")
                    state.status = "confirmed"
                    state.is_recording = False
                    break
 
        # write the complete, gap-free audio stream
        final_audio = snapshot()
        duration = final_audio.size / SAMPLE_RATE
        final_filename = recording_manager.create_temp_file("_final.wav")  
        wav.write(final_filename, SAMPLE_RATE, final_audio)
        state.final_file = final_filename
        # Only a keyword-confirmed recording advances the sequence the live
        # session loop waits on. A manual stop / timeout leaves status != of
        # "confirmed", so it writes the file but never triggers processing —
        # matching the original behaviour where only the stop phrase completed.
        if state.status == "confirmed":
            state.recording_seq += 1
        logger.info(f"Recording saved: {final_filename} ({duration:.1f}s captured)")
        print(f"[REC] Recording finished ({duration:.1f}s).")
 
    except Exception as e:
        logger.error(f"Recording error: {e}")
        print(f"[REC] Error: {e}")
        state.is_recording = False

_SERVER_STARTED = False
_SHARED_STATE = RecordingState()
_TRANSCRIBE_FN = None

def publish_result(data: dict) -> None:
    """Hand a finished handoff (or an ``{"error": ...}`` payload) to the web UI.

    Called by main.py once extraction is done, so the live-recording page can
    swap its "generating…" overlay for the hospital dashboard. Safe to call even
    when no browser is connected — it just sets in-memory state that the next
    ``/get_result`` poll would read. Cleared by ``RecordingState.reset()`` at the
    start of each new recording, so one report never leaks into the next.
    """
    _SHARED_STATE.result = data
    logger.info("Published handoff result to web UI (%s)", "error" if isinstance(data, dict) and data.get("error") else "ok")

def publish_transcript(text: str) -> None:
    """Hand the final full-audio transcription to the web UI so the loading
    screen can show it before extraction finishes. Cleared by reset() and
    /start_recording so one recording's transcript never leaks into the next."""
    _SHARED_STATE.final_transcript = text
    logger.info("Published final transcript to web UI (%d chars)", len(text or ""))

def publish_stage(text: str) -> None:
    """Update the loading-screen message as the pipeline moves through its
    stages (transcribe -> extract -> suggest). Harmless no-op for terminal
    cases where no browser is polling."""
    _SHARED_STATE.stage = text
    logger.info("Stage: %s", text)

def _launch_worker() -> None:
    """Start the recording worker thread for the current transcriber."""
    threading.Thread(
        target=lambda: _run_recording_worker(_SHARED_STATE, _TRANSCRIBE_FN),
        daemon=True,
    ).start()

def _ensure_server() -> None:
    """
    Start the Flask server exactly once and register its routes.

    The server is a module-level singleton: Flask binds port 5000 on the first
    call and that thread lives for the rest of the process. Re-creating the app
    on a later call would silently fail to bind the port, so we start it once
    and reuse it. All routes read/write the shared ``_SHARED_STATE`` object.
    """
    global _SERVER_STARTED
    if _SERVER_STARTED:
        logger.info("Reusing existing Flask web UI server")
        print("[WEB] Reusing local content server...")
        return

    state = _SHARED_STATE
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def index():
        return _render_index()

    @app.route("/health")
    def health() -> dict:
        return jsonify({"status": "ok"})

    @app.route("/start_recording")
    def start_recording() -> dict:
        if not state.is_recording:
            state.is_recording = True
            state.status = "recording"
            state.transcript = ""
            state.result = None
            state.final_file = None
            state.final_transcript = None
            state.stage = None
            state.start_time = time.time()
            state.audio_level = 0.0
            _launch_worker()
            logger.info("Recording started via web UI")
        return jsonify({"status": "started"})

    @app.route("/stop_recording")
    def stop_recording() -> dict:
        state.is_recording = False
        state.status = "ready"
        logger.info("Recording stopped via web UI")
        return jsonify({"status": "stopped"})

    @app.route("/get_state")
    def get_state() -> dict:
        return jsonify({
            "is_recording": state.is_recording,
            "transcript": state.transcript,
            "status": state.status,
            "audio_level": state.audio_level
        })

    @app.route("/get_result")
    def get_result() -> dict:
        # The browser polls this after confirmation. `result` stays None until
        # main.py finishes extraction and calls publish_result().
        return jsonify({
            "ready": state.result is not None,
            "result": state.result,
            "transcript": state.final_transcript,
            "stage": state.stage,
        })

    @app.route("/finish")
    def finish() -> dict:
        # The dashboard's "Finish" button ends the live session so the terminal
        # loop in run_live_session() returns to the main menu.
        state.session_active = False
        logger.info("Live session finished via web UI")
        return jsonify({"status": "finished"})

    logger.info("Starting Flask web UI server...")
    print("[WEB] Starting local content server...")
    threading.Thread(target=lambda: app.run(port=5000, use_reloader=False), daemon=True).start()
    _SERVER_STARTED = True
    time.sleep(1)


def run_live_session(transcribe_chunk_fn: Callable, process_fn: Callable) -> None:
    """
    Drive a full browser-based live session: the user can record a report, see
    the hospital dashboard, then record another — all without touching the
    terminal — until they click "Finish".

    For each recording the loop:
      1. waits for the browser to confirm a recording (stop phrase detected),
      2. calls ``process_fn(final_wav_path)`` — which transcribes, extracts, and
         publishes the dashboard result to the page,
      3. waits for the browser to either start another recording (which flips
         the status back to "recording") or finish the session.

    ``process_fn`` owns publishing (via main.process_case -> publish_result), so
    this loop only needs to catch hard failures and surface them to the page.
    Returns when the user finishes, closes with Ctrl-C, or the session ends.
    """
    global _TRANSCRIBE_FN
    _TRANSCRIBE_FN = transcribe_chunk_fn

    state = _SHARED_STATE
    state.reset()
    state.session_active = True

    _ensure_server()
    webbrowser.open("http://127.0.0.1:5000")
    logger.info("Opened browser to http://127.0.0.1:5000")
    print("[WEB] Waiting for recording from Web UI...")

    try:
        while state.session_active:
            # Wait for a completed recording (or the user finishing).
            while state.session_active and not (
                state.status == "confirmed" and state.final_file
            ):
                time.sleep(0.3)
            if not state.session_active:
                break

            final_file = state.final_file
            logger.info(f"Recording complete: {final_file}")
            recording_manager.save_final(final_file)

            # Process this recording. process_fn publishes the dashboard.
            try:
                process_fn(final_file)
            except Exception as e:
                logger.exception("Live processing failed")
                publish_result({"error": f"{type(e).__name__}: {e}"})

            # Hold here until the browser starts another recording or the session ends.
            while state.session_active and state.status == "confirmed":
                time.sleep(0.3)
    except KeyboardInterrupt:
        logger.info("Live session canceled by user")
    finally:
        state.session_active = False
        print("[WEB] Live session ended.")

def start_web_ui(transcribe_chunk_fn: Callable) -> Optional[str]:
    """
    Single-shot helper: start the server, wait for ONE confirmed recording, and
    return its audio path. Kept for callers that want a single capture;
    run_live_session() is the browser-driven multi-recording entry point.
    """
    global _TRANSCRIBE_FN
    state = _SHARED_STATE
    state.reset()
    _TRANSCRIBE_FN = transcribe_chunk_fn

    _ensure_server()
    webbrowser.open("http://127.0.0.1:5000")
    logger.info("Opened browser to http://127.0.0.1:5000")
    print("[WEB] Waiting for recording from Web UI...")

    try:
        while True:
            if state.status == "confirmed" and state.final_file:
                logger.info(f"Recording complete: {state.final_file}")
                recording_manager.save_final(state.final_file)
                return state.final_file
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Web UI canceled by user")
        return None
