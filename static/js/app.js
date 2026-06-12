/**
 * ReviewLens AI — Frontend
 *
 * All API calls go through the /api/* endpoints. Keeping everything in
 * vanilla JS with no build step so Render doesn't need Node to serve it.
 */

const API_BASE = "/api";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const textarea      = document.getElementById("reviewInput");
const charCount     = document.getElementById("charCount");
const analyzeBtn    = document.getElementById("analyzeBtn");
const clearBtn      = document.getElementById("clearBtn");
const loadingPanel  = document.getElementById("loadingPanel");
const errorPanel    = document.getElementById("errorPanel");
const errorMessage  = document.getElementById("errorMessage");
const resultsPanel  = document.getElementById("resultsPanel");
const statusDot     = document.getElementById("statusDot");
const statusLabel   = document.getElementById("statusLabel");
const sentimentBadge = document.getElementById("sentimentBadge");
const confidenceBars = document.getElementById("confidenceBars");
const summaryText   = document.getElementById("summaryText");
const summaryMeta   = document.getElementById("summaryMeta");
const examplesGrid  = document.getElementById("examplesGrid");


// ── Character count ───────────────────────────────────────────────────────────
textarea.addEventListener("input", () => {
  const len  = textarea.value.length;
  const max  = parseInt(textarea.getAttribute("maxlength"), 10);
  charCount.textContent = `${len} / ${max}`;
  charCount.classList.toggle("near-limit", len > max * 0.8);
  charCount.classList.toggle("at-limit",   len >= max);
});


// ── Example chips ─────────────────────────────────────────────────────────────
examplesGrid.addEventListener("click", (e) => {
  const chip = e.target.closest(".example-chip");
  if (!chip) return;
  const review = chip.dataset.review;
  textarea.value = review;
  // fire the input event so the char count updates
  textarea.dispatchEvent(new Event("input"));
  textarea.focus();
  // scroll the textarea into view on mobile
  textarea.scrollIntoView({ behavior: "smooth", block: "center" });
});


// ── Clear ─────────────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  textarea.value = "";
  charCount.textContent = "0 / 2000";
  charCount.classList.remove("near-limit", "at-limit");
  hideAll();
  textarea.focus();
});


// ── Analyze ───────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", runAnalysis);

textarea.addEventListener("keydown", (e) => {
  // Ctrl/Cmd + Enter submits — nice QoL touch
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    runAnalysis();
  }
});


async function runAnalysis() {
  const text = textarea.value.trim();
  if (!text) {
    showError("Please paste a review before analyzing.");
    return;
  }

  if (text.length < 10) {
    showError("Review is too short. Please enter at least 10 characters.");
    return;
  }

  hideAll();
  setLoading(true);

  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || `Server error (${response.status})`);
      return;
    }

    if (data.error) {
      showError(data.error);
      return;
    }

    renderResults(data.result);
  } catch (err) {
    // network errors (offline, CORS, etc.) end up here
    showError("Could not reach the API. Is the server running?");
    console.error("Fetch error:", err);
  } finally {
    setLoading(false);
  }
}


// ── Render results ────────────────────────────────────────────────────────────
function renderResults(result) {
  hideAll(); 
  // ── sentiment ──
  const sentiment = result.sentiment;
  if (sentiment && !sentiment.error) {
    const label = sentiment.label;   // "Positive" | "Neutral" | "Negative"
    const cls   = label.toLowerCase();

    sentimentBadge.textContent = label;
    sentimentBadge.className   = `sentiment-badge ${cls}`;

    // draw confidence bars from scratch each time
    confidenceBars.innerHTML = "";
    const scores = sentiment.scores || [];

    scores.forEach(({ label: scoreLabel, score }) => {
      const pct      = (score * 100).toFixed(1);
      const fillCls  = scoreLabel.toLowerCase();

      const row = document.createElement("div");
      row.className = "conf-row";
      row.innerHTML = `
        <div class="conf-row-header">
          <span class="conf-label">${scoreLabel}</span>
          <span class="conf-score">${pct}%</span>
        </div>
        <div class="conf-track">
          <div class="conf-fill ${fillCls}" style="width: 0%"></div>
        </div>
      `;
      confidenceBars.appendChild(row);

      // animate the fill bar after a tick so the CSS transition fires
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          row.querySelector(".conf-fill").style.width = `${pct}%`;
        });
      });
    });
  } else if (sentiment?.error) {
    sentimentBadge.textContent = "Unavailable";
    sentimentBadge.className   = "sentiment-badge";
    confidenceBars.innerHTML   = `<p style="color: var(--text-muted); font-size:12px;">${sentiment.error}</p>`;
  }

  // ── summary ──
  const summarization = result.summarization;
  if (summarization && !summarization.error && !summarization.skipped) {
    summaryText.textContent = summarization.summary || "—";
    summaryMeta.textContent = summarization.input_length
      ? `${summarization.input_length} words → ${summarization.summary_length} words`
      : "";
  } else if (summarization?.skipped) {
    summaryText.textContent  = "Review is too short to summarize meaningfully.";
    summaryText.style.color  = "var(--text-muted)";
    summaryMeta.textContent  = "";
  } else if (summarization?.error) {
    summaryText.textContent = `Model unavailable: ${summarization.error}`;
    summaryText.style.color = "var(--text-muted)";
    summaryMeta.textContent = "";
  }

  resultsPanel.hidden = false;
}


// ── UI helpers ────────────────────────────────────────────────────────────────
function setLoading(isLoading) {
  analyzeBtn.disabled    = isLoading;
  loadingPanel.hidden    = !isLoading;
}

function showError(msg) {
  errorMessage.textContent = msg;
  errorPanel.hidden        = false;
}

function hideAll() {
  loadingPanel.hidden  = true;
  errorPanel.hidden    = true;
  resultsPanel.hidden  = true;
  errorMessage.textContent = "";
  // reset summary text color in case it was muted by a previous skip/error
  summaryText.style.color = "";
}


// ── Health check on load ──────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/health`);
    const data = await res.json();

    const { sentiment_model, summarization_model } = data.models || {};
    const sentimentOk      = sentiment_model      === "loaded";
    const summarizationOk  = summarization_model  === "loaded";

    if (sentimentOk && summarizationOk) {
      statusDot.className  = "status-dot ready";
      statusLabel.textContent = "models ready";
    } else if (sentimentOk || summarizationOk) {
      statusDot.className  = "status-dot partial";
      const missing = [
        !sentimentOk     ? "sentiment"     : null,
        !summarizationOk ? "summarization" : null,
      ].filter(Boolean).join(", ");
      statusLabel.textContent = `${missing} model not loaded`;
    } else {
      statusDot.className  = "status-dot error";
      statusLabel.textContent = "models not loaded — run training first";
    }
  } catch {
    statusDot.className  = "status-dot error";
    statusLabel.textContent = "API unreachable";
  }
}

checkHealth();
hideAll();
