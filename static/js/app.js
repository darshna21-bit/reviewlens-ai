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

// New DOM Refs
const emptyPlaceholder = document.getElementById("emptyPlaceholder");
const analysisMetaRow  = document.getElementById("analysisMetaRow");
const analysisTimeValue = document.getElementById("analysisTimeValue");
const confidenceIndicatorValue = document.getElementById("confidenceIndicatorValue");
const confidenceIndicatorBadge = document.getElementById("confidenceIndicatorBadge");
const keywordsSection  = document.getElementById("keywordsSection");
const keywordsGrid     = document.getElementById("keywordsGrid");
const stepSentiment    = document.getElementById("stepSentiment");
const stepSummary      = document.getElementById("stepSummary");

// Sentiment Keywords Lists
const POS_KEYWORDS = ["excellent", "premium", "comfortable", "recommend", "great", "love", "satisfied", "satisfying", "good", "amazing", "beautiful", "fabulous", "impressive", "happy", "neat", "perfectly", "perfect", "hot", "cool", "best", "nice", "quiet", "silent", "satisfaction"];
const NEG_KEYWORDS = ["terrible", "damaged", "waste", "disappointing", "disappointed", "cracked", "crack", "unusable", "leaking", "leak", "stuck", "frustrating", "cheap", "plastic", "ugly", "laggy", "slow", "noise", "loud", "difficult", "missing", "faded", "fade", "annoying", "refund", "worse", "bad", "worst", "unhappy", "tinny", "blurry", "disaster"];
const NEU_KEYWORDS = ["average", "okay", "acceptable", "decent", "fine", "basic", "normal", "moderate", "expected", "serves", "standard"];


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
  if (emptyPlaceholder) emptyPlaceholder.hidden = false;
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
  if (emptyPlaceholder) emptyPlaceholder.hidden = true;
  setLoading(true);

  // Initialize checklist step animation
  stepSentiment.innerHTML = '<span class="step-icon">⏳</span> Processing Sentiment Analysis...';
  stepSentiment.className = 'loading-step';
  stepSummary.innerHTML = '<span class="step-icon">⏳</span> Generating Summary...';
  stepSummary.className = 'loading-step pending';

  const startTime = performance.now();

  // Fake step transition for realistic loading experience
  const stepTimeout = setTimeout(() => {
    stepSentiment.innerHTML = '<span class="step-icon">✓</span> Processing Sentiment Analysis';
    stepSummary.className = 'loading-step';
  }, 450);

  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    clearTimeout(stepTimeout);

    if (!response.ok) {
      showError(data.error || `Server error (${response.status})`);
      return;
    }

    if (data.error) {
      showError(data.error);
      return;
    }

    // Complete checklist steps
    stepSentiment.innerHTML = '<span class="step-icon">✓</span> Processing Sentiment Analysis';
    stepSummary.innerHTML = '<span class="step-icon">✓</span> Generating Summary';
    stepSummary.className = 'loading-step';

    const endTime = performance.now();
    const duration = ((endTime - startTime) / 1000).toFixed(2);
    analysisTimeValue.textContent = duration;

    renderResults(data.result, text);
  } catch (err) {
    clearTimeout(stepTimeout);
    showError("Could not reach the API. Is the server running?");
    console.error("Fetch error:", err);
  } finally {
    setLoading(false);
  }
}


// ── Render results ────────────────────────────────────────────────────────────
function renderResults(result, text) {
  hideAll();
  if (emptyPlaceholder) emptyPlaceholder.hidden = true;

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

    // ── confidence indicator card ──
    const maxScore = Math.max(...scores.map(s => s.score));
    const maxPct = (maxScore * 100).toFixed(1);
    confidenceIndicatorValue.textContent = `${maxPct}%`;
    
    let badgeText = "Low Confidence";
    let badgeClass = "low";
    if (maxScore >= 0.90) {
      badgeText = "Very High Confidence";
      badgeClass = "very-high";
    } else if (maxScore >= 0.75) {
      badgeText = "High Confidence";
      badgeClass = "high";
    } else if (maxScore >= 0.60) {
      badgeText = "Moderate Confidence";
      badgeClass = "moderate";
    }
    
    confidenceIndicatorBadge.textContent = badgeText;
    confidenceIndicatorBadge.className = `indicator-badge ${badgeClass}`;

  } else if (sentiment?.error) {
    sentimentBadge.textContent = "Unavailable";
    sentimentBadge.className   = "sentiment-badge";
    confidenceBars.innerHTML   = `<p style="color: var(--text-muted); font-size:12px;">${sentiment.error}</p>`;
    confidenceIndicatorValue.textContent = "0.0%";
    confidenceIndicatorBadge.textContent = "N/A";
    confidenceIndicatorBadge.className = "indicator-badge low";
  }

  // ── highlight keywords ──
  const reviewWords = text.toLowerCase().match(/\b\w+\b/g) || [];
  const uniquePos = new Set();
  const uniqueNeg = new Set();
  const uniqueNeu = new Set();

  reviewWords.forEach(word => {
    if (POS_KEYWORDS.includes(word)) uniquePos.add(word);
    else if (NEG_KEYWORDS.includes(word)) uniqueNeg.add(word);
    else if (NEU_KEYWORDS.includes(word)) uniqueNeu.add(word);
  });

  keywordsGrid.innerHTML = "";
  if (uniquePos.size > 0 || uniqueNeg.size > 0 || uniqueNeu.size > 0) {
    uniquePos.forEach(word => {
      const badge = document.createElement("span");
      badge.className = "keyword-badge positive";
      badge.textContent = word;
      keywordsGrid.appendChild(badge);
    });
    uniqueNeg.forEach(word => {
      const badge = document.createElement("span");
      badge.className = "keyword-badge negative";
      badge.textContent = word;
      keywordsGrid.appendChild(badge);
    });
    uniqueNeu.forEach(word => {
      const badge = document.createElement("span");
      badge.className = "keyword-badge neutral";
      badge.textContent = word;
      keywordsGrid.appendChild(badge);
    });
    keywordsSection.hidden = false;
  } else {
    keywordsSection.hidden = true;
  }

  // ── summary ──
  const summarization = result.summarization;
  if (summarization && !summarization.error && !summarization.skipped) {
    summaryText.textContent = summarization.summary || "Summary could not be generated.";
    summaryText.style.color = "";
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
  } else {
    summaryText.textContent = "Summary could not be generated.";
    summaryText.style.color = "var(--text-muted)";
    summaryMeta.textContent = "";
  }

  resultsPanel.hidden = false;
  analysisMetaRow.hidden = false;
}


// ── UI helpers ────────────────────────────────────────────────────────────────
function setLoading(isLoading) {
  analyzeBtn.disabled    = isLoading;
  loadingPanel.hidden    = !isLoading;
}

function showError(msg) {
  errorMessage.textContent = msg;
  errorPanel.hidden        = false;
  if (emptyPlaceholder) emptyPlaceholder.hidden = true;
}

// Reset UI
function hideAll() {
  loadingPanel.hidden  = true;
  errorPanel.hidden    = true;
  resultsPanel.hidden  = true;
  if (analysisMetaRow) analysisMetaRow.hidden = true;
  if (keywordsSection) keywordsSection.hidden = true;
  if (emptyPlaceholder) emptyPlaceholder.hidden = true;
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
if (emptyPlaceholder) emptyPlaceholder.hidden = false;

