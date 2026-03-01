/* ============================================================
   MINI CODE MENTOR — app.js  (Multi-Language)
   API:    POST /run
   Fields: code, language, mode, user_query, user_id, intent
   ============================================================ */

"use strict";

const API_URL = (() => {
  if (
    window.location.protocol === "file:" ||
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
  ) {
    return "http://127.0.0.1:8000/run";
  }
  return `${window.location.origin}/run`;
})();
const CODE_MAX = 5000;

// ── App state ──────────────────────────────────────────────
const appState = {
  intent: "analyze",
  mode: "beginner",
  user_id: "default_user",
  language: "c",
  loading: false,
};

// ── Intent config ──────────────────────────────────────────
const INTENT_CONFIG = {
  analyze: {
    label: "Analyze",
    defaultQuery: "Analyze this code",
    loadingMsg: "Analyzing code…",
  },
  explain: {
    label: "Explain",
    defaultQuery: "Explain this code",
    loadingMsg: "Generating explanation…",
  },
  fix: {
    label: "Fix",
    defaultQuery: "Fix and debug this code",
    loadingMsg: "Fixing bugs…",
  },
  practice: {
    label: "Practice",
    defaultQuery: "Generate practice problems from this code",
    loadingMsg: "Building practice set…",
  },
  score: {
    label: "Score",
    defaultQuery: "Score and evaluate this code",
    loadingMsg: "Scoring code quality…",
  },
  full_review: {
    label: "Full Review",
    defaultQuery: "Full review of this code",
    loadingMsg: "Running full review…",
  },
  mistake_fixer: {
    label: "Correction Engine",
    defaultQuery: "Show my common mistakes",
    loadingMsg: "Analyzing your common mistakes…",
  },
};

const LANG_LABEL = {
  c: "C",
  cpp: "C++",
  python: "Python",
  java: "Java",
  csharp: "C#",
  javascript: "JavaScript",
};

// ── DOM refs ───────────────────────────────────────────────
const sidebar = document.getElementById("sidebar");
const menuBtn = document.getElementById("menuBtn");
const navItems = document.querySelectorAll(".nav-item");
const modeButtons = document.querySelectorAll(".mode-btn");
const userIdInput = document.getElementById("userIdInput");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

const breadcrumbIntent = document.getElementById("breadcrumbIntent");
const breadcrumbLevel = document.getElementById("breadcrumbLevel");
const breadcrumbUid = document.getElementById("breadcrumbUid");
const charCount = document.getElementById("charCount");

const codeEditor = document.getElementById("codeEditor");
const languageSelect = document.getElementById("langDropdown");
const editorLabel = document.getElementById("editorLabel");
const lineNumbers = document.getElementById("lineNumbers");
const clearBtn = document.getElementById("clearBtn");
const copyCodeBtn = document.getElementById("copyCodeBtn");

const userQueryInput = document.getElementById("userQueryInput");
const runBtn = document.getElementById("runBtn");
const runBtnLabel = document.getElementById("runBtnLabel");
const copyOutputBtn = document.getElementById("copyOutputBtn");

const editorPanel = document.getElementById("editorPanelEl");
const mistakeFixerPanel = document.getElementById("mistakeFixerPanel");
const mistakeFixerRunBtn = document.getElementById("mistakeFixerRunBtn");
const welcomeState = document.getElementById("welcomeState");
const loadingState = document.getElementById("loadingState");
const loadingLabel = document.getElementById("loadingLabel");
const loadingIntent = document.getElementById("loadingIntent");
const resultRoot = document.getElementById("resultRoot");

const toast = document.getElementById("toast");
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
const navIndicator = document.getElementById("navIndicator");

// ── Theme toggle ───────────────────────────────────────────
const MOON_SVG =
  '<path d="M12.5 3.5A5 5 0 008 13a5 5 0 110-10 5 5 0 014.5.5z"/>';
const SUN_SVG =
  '<circle cx="8" cy="8" r="3.5"/><path d="M8 1v1.5M8 13.5V15M1 8h1.5M13.5 8H15M3.2 3.2l1 1M11.8 11.8l1 1M11.8 3.2l-1 1M4.2 11.8l-1 1" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" fill="none"/>';

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("mcm-theme", theme);
  themeIcon.innerHTML = theme === "light" ? SUN_SVG : MOON_SVG;
}

// Init theme from storage
applyTheme(localStorage.getItem("mcm-theme") || "dark");

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "light" ? "dark" : "light");
});

// ══════════════════════════════════════════════════════════
// SIDEBAR — mobile toggle
// ══════════════════════════════════════════════════════════
menuBtn.addEventListener("click", () => sidebar.classList.toggle("open"));
document.addEventListener("click", (e) => {
  if (
    sidebar.classList.contains("open") &&
    !sidebar.contains(e.target) &&
    e.target !== menuBtn
  ) {
    sidebar.classList.remove("open");
  }
});

// ══════════════════════════════════════════════════════════
// INTENT NAV
// ══════════════════════════════════════════════════════════
navItems.forEach((item) => {
  item.addEventListener("click", () => {
    selectIntent(item.dataset.intent, item.dataset.query);
    sidebar.classList.remove("open");
  });
});

function selectIntent(intent, query) {
  appState.intent = intent;
  const cfg = INTENT_CONFIG[intent];

  navItems.forEach((n) =>
    n.classList.toggle("active", n.dataset.intent === intent),
  );

  userQueryInput.value = query || cfg.defaultQuery;
  breadcrumbIntent.textContent = cfg.label;
  runBtnLabel.textContent = cfg.label;

  resetOutput();
  setStatus("ready", "Ready");
}

// ══════════════════════════════════════════════════════════
// LANGUAGE SELECTOR (custom dropdown)
// ══════════════════════════════════════════════════════════
const langDropdown = document.getElementById("langDropdown");
const langDropdownSelected = document.getElementById("langDropdownSelected");
const langDropdownMenu = document.getElementById("langDropdownMenu");
const langDropdownLogo = document.getElementById("langDropdownLogo");
const langDropdownLabel = document.getElementById("langDropdownLabel");
const langOptions = document.querySelectorAll(".lang-option");

function setLanguage(lang, label, logoSVG) {
  appState.language = lang;
  editorLabel.textContent = `${LANG_LABEL[lang]} — Input`;
  langDropdownLabel.textContent = label;
  langDropdownLogo.innerHTML = logoSVG;
  langOptions.forEach((o) =>
    o.classList.toggle("active", o.dataset.lang === lang),
  );
}

// Init with first option (C)
const firstOption = langDropdownMenu.querySelector(".lang-option");
setLanguage("c", "C", firstOption.querySelector("svg").outerHTML);

langDropdownSelected.addEventListener("click", (e) => {
  e.stopPropagation();
  langDropdown.classList.toggle("open");
});

langOptions.forEach((opt) => {
  opt.addEventListener("click", () => {
    const lang = opt.dataset.lang;
    const label = opt.querySelector("span").textContent;
    const svg = opt.querySelector("svg").outerHTML;
    setLanguage(lang, label, svg);
    langDropdown.classList.remove("open");
  });
});

document.addEventListener("click", () => langDropdown.classList.remove("open"));

// ══════════════════════════════════════════════════════════
// MODE TOGGLE
// ══════════════════════════════════════════════════════════
modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const mode = btn.dataset.mode;
    appState.mode = mode;
    modeButtons.forEach((b) =>
      b.classList.toggle("active", b.dataset.mode === mode),
    );
    breadcrumbLevel.textContent = mode;
  });
});

// ══════════════════════════════════════════════════════════
// USER_ID
// ══════════════════════════════════════════════════════════
userIdInput.addEventListener("input", () => {
  const val = userIdInput.value.trim() || "default_user";
  appState.user_id = val;
  breadcrumbUid.textContent = val;
});

// ══════════════════════════════════════════════════════════
// CODE EDITOR — line numbers + char count
// ══════════════════════════════════════════════════════════
function updateLineNumbers() {
  const lines = codeEditor.value.split("\n").length;
  lineNumbers.textContent = Array.from({ length: lines }, (_, i) => i + 1).join(
    "\n",
  );
}

function updateCharCount() {
  const len = codeEditor.value.length;
  charCount.textContent = `${len} / ${CODE_MAX}`;
  charCount.className = "char-count";
  if (len > CODE_MAX) charCount.classList.add("over");
  else if (len > CODE_MAX * 0.8) charCount.classList.add("warn");
}

function syncScrollGutter() {
  lineNumbers.scrollTop = codeEditor.scrollTop;
}

codeEditor.addEventListener("input", () => {
  updateLineNumbers();
  updateCharCount();
});
codeEditor.addEventListener("scroll", syncScrollGutter);

codeEditor.addEventListener("keydown", (e) => {
  if (e.key === "Tab") {
    e.preventDefault();
    const s = codeEditor.selectionStart;
    const end = codeEditor.selectionEnd;
    codeEditor.value =
      codeEditor.value.substring(0, s) +
      "    " +
      codeEditor.value.substring(end);
    codeEditor.selectionStart = codeEditor.selectionEnd = s + 4;
    updateLineNumbers();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runBtn.click();
});

clearBtn.addEventListener("click", () => {
  codeEditor.value = "";
  updateLineNumbers();
  updateCharCount();
  resetOutput();
});

copyCodeBtn.addEventListener("click", () => {
  if (!codeEditor.value) return showToast("Nothing to copy.", "info");
  navigator.clipboard
    .writeText(codeEditor.value)
    .then(() => showToast("Code copied!", "success"));
});

copyOutputBtn.addEventListener("click", () => {
  const text = resultRoot.innerText || "";
  if (!text) return showToast("No output to copy.", "info");
  navigator.clipboard
    .writeText(text)
    .then(() => showToast("Output copied!", "success"));
});

// ══════════════════════════════════════════════════════════
// RUN
// ══════════════════════════════════════════════════════════
runBtn.addEventListener("click", triggerRun);
mistakeFixerRunBtn.addEventListener("click", triggerRun);

function triggerRun() {
  if (appState.loading) return;

  const code = codeEditor.value;
  const mode = appState.mode;
  const user_query =
    userQueryInput.value.trim() || INTENT_CONFIG[appState.intent].defaultQuery;
  const user_id = appState.user_id || "default_user";

  if (!code || code.trim().length === 0) {
    showToast("Please paste some code first.", "error");
    return;
  }
  if (code.length > CODE_MAX) {
    showToast(`Code too long — max ${CODE_MAX} characters.`, "error");
    return;
  }

  const requestBody = {
    code,
    language: appState.language,
    mode,
    user_query,
    user_id,
    intent: appState.intent,
  };

  postToBackend(requestBody);
}

async function postToBackend(requestBody) {
  appState.loading = true;
  runBtn.disabled = true;

  const cfg = INTENT_CONFIG[appState.intent];
  setStatus("loading", "Working…");
  showLoadingState(
    cfg.loadingMsg,
    `intent → ${appState.intent} · lang → ${appState.language} · mode → ${appState.mode}`,
  );

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 150_000);

    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const j = await response.json();
        detail = j.detail || j.error || detail;
      } catch {}
      throw new Error(detail);
    }

    const data = await response.json();
    renderResponse(data);
    setStatus("ready", "Done");
    showToast("Complete!", "success");
  } catch (err) {
    const msg =
      err.name === "AbortError"
        ? "Request timed out. Try a single intent instead of Full Review."
        : err.message;
    renderError(msg);
    setStatus("error", "Error");
    showToast(msg, "error");
  } finally {
    appState.loading = false;
    runBtn.disabled = false;
  }
}

// ══════════════════════════════════════════════════════════
// TRUNCATION HELPER
// ══════════════════════════════════════════════════════════
function checkAndAppendTruncationWarning(container, text) {
  const MARKER = "[Output truncated due to token limit.]";
  if (typeof text === "string" && text.includes(MARKER)) {
    container.textContent = text.replace(MARKER, "").trim();
    const w = el("div", "truncation-warning");
    w.textContent = "⚠ Response shortened due to token limit.";
    container.appendChild(w);
  } else {
    container.textContent = text;
  }
}
const renderWithTruncation = checkAndAppendTruncationWarning;

// ══════════════════════════════════════════════════════════
// RENDER RESPONSE
// ══════════════════════════════════════════════════════════
function renderResponse(data) {
  hideAllOutputStates();
  resultRoot.style.display = "";
  resultRoot.innerHTML = "";

  if (data.error) {
    resultRoot.appendChild(buildErrorCard(data.error));
    return;
  }

  let hasContent = false;

  function safeAppend(builder, payload, label) {
    try {
      resultRoot.appendChild(builder(payload));
      hasContent = true;
    } catch (err) {
      resultRoot.appendChild(
        buildErrorCard(`${label} failed to render: ${err.message}`),
      );
      hasContent = true;
    }
  }

  if (data.analysis !== undefined)
    safeAppend(buildAnalysisCard, data.analysis, "Analysis");
  if (data.explanation !== undefined)
    safeAppend(buildExplanationCard, data.explanation, "Explanation");
  if (data.fix !== undefined) safeAppend(buildFixCard, data.fix, "Fix");
  if (data.score !== undefined && appState.intent !== "practice")
    safeAppend(buildScoreCard, data.score, "Score");
  if (data.practice !== undefined)
    safeAppend(buildPracticeCard, data.practice, "Practice");
  if (data.mistake_fixer !== undefined)
    safeAppend(buildMistakeFixerCard, data.mistake_fixer, "Correction Engine");

  if (!hasContent) {
    resultRoot.appendChild(
      buildErrorCard("Unexpected response — no known fields found."),
    );
  }

  // Add Download Report button for Full Review intent
  if (appState.intent === "full_review" && hasContent) {
    const dlWrap = el("div", "download-report-wrap");
    const dlBtn = el("button", "download-report-btn");
    dlBtn.innerHTML = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" width="15" height="15"><path d="M8 2v8M5 7l3 3 3-3M2 12h12"/></svg> Download Report (PDF)`;
    dlBtn.title = "Download the full review as a PDF";
    dlBtn.addEventListener("click", () => downloadReviewAsPDF(dlBtn));
    dlWrap.appendChild(dlBtn);
    resultRoot.appendChild(dlWrap);
  }
}

function downloadReviewAsPDF(btn) {
  const ICON_DOWNLOAD = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" width="15" height="15"><path d="M8 2v8M5 7l3 3 3-3M2 12h12"/></svg>`;
  const ICON_SPIN = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" width="15" height="15" style="animation:spin 1s linear infinite"><path d="M8 2a6 6 0 100 12A6 6 0 008 2z" stroke-dasharray="28" stroke-dashoffset="10"/></svg>`;

  btn.disabled = true;
  btn.innerHTML = `${ICON_SPIN} Generating…`;

  // Hide the download button row so it's not captured in the PDF
  const dlWrap = btn.closest(".download-report-wrap");
  if (dlWrap) dlWrap.style.visibility = "hidden";

  // Build a wrapper with a header that will appear at the top of the PDF
  const wrapper = document.createElement("div");
  wrapper.style.cssText = "font-family:inherit;padding:4px 0;";

  const header = document.createElement("div");
  header.style.cssText =
    "text-align:center;margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid rgba(255,255,255,0.12);";
  header.innerHTML = `
    <div style="font-size:18px;font-weight:700;letter-spacing:-.3px;color:#818cf8;">PrismAI — Full Code Review</div>
    <div style="font-size:11px;opacity:.55;margin-top:4px;">Generated ${new Date().toLocaleString()}</div>
  `;

  // Clone all result cards (everything except the download button row)
  const contentClone = resultRoot.cloneNode(true);
  const dlWrapInClone = contentClone.querySelector(".download-report-wrap");
  if (dlWrapInClone) dlWrapInClone.remove();

  wrapper.appendChild(header);
  wrapper.appendChild(contentClone);

  // Mount wrapper temporarily in the actual document so it fully inherits
  // all CSS variables and computed styles from the live page
  wrapper.style.cssText +=
    "position:fixed;top:0;left:0;width:780px;z-index:-1;pointer-events:none;opacity:0;";
  document.body.appendChild(wrapper);

  // Small tick to let browser compute styles before html2canvas reads them
  requestAnimationFrame(() => {
    wrapper.style.opacity = "1";

    const opt = {
      margin: [10, 12, 10, 12],
      filename: `prismai-full-review-${Date.now()}.pdf`,
      image: { type: "jpeg", quality: 0.98 },
      html2canvas: {
        scale: 2,
        useCORS: true,
        logging: false,
        backgroundColor: null, // preserve the element's own background
        windowWidth: 800,
      },
      jsPDF: { unit: "mm", format: "a4", orientation: "portrait" },
      pagebreak: { mode: ["avoid-all", "css"] },
    };

    html2pdf()
      .set(opt)
      .from(wrapper)
      .save()
      .then(() => {
        document.body.removeChild(wrapper);
        if (dlWrap) dlWrap.style.visibility = "";
        btn.disabled = false;
        btn.innerHTML = `${ICON_DOWNLOAD} Download Report (PDF)`;
        showToast("PDF downloaded!", "success");
      })
      .catch((err) => {
        document.body.removeChild(wrapper);
        if (dlWrap) dlWrap.style.visibility = "";
        btn.disabled = false;
        btn.innerHTML = `${ICON_DOWNLOAD} Download Report (PDF)`;
        showToast("PDF failed: " + err.message, "error");
      });
  });
}

function renderError(msg) {
  hideAllOutputStates();
  resultRoot.style.display = "";
  resultRoot.innerHTML = "";
  resultRoot.appendChild(buildErrorCard(msg));
}

// ══════════════════════════════════════════════════════════
// CARD BUILDERS
// ══════════════════════════════════════════════════════════
function buildAnalysisCard(analysis) {
  const card = makeCard("Analysis", "card-analyze");

  if (typeof analysis === "string") {
    try {
      analysis = JSON.parse(analysis);
    } catch (_) {
      const pre = el("div", "result-text");
      renderWithTruncation(pre, analysis);
      card.appendChild(pre);
      return card;
    }
  }

  if (!analysis || typeof analysis !== "object") {
    const pre = el("div", "result-text");
    pre.textContent = "No analysis data returned.";
    card.appendChild(pre);
    return card;
  }

  if (analysis.syntax_errors !== undefined) {
    card.appendChild(fieldLabel("Syntax Errors"));
    card.appendChild(buildList(analysis.syntax_errors));
  }
  if (analysis.logical_errors !== undefined) {
    card.appendChild(fieldLabel("Logical Errors"));
    card.appendChild(buildList(analysis.logical_errors));
  }
  if (analysis.inefficiencies !== undefined) {
    card.appendChild(fieldLabel("Inefficiencies"));
    card.appendChild(buildList(analysis.inefficiencies));
  }
  if (analysis.summary !== undefined) {
    card.appendChild(fieldLabel("Summary"));
    const sm = el("div", "result-text");
    sm.textContent = analysis.summary;
    card.appendChild(sm);
  }

  return card;
}

function buildExplanationCard(explanation) {
  const card = makeCard("Explanation", "card-explain");

  if (typeof explanation === "string" && explanation.trim()) {
    const lines = explanation.split("\n").filter((l) => l.trim());
    if (lines.length > 0) {
      const list = el("div", "result-list");
      lines.forEach((line) => {
        const row = el("div", "tip-row");
        const icon = el("span", "tip-icon");
        icon.innerHTML = `<svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm-.75 9.75l-2-2 1.06-1.06 .94.94 2.44-2.44 1.06 1.06-3.5 3.5z"/></svg>`;
        const text = el("span", "");
        text.textContent = line.replace(/^-\s*/, "");
        row.appendChild(icon);
        row.appendChild(text);
        list.appendChild(row);
      });
      card.appendChild(list);
    } else {
      const pre = el("div", "result-text");
      pre.textContent = explanation;
      card.appendChild(pre);
    }
  } else if (typeof explanation === "object" && explanation !== null) {
    const pre = el("div", "result-text");
    pre.textContent = JSON.stringify(explanation, null, 2);
    card.appendChild(pre);
  } else {
    const pre = el("div", "result-text");
    pre.textContent = "No explanation returned.";
    card.appendChild(pre);
  }

  return card;
}

function buildFixCard(fix) {
  const card = makeCard("Fix & Debug", "card-fix");

  if (fix.error) {
    card.appendChild(issueRow(fix.error));
    if (fix.raw_response) {
      card.appendChild(fieldLabel("Raw response"));
      card.appendChild(codeBlock(fix.raw_response));
    }
    return card;
  }

  if (fix.issue) {
    card.appendChild(fieldLabel("Issue"));
    const t = el("div", "result-text");
    t.textContent = fix.issue;
    card.appendChild(t);
  }
  if (fix.corrected_code) {
    card.appendChild(fieldLabel("Corrected Code"));
    card.appendChild(codeBlock(fix.corrected_code));
  }
  if (fix.explanation) {
    card.appendChild(fieldLabel("Explanation"));
    const e = el("div", "result-text");
    checkAndAppendTruncationWarning(e, fix.explanation);
    card.appendChild(e);
  }

  return card;
}

function buildScoreCard(score) {
  const card = makeCard("Code Score", "card-score");

  if (score.error) {
    card.appendChild(issueRow(score.error));
    if (score.raw_response) card.appendChild(codeBlock(score.raw_response));
    return card;
  }

  const overall =
    typeof score.overall_score === "number" ? score.overall_score : 0;

  const overallWrap = el("div", "overall-wrap");
  const ringContainer = el("div", "ring-container");

  const svg = svgEl("svg", {
    class: "score-ring",
    viewBox: "0 0 80 80",
    width: "80",
    height: "80",
  });
  const bgCircle = svgEl("circle", {
    class: "ring-bg",
    cx: "40",
    cy: "40",
    r: "30",
  });
  const fillCircle = svgEl("circle", {
    class: "ring-fill",
    cx: "40",
    cy: "40",
    r: "30",
  });

  const circ = 2 * Math.PI * 30;
  fillCircle.style.strokeDasharray = circ;
  fillCircle.style.strokeDashoffset = circ;
  fillCircle.style.stroke = scoreColor(overall);

  svg.appendChild(bgCircle);
  svg.appendChild(fillCircle);

  const ringLabel = el("div", "ring-label");
  const ringScore = el("div", "ring-score");
  ringScore.textContent = "–";
  const ringDenom = el("div", "ring-denom");
  ringDenom.textContent = "/ 10";
  ringLabel.appendChild(ringScore);
  ringLabel.appendChild(ringDenom);

  ringContainer.appendChild(svg);
  ringContainer.appendChild(ringLabel);

  const overallMeta = el("div", "overall-meta");
  const overallTitle = el("div", "overall-title");
  overallTitle.textContent = "Overall Score";
  const overallVerdict = el("div", "overall-verdict");
  overallVerdict.textContent = scoreVerdict(overall);
  overallVerdict.style.color = scoreColor(overall);
  const overallDesc = el("div", "overall-desc");
  overallDesc.textContent = scoreDesc(overall);

  overallMeta.appendChild(overallTitle);
  overallMeta.appendChild(overallVerdict);
  overallMeta.appendChild(overallDesc);
  overallWrap.appendChild(ringContainer);
  overallWrap.appendChild(overallMeta);
  card.appendChild(overallWrap);

  const breakdown = {
    syntax_score: score.syntax_score,
    logic_score: score.logic_score,
    clarity_score: score.clarity_score,
    robustness_score: score.robustness_score,
  };

  const grid = el("div", "score-grid");
  Object.entries(breakdown).forEach(([key, val]) => {
    if (val === undefined || val === null) return;
    const metric = el("div", "score-metric");
    const labelEl = el("div", "score-metric-label");
    labelEl.textContent = key.replace("_score", "").replace("_", " ");
    const barWrap = el("div", "score-metric-bar-wrap");
    const bar = el("div", "score-metric-bar");
    bar.style.background = scoreColor(val);
    barWrap.appendChild(bar);
    const valueEl = el("div", "score-metric-value");
    const denom = el("span", "score-metric-denom");
    denom.textContent = " / 10";
    valueEl.textContent = "–";
    valueEl.appendChild(denom);

    metric.appendChild(labelEl);
    metric.appendChild(barWrap);
    metric.appendChild(valueEl);
    grid.appendChild(metric);

    requestAnimationFrame(() => {
      setTimeout(() => {
        bar.style.width = `${(val / 10) * 100}%`;
        valueEl.firstChild.textContent = formatScore(val);
      }, 80);
    });
  });
  card.appendChild(grid);

  requestAnimationFrame(() => {
    setTimeout(() => {
      fillCircle.style.strokeDashoffset = circ - (overall / 10) * circ;
      ringScore.textContent = formatScore(overall);
    }, 80);
  });

  return card;
}

function buildPracticeCard(practice) {
  const card = makeCard("Practice Problems", "card-practice");

  if (practice.skipped) {
    const info = el("div", "result-text");
    info.textContent = "Practice problems not generated for this request.";
    card.appendChild(info);
    return card;
  }

  if (practice.error) {
    card.appendChild(issueRow(practice.error));
    if (practice.raw_response)
      card.appendChild(codeBlock(practice.raw_response));
    return card;
  }

  if (
    Array.isArray(practice.similar_problems) &&
    practice.similar_problems.length
  ) {
    card.appendChild(fieldLabel("Similar Problems"));
    const list = el("div", "problem-list");
    practice.similar_problems.forEach((prob, idx) => {
      const item = el("div", "problem-item");
      const num = el("span", "problem-num");
      num.textContent = idx + 1 + ".";
      const txt = el("span", "");
      txt.textContent = prob;
      item.appendChild(num);
      item.appendChild(txt);
      list.appendChild(item);
    });
    card.appendChild(list);
  }

  if (practice.challenge_problem) {
    const wrap = el("div", "");
    wrap.style.marginTop = "14px";
    const cl = el("div", "challenge-label");
    cl.textContent = "⚡ Challenge Problem";
    const cb = el("div", "challenge-block");
    cb.textContent = practice.challenge_problem;
    wrap.appendChild(cl);
    wrap.appendChild(cb);
    card.appendChild(wrap);
  }

  if (Array.isArray(practice.hints) && practice.hints.length) {
    card.appendChild(fieldLabel("Hints"));
    practice.hints.forEach((h) => card.appendChild(tipRow(h)));
  }

  return card;
}

function buildMistakeFixerCard(data) {
  const card = makeCard("Correction Engine", "card-mistake");

  if (data.no_data) {
    const info = el("div", "result-text");
    info.textContent =
      "No mistake history found yet. Submit some code first so we can learn your patterns!";
    card.appendChild(info);
    return card;
  }

  if (data.error) {
    card.appendChild(issueRow(data.error));
    return card;
  }

  if (Array.isArray(data.mistakes_targeted) && data.mistakes_targeted.length) {
    card.appendChild(fieldLabel("Your Recurring Mistakes"));
    const mistakeList = el("div", "result-list");
    data.mistakes_targeted.forEach((m) => {
      const row = el("div", "tip-row");
      const icon = el("span", "tip-icon");
      icon.innerHTML = `<svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zM7 5h2v4H7zm0 5h2v2H7z"/></svg>`;
      const text = el("span", "");
      text.textContent = m;
      row.appendChild(icon);
      row.appendChild(text);
      mistakeList.appendChild(row);
    });
    card.appendChild(mistakeList);
  }

  if (Array.isArray(data.questions) && data.questions.length) {
    card.appendChild(fieldLabel("Practice Questions Targeting Your Mistakes"));
    data.questions.forEach((q, i) => {
      const wrap = el("div", "problem-item");
      wrap.style.marginBottom = "12px";

      const num = el("div", "challenge-label");
      num.textContent = `Q${i + 1}`;
      num.style.marginBottom = "4px";

      const qText = el("div", "");
      qText.textContent = q.question;
      qText.style.marginBottom = "4px";

      if (q.targets) {
        const tag = el("div", "score-metric-label");
        tag.textContent = "Targets: " + q.targets;
        tag.style.opacity = "0.6";
        tag.style.fontSize = "11px";
        wrap.appendChild(num);
        wrap.appendChild(qText);
        wrap.appendChild(tag);
      } else {
        wrap.appendChild(num);
        wrap.appendChild(qText);
      }

      card.appendChild(wrap);
    });
  } else {
    const info = el("div", "result-text");
    info.textContent = "No questions generated. Try again.";
    card.appendChild(info);
  }

  return card;
}

function buildErrorCard(message) {
  const card = makeCard("Error", "card-error");
  const txt = el("div", "error-text");
  txt.textContent = message;
  card.appendChild(txt);
  return card;
}

// ══════════════════════════════════════════════════════════
// DOM HELPERS
// ══════════════════════════════════════════════════════════
function el(tag, className) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  return e;
}
function svgEl(tag, attrs) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
  return e;
}
function makeCard(title, extraClass = "") {
  const card = el("div", `result-card ${extraClass}`);
  const titleEl = el("div", "result-card-title");
  titleEl.textContent = title;
  card.appendChild(titleEl);
  return card;
}
function fieldLabel(text) {
  const lbl = el("div", "field-label");
  lbl.textContent = text;
  return lbl;
}
function buildList(items) {
  const wrap = el("div", "result-list");
  if (!items || items.length === 0) {
    const none = el("div", "result-text");
    none.textContent = "None";
    wrap.appendChild(none);
    return wrap;
  }
  items.forEach((item) => {
    const row = el("div", "issue-row");
    const text = el("span", "");
    text.textContent = item;
    row.appendChild(text);
    wrap.appendChild(row);
  });
  return wrap;
}
function codeBlock(content) {
  const text =
    typeof content === "string"
      ? content.replace(/\\\\n/g, "\n")
      : JSON.stringify(content, null, 2);

  const wrap = el("div", "code-block-wrap");

  const copyBtn = el("button", "code-copy-btn");
  copyBtn.title = "Copy code";
  copyBtn.innerHTML =
    '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="5" y="5" width="8" height="9" rx="1"/><path d="M3 11V3a1 1 0 011-1h7"/></svg>';
  copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(text).then(() => {
      copyBtn.innerHTML =
        '<svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor"><path d="M6.5 11.5l-3-3 1.06-1.06 1.94 1.94 4.44-4.44 1.06 1.06z"/></svg>';
      setTimeout(() => {
        copyBtn.innerHTML =
          '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="5" y="5" width="8" height="9" rx="1"/><path d="M3 11V3a1 1 0 011-1h7"/></svg>';
      }, 1500);
    });
  });

  const block = document.createElement("pre");
  block.className = "result-code-block";
  block.textContent = text;

  wrap.appendChild(copyBtn);
  wrap.appendChild(block);
  return wrap;
}
function issueRow(msg) {
  const row = el("div", "issue-row");
  const icon = el("span", "issue-icon");
  icon.innerHTML = `<svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm-.75 3.75h1.5v4.5h-1.5v-4.5zm0 6h1.5v1.5h-1.5v-1.5z"/></svg>`;
  const text = el("span", "");
  text.textContent = msg;
  row.appendChild(icon);
  row.appendChild(text);
  return row;
}
function tipRow(msg) {
  const row = el("div", "tip-row");
  const icon = el("span", "tip-icon");
  icon.innerHTML = `<svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm-.75 9.75l-2-2 1.06-1.06 .94.94 2.44-2.44 1.06 1.06-3.5 3.5z"/></svg>`;
  const text = el("span", "");
  text.textContent = msg;
  row.appendChild(icon);
  row.appendChild(text);
  return row;
}

// ══════════════════════════════════════════════════════════
// SCORE HELPERS
// ══════════════════════════════════════════════════════════
function formatScore(val) {
  if (typeof val !== "number") return "–";
  return Number.isInteger(val) ? String(val) : val.toFixed(1);
}
function scoreColor(val) {
  if (val >= 8) return "var(--green)";
  if (val >= 5) return "var(--amber)";
  return "var(--red)";
}
function scoreVerdict(val) {
  if (val >= 9) return "Excellent";
  if (val >= 7) return "Good";
  if (val >= 5) return "Average";
  if (val >= 3) return "Needs Work";
  return "Poor";
}
function scoreDesc(val) {
  if (val >= 9) return "Outstanding code quality across all metrics.";
  if (val >= 7) return "Solid code with minor areas to improve.";
  if (val >= 5) return "Functional but several improvements needed.";
  if (val >= 3) return "Significant issues detected. Review carefully.";
  return "Critical problems — major refactoring required.";
}

// ══════════════════════════════════════════════════════════
// OUTPUT STATE
// ══════════════════════════════════════════════════════════
function showLoadingState(label, intentLine) {
  hideAllOutputStates();
  loadingLabel.textContent = label;
  loadingIntent.textContent = intentLine;
  loadingState.style.display = "flex";
}
function resetOutput() {
  resultRoot.innerHTML = "";
  hideAllOutputStates();
  welcomeState.style.display = "flex";
}
function hideAllOutputStates() {
  welcomeState.style.display = "none";
  loadingState.style.display = "none";
  resultRoot.style.display = "none";
}

// ══════════════════════════════════════════════════════════
// STATUS
// ══════════════════════════════════════════════════════════
function setStatus(type, text) {
  statusDot.className = `status-dot ${type}`;
  statusText.textContent = text;
}

// ══════════════════════════════════════════════════════════
// TOAST
// ══════════════════════════════════════════════════════════
let toastTimer;
function showToast(message, type = "info") {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.className = `toast ${type} show`;
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3200);
}

// ══════════════════════════════════════════════════════════
// DRAG-RESIZE
// ══════════════════════════════════════════════════════════
const resizeHandle = document.getElementById("resizeHandle");
const panelEditor = document.querySelector(".panel-editor");
const panelOutput = document.querySelector(".panel-output");
const workspace = document.querySelector(".workspace");
let isResizing = false;

resizeHandle.addEventListener("mousedown", () => {
  isResizing = true;
  resizeHandle.classList.add("dragging");
  document.body.style.cursor = "col-resize";
  document.body.style.userSelect = "none";
});
document.addEventListener("mousemove", (e) => {
  if (!isResizing) return;
  const rect = workspace.getBoundingClientRect();
  const pct = Math.min(
    Math.max(((e.clientX - rect.left) / rect.width) * 100, 25),
    72,
  );
  panelEditor.style.flex = "none";
  panelEditor.style.width = `${pct}%`;
  panelOutput.style.flex = "1";
  panelOutput.style.width = "";
});
document.addEventListener("mouseup", () => {
  if (!isResizing) return;
  isResizing = false;
  resizeHandle.classList.remove("dragging");
  document.body.style.cursor = "";
  document.body.style.userSelect = "";
});

// ══════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════
(function init() {
  setStatus("ready", "Ready");
  updateLineNumbers();
  updateCharCount();
  selectIntent("analyze", INTENT_CONFIG.analyze.defaultQuery);
  breadcrumbLevel.textContent = appState.mode;
  breadcrumbUid.textContent = appState.user_id;
})();
