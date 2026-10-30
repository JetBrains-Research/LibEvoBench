// LibEvoBench leaderboard: renders static/data/leaderboard.json into the table on index.html.
(function () {
  var LIBS = ["pytorch", "numpy", "scipy"];
  var VIEWS = {
    seus: {
      label: "Overall (SEUS)",
      cols: [["rank", "#", true], ["name", "Model", true], ["seus", "SEUS"], ["stable", "Stable"], ["evolving", "Evolving"],
             ["retention", "Retention (%)"], ["stability_penalty", "Stability pen.", true], ["anachronism_penalty", "Anachr. pen.", true]],
      main: "seus",
      note: "SEUS combines API-C@L2, API-C@L3, API-I and SR across PyTorch, NumPy and SciPy. Retention = evolving / stable. Penalties are subtracted from the score (lower is better)."
    },
    apic: { label: "API Calling", task: "apic", main: "evolving",
      note: "Exact match at prompt level L1 (code + redacted docstring), on stable and evolving APIs. Ranked by the evolving-API score." },
    apii: { label: "API Identification", task: "apii", main: "evolving",
      note: "Exact match on the API name, on stable and evolving APIs. Ranked by the evolving-API score." },
    sr: { label: "Signature Recall", task: "sr", main: "evolving",
      note: "Parameter F1 against the versioned signature, on stable and evolving APIs. Ranked by the evolving-API score." },
    errors: { label: "Error types", main: "score",
      note: "Share of predictions that are correct, invalid (a real symbol, but the wrong one), anachronistic (exists in another release) or unknown (never existed). API-level tasks use version-specific APIs; parameter-level tasks use signature-changed APIs, where 'invalid' is not defined." }
  };
  var ERR_TASKS = [["apic2", "API-C@L2"], ["apii", "API-I"], ["apic3", "API-C@L3"], ["sr", "SR"]];
  var TASK_COLS = [["rank", "#", true], ["name", "Model", true], ["stable", "Stable"], ["evolving", "Evolving"], ["gap", "Gap", true]];

  var state = { view: "seus", lib: "all", errTask: "apic2", who: "all", sort: null, asc: true };
  var models = [];

  function mean(xs) { return xs.reduce(function (a, b) { return a + b; }, 0) / xs.length; }
  function fmt(x) { return x == null ? "&ndash;" : x.toFixed(1); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c]; }); }

  function libsInScope() { return state.lib === "all" ? LIBS : [state.lib]; }

  function buildRows() {
    var v = VIEWS[state.view];
    var rows = models.map(function (m) {
      var r = { name: m.name, org: m.organization, open: m.open_weights };
      if (state.view === "seus") {
        Object.keys(m.seus).forEach(function (k) { r[k] = m.seus[k]; });
      } else if (state.view === "errors") {
        var cells = libsInScope().map(function (l) { return m.errors[l][state.errTask]; });
        ["score", "correct", "invalid", "anachronistic", "unknown"].forEach(function (k) {
          r[k] = cells[0][k] == null ? null : mean(cells.map(function (c) { return c[k]; }));
        });
      } else {
        var t = libsInScope().map(function (l) { return m.tasks[l][v.task]; });
        r.stable = mean(t.map(function (c) { return c.stable; }));
        r.evolving = mean(t.map(function (c) { return c.evolving; }));
        r.gap = r.stable - r.evolving;
        if (t.length === 1) { r.stable_ci = t[0].stable_ci; r.evolving_ci = t[0].evolving_ci; }
      }
      return r;
    });
    rows.slice().sort(function (a, b) { return b[v.main] - a[v.main]; }).forEach(function (r, i) { r.rank = i + 1; });
    return rows;
  }

  function columns() {
    if (state.view === "seus") return VIEWS.seus.cols;
    if (state.view === "errors") {
      var scoreLabel = state.errTask === "apic3" ? "Param. recall" : state.errTask === "sr" ? "Param. F1" : "Accuracy";
      return [["rank", "#", true], ["name", "Model", true], ["score", scoreLabel], ["correct", "Correct (%)"],
              ["invalid", "Invalid (%)", true], ["anachronistic", "Anachronistic (%)", true], ["unknown", "Unknown (%)", true]];
    }
    return TASK_COLS;
  }

  function render() {
    var v = VIEWS[state.view];
    document.querySelectorAll("#lb-tabs li").forEach(function (li) {
      li.classList.toggle("is-active", li.dataset.view === state.view);
    });
    document.getElementById("lb-lib-field").hidden = state.view === "seus";
    document.getElementById("lb-err-field").hidden = state.view !== "errors";

    var rows = buildRows().filter(function (r) {
      return state.who === "all" || (state.who === "open" ? r.open : !r.open);
    });
    var key = state.sort || "rank";
    var asc = state.sort ? state.asc : true;
    rows.sort(function (a, b) {
      var x = a[key], y = b[key];
      if (x == null) return 1;
      if (y == null) return -1;
      var d = typeof x === "string" ? x.localeCompare(y) : x - y;
      return asc ? d : -d;
    });

    var cols = columns();
    var html = "<thead><tr>";
    cols.forEach(function (c) {
      var left = c[0] === "name" ? " class=\"has-text-left\"" : "";
      var sorted = key === c[0] ? " aria-sort=\"" + (asc ? "ascending" : "descending") + "\"" : "";
      html += "<th" + left + sorted + "><button type=\"button\" data-key=\"" + c[0] + "\" data-asc=\"" + (c[2] ? 1 : 0) + "\">" + c[1] + "</button></th>";
    });
    html += "</tr></thead><tbody>";
    rows.forEach(function (r) {
      html += "<tr>";
      cols.forEach(function (c) {
        var k = c[0];
        if (k === "rank") html += "<td>" + r.rank + "</td>";
        else if (k === "name") {
          html += "<td class=\"has-text-left\"><strong>" + esc(r.name) + "</strong>" +
            (r.open ? "<span class=\"tag is-light is-open\">open weights</span>" : "") +
            " <span class=\"org\">" + esc(r.org) + "</span></td>";
        } else {
          var ci = "";
          if (k === "stable" && r.stable_ci != null) ci = "<span class=\"ci\">&plusmn;" + r.stable_ci.toFixed(1) + "</span>";
          if (k === "evolving" && r.evolving_ci != null) ci = "<span class=\"ci\">&plusmn;" + r.evolving_ci.toFixed(1) + "</span>";
          html += "<td" + (k === v.main ? " class=\"score\"" : "") + ">" + fmt(r[k]) + ci + "</td>";
        }
      });
      html += "</tr>";
    });
    html += "</tbody>";
    var table = document.getElementById("lb-table");
    table.innerHTML = html;
    table.querySelectorAll("th button").forEach(function (b) {
      b.addEventListener("click", function () {
        var k = b.dataset.key;
        if (k === "rank") { state.sort = null; }
        else if (state.sort === k) { state.asc = !state.asc; }
        else { state.sort = k; state.asc = b.dataset.asc === "1"; }
        render();
      });
    });

    var note = v.note;
    if (state.view !== "seus") {
      note += state.lib === "all"
        ? " Values are the unweighted mean of the three libraries; select one library to see 95% bootstrap confidence intervals."
        : (state.view === "errors" ? "" : " ± is the 95% bootstrap confidence interval.");
    }
    document.getElementById("lb-note").textContent = note;
  }

  function init(data) {
    models = data.models;
    var tabs = document.getElementById("lb-tabs");
    Object.keys(VIEWS).forEach(function (k) {
      var li = document.createElement("li");
      li.dataset.view = k;
      var a = document.createElement("a");
      a.href = "#leaderboard";
      a.textContent = VIEWS[k].label;
      a.addEventListener("click", function (e) { e.preventDefault(); state.view = k; state.sort = null; render(); });
      li.appendChild(a);
      tabs.appendChild(li);
    });
    var errSel = document.getElementById("lb-err");
    ERR_TASKS.forEach(function (t) { var o = document.createElement("option"); o.value = t[0]; o.textContent = t[1]; errSel.appendChild(o); });
    document.getElementById("lb-lib").addEventListener("change", function (e) { state.lib = e.target.value; render(); });
    errSel.addEventListener("change", function (e) { state.errTask = e.target.value; state.sort = null; render(); });
    document.getElementById("lb-who").addEventListener("change", function (e) { state.who = e.target.value; render(); });
    render();
  }

  fetch("static/data/leaderboard.json")
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(init)
    .catch(function () {
      document.getElementById("lb-note").textContent =
        "The leaderboard could not load static/data/leaderboard.json. If you opened index.html straight from disk, serve the folder instead (python3 -m http.server).";
    });
})();
