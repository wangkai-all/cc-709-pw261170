/* 数据集划分工作台前端逻辑（原生 JS，无外部依赖） */
"use strict";

const SPLITS = ["train", "val", "test"];
const SPLIT_LABEL = { train: "训练", val: "验证", test: "测试", quarantine: "隔离" };
const state = {
  dataset: null,
  generation: null,
  generations: [],
  candidate: null,
  leakage: null,
};

/* ---------------- 基础工具 ---------------- */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

function pct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

let toastTimer = null;
function toast(message, kind = "") {
  const el = $("#toast");
  el.textContent = message;
  el.className = `toast ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 4200);
}

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  const payload = text ? JSON.parse(text) : {};
  if (!res.ok) {
    throw new Error(payload.error || `HTTP ${res.status}`);
  }
  return payload;
}

function hideTooltip() {
  document.querySelectorAll(".tooltip").forEach((t) => t.remove());
}
function showTooltip(event, html) {
  hideTooltip();
  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.innerHTML = html;
  document.body.appendChild(tip);
  const move = (ev) => {
    tip.style.left = `${Math.min(ev.clientX + 14, window.innerWidth - 340)}px`;
    tip.style.top = `${ev.clientY + 16}px`;
  };
  move(event);
  tip.addEventListener("mouseleave", hideTooltip);
  document.addEventListener("mousemove", move, { once: true });
  setTimeout(hideTooltip, 6000);
}

/* ---------------- Tabs ---------------- */

$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    $$(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $(`#panel-${tab.dataset.tab}`).classList.add("active");
    if (tab.dataset.tab === "versions") refreshGenerations();
    if (tab.dataset.tab === "revision") refreshRelationTable();
  });
});

/* ---------------- 数据集 ---------------- */

async function refreshDatasets(selectName) {
  const data = await api("GET", "/api/datasets");
  const select = $("#dataset-select");
  select.innerHTML = '<option value="">— 请选择或新建 —</option>' +
    data.datasets
      .map((d) => `<option value="${escapeHtml(d.name)}">${escapeHtml(d.name)}（${escapeHtml(d.current_relation_version)}）</option>`)
      .join("");
  if (selectName) select.value = selectName;
}

$("#dataset-select").addEventListener("change", async (e) => {
  const name = e.target.value;
  if (!name) {
    state.dataset = null;
    return;
  }
  await loadDataset(name);
});

$("#btn-refresh").addEventListener("click", () => refreshDatasets(state.dataset?.name));

async function loadDataset(name) {
  state.dataset = await api("GET", `/api/datasets/${encodeURIComponent(name)}`);
  state.generation = null;
  const meta = $("#dataset-meta");
  meta.classList.remove("hidden");
  meta.innerHTML = `
    <strong>${escapeHtml(state.dataset.name)}</strong><br>
    样本版本 <code>${escapeHtml(state.dataset.sample_version)}</code> ·
    关系版本 <code>${escapeHtml(state.dataset.relation_version)}</code><br>
    样本 ${state.dataset.sample_count} 条 · 关系 ${state.dataset.relation_count} 条 ·
    主体 ${new Set(state.dataset.samples.map((s) => s.subject_id)).size} 个`;
  await refreshRelationTable();
  toast(`已载入数据集 ${name}`, "success");
}

$("#btn-load-example").addEventListener("click", async () => {
  try {
    const result = await api("POST", "/api/datasets/example", { name: "lake_demo" });
    await refreshDatasets("lake_demo");
    await loadDataset("lake_demo");
    toast(`内置示例已创建：${result.samples} 个样本，${result.relations} 条关系`, "success");
  } catch (err) {
    toast(err.message, "error");
  }
});

$("#form-create").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  try {
    const result = await api("POST", "/api/datasets", {
      name: fd.get("name").trim(),
      samples_csv: fd.get("samples_csv"),
      relations_csv: fd.get("relations_csv"),
    });
    await refreshDatasets(result.name);
    await loadDataset(result.name);
    e.target.reset();
  } catch (err) {
    toast(err.message, "error");
  }
});

/* ---------------- 配置与生成 ---------------- */

function readConfig() {
  const fd = new FormData($("#form-config"));
  const typesRaw = (fd.get("relation_types") || "").trim();
  return {
    seed: Number(fd.get("seed")),
    target_ratios: {
      train: Number(fd.get("train")),
      val: Number(fd.get("val")),
      test: Number(fd.get("test")),
    },
    train_cutoff: fd.get("train_cutoff") || null,
    val_cutoff: fd.get("val_cutoff") || null,
    cross_boundary_rule: fd.get("cross_boundary_rule"),
    relation_types: typesRaw ? typesRaw.split(",").map((s) => s.trim()).filter(Boolean) : null,
    exhaustive_limit: Number(fd.get("exhaustive_limit")),
  };
}

$("#form-config").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.dataset) {
    toast("请先选择或创建数据集", "error");
    return;
  }
  try {
    const gen = await api("POST", "/api/generate", {
      dataset: state.dataset.name,
      config: readConfig(),
    });
    state.generation = gen;
    state.leakage = gen.leakage_report || null;
    renderGeneration(gen);
    await refreshDatasets(state.dataset.name);
    toast(`生成完成：${gen.generation_id}`, "success");
  } catch (err) {
    toast(err.message, "error");
  }
});

/* ---------------- 图表渲染 ---------------- */

function renderRatioChart(gen) {
  const alloc = gen.allocation;
  const target = alloc.target_ratios;
  const actual = alloc.actual_ratios;
  const counts = alloc.split_samples;
  const maxCount = Math.max(1, ...SPLITS.map((sp) => counts[sp]), ...SPLITS.map((sp) => alloc.target_samples[sp]));

  const rows = SPLITS.map((sp) => {
    const actualWidth = (counts[sp] / maxCount) * 100;
    const targetWidth = (alloc.target_samples[sp] / maxCount) * 100;
    const dev = alloc.deviation[sp];
    return `
    <div class="bar-row" data-tip='${escapeHtml(JSON.stringify({
      集合: SPLIT_LABEL[sp],
      目标比例: pct(target[sp]),
      实际比例: pct(actual[sp]),
      目标样本数: alloc.target_samples[sp].toFixed(1),
      实际样本数: counts[sp],
      偏差: `${dev >= 0 ? "+" : ""}${dev.toFixed(1)} 个样本`,
    }))}'>
      <div class="bar-label"><span class="swatch ${sp}"></span>${SPLIT_LABEL[sp]} ${sp}</div>
      <div class="bar-track">
        <div class="bar-fill target ${sp}" style="width:${targetWidth}%"></div>
        <div class="bar-fill ${sp}" style="width:${actualWidth}%"></div>
      </div>
      <div class="bar-value">${pct(actual[sp])} / 目标 ${pct(target[sp])}</div>
    </div>`;
  }).join("");

  $("#ratio-chart").innerHTML = `
    <div class="legend">
      <span><span class="swatch train"></span> 实心条 = 实际样本数</span>
      <span><span class="swatch val"></span> 斜纹条 = 目标样本量</span>
    </div>${rows}`;
  bindTooltips("#ratio-chart .bar-row");

  const noteClass = { exhaustive: "info", greedy: "info" };
  $("#deviation-notes").innerHTML = alloc.notes
    .map((n, i) => `<div class="note ${i === 0 ? "warn" : noteClass[alloc.strategy] || ""}">${escapeHtml(n)}</div>`)
    .join("");
}

function renderGroupChart(gen) {
  const groups = [...gen.groups].sort((a, b) => b.sample_count - a.sample_count);
  const maxN = Math.max(1, ...groups.map((g) => g.sample_count));
  const total = groups.reduce((acc, g) => acc + g.sample_count, 0);
  $("#group-chart").innerHTML = groups.map((g) => {
    const split = g.assigned_split || (g.quarantined ? "quarantine" : "unassigned");
    const color = g.quarantined ? "quarantine" : (g.assigned_split || "");
    return `
    <div class="bar-row group-bar" data-tip='${escapeHtml(JSON.stringify({
      组: g.group_id,
      主体数: g.subject_count,
      样本数: g.sample_count,
      占总量: pct(g.sample_count / total),
      日期范围: `${g.min_date} ~ ${g.max_date}`,
      跨边界: g.crosses_boundary ? "是" : "否",
      去向: SPLIT_LABEL[split] || split,
      连接边: g.edge_ids.join(", ") || "无（单主体组）",
    }))}'>
      <div class="bar-label"><code>${escapeHtml(g.group_id)}</code></div>
      <div class="bar-track"><div class="bar-fill ${color}" style="width:${(g.sample_count / maxN) * 100}%"></div></div>
      <div class="bar-value">${g.sample_count} 样本 / ${g.subject_count} 主体</div>
    </div>`;
  }).join("");
  bindTooltips("#group-chart .bar-row");
}

function bindTooltips(selector) {
  $$(selector).forEach((row) => {
    row.addEventListener("mouseenter", (ev) => {
      const data = JSON.parse(row.dataset.tip);
      showTooltip(ev, Object.entries(data).map(([k, v]) => `<div><strong>${escapeHtml(k)}</strong>: ${escapeHtml(v)}</div>`).join(""));
    });
    row.addEventListener("mouseleave", hideTooltip);
  });
}

function renderGroupTable(gen) {
  const body = $("#group-table tbody");
  body.innerHTML = gen.groups.map((g) => {
    const split = g.assigned_split || (g.quarantined ? "quarantine" : null);
    const pill = split
      ? `<span class="pill ${split}">${SPLIT_LABEL[split]} ${split}</span>`
      : '<span class="pill muted">未分配</span>';
    return `<tr>
      <td><code>${escapeHtml(g.group_id)}</code></td>
      <td>${g.subject_count}</td>
      <td><strong>${g.sample_count}</strong></td>
      <td class="mono">${escapeHtml(g.min_date)} ~ ${escapeHtml(g.max_date)}</td>
      <td>${g.crosses_boundary ? '<span class="pill crit">跨边界</span>' : '<span class="pill muted">否</span>'}</td>
      <td>${g.eligible_splits.map((sp) => `<span class="pill ${sp}">${sp}</span>`).join(" ")}</td>
      <td>${pill}</td>
      <td>${escapeHtml(g.reason || "")}</td>
    </tr>`;
  }).join("");
}

/* ---------------- 清单下载与预览 ---------------- */

function renderManifests(gen) {
  $("#freeze-box").innerHTML = `
    <strong>冻结指纹 freeze_id</strong>：<code>${escapeHtml(gen.freeze_id)}</code><br>
    generation_id：<code>${escapeHtml(gen.generation_id)}</code> ·
    样本版 <code>${escapeHtml(gen.sample_version)}</code> ·
    关系版 <code>${escapeHtml(gen.relation_version)}</code><br>
    样本哈希 <code>${escapeHtml(gen.sample_hash.slice(0, 16))}…</code> ·
    关系哈希 <code>${escapeHtml(gen.relation_hash.slice(0, 16))}…</code> ·
    策略 <code>${escapeHtml(gen.allocation.strategy)}</code> ·
    种子 <code>${escapeHtml(String(gen.config.seed))}</code>`;

  const cards = SPLITS.map((sp) => {
    const rows = gen.manifests[sp];
    return `
    <div class="download-card">
      <span class="pill ${sp}">${SPLIT_LABEL[sp]} ${sp}</span>
      <span class="big">${rows.length}</span>
      <span class="hint">个样本 · ${new Set(rows.map((r) => r.subject_id)).size} 个主体 · ${new Set(rows.map((r) => r.group_id)).size} 个组</span>
      <a class="btn small btn-primary" href="/api/generations/${encodeURIComponent(gen.generation_id)}/manifest/${sp}" download>下载 ${sp}.csv</a>
    </div>`;
  });
  const qCount = gen.quarantine.length;
  cards.push(`
    <div class="download-card">
      <span class="pill quarantine">隔离 quarantine</span>
      <span class="big">${qCount}</span>
      <span class="hint">跨边界且按 isolate 规则整组隔离的样本</span>
      <a class="btn small" href="/api/generations/${encodeURIComponent(gen.generation_id)}/manifest/quarantine" download>下载 quarantine.csv</a>
    </div>`);
  $("#downloads").innerHTML = cards.join("");

  const previewRows = SPLITS.flatMap((sp) =>
    gen.manifests[sp].slice(0, 10).map((r) => ({ ...r, split: sp }))
  );
  $("#manifest-preview").innerHTML = tableHtml(previewRows, ["split", "sample_id", "subject_id", "event_date", "group_id"]);
}

function tableHtml(rows, cols) {
  if (!rows.length) return '<p class="hint">无数据行</p>';
  const head = cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("");
  const body = rows.map((r) =>
    `<tr>${cols.map((c) => {
      const v = r[c] ?? "";
      const cls = SPLITS.includes(c) || c === "split" ? `pill ${v}` : "";
      return `<td class="${cls}">${escapeHtml(v)}</td>`;
    }).join("")}</tr>`
  ).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderGeneration(gen) {
  renderRatioChart(gen);
  renderGroupChart(gen);
  renderGroupTable(gen);
  renderManifests(gen);
  // 自动切到解释页，方便看大组偏差
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === "groups"));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-groups"));
}

/* ---------------- 关系修订 ---------------- */

async function refreshRelationTable() {
  if (!state.dataset) return;
  const rels = state.dataset.relations;
  $("#relation-table").innerHTML = tableHtml(
    rels.map((r) => ({ ...r, active: r.active ? "true" : "false" })),
    ["rel_id", "src_subject", "dst_subject", "rel_type", "active"]
  );
}

$("#form-withdraw").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.dataset) return toast("请先选择数据集", "error");
  const relId = new FormData(e.target).get("rel_id").trim();
  try {
    const result = await api("POST", "/api/revise/withdraw", {
      dataset: state.dataset.name,
      rel_id: relId,
    });
    toast(`关系 ${result.withdrawn} 已撤回，关系版升至 ${result.relation_version}`, "success");
    e.target.reset();
    await loadDataset(state.dataset.name);
  } catch (err) {
    toast(err.message, "error");
  }
});

$("#form-merge").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.dataset) return toast("请先选择数据集", "error");
  const fd = new FormData(e.target);
  try {
    const result = await api("POST", "/api/revise/merge", {
      dataset: state.dataset.name,
      subject_a: fd.get("subject_a").trim(),
      subject_b: fd.get("subject_b").trim(),
      rel_type: fd.get("rel_type").trim() || "same_as",
    });
    toast(`新关系 ${result.added_relation} 已写入，关系版升至 ${result.relation_version}`, "success");
    e.target.reset();
    await loadDataset(state.dataset.name);
  } catch (err) {
    toast(err.message, "error");
  }
});

$("#btn-candidate").addEventListener("click", async () => {
  if (!state.dataset) return toast("请先选择数据集", "error");
  if (!state.generation) return toast("请先生成一个基线版本（旧清单将原样保留）", "error");
  try {
    const result = await api("POST", "/api/revise/candidate", {
      dataset: state.dataset.name,
      old_generation_id: state.generation.generation_id,
      config: readConfig(),
    });
    state.candidate = result.candidate;
    state.leakage = result.leakage;
    renderLeakage(result.leakage, result.old_generation, result.candidate);
    toast("候选版已生成，旧清单未被修改", "success");
  } catch (err) {
    toast(err.message, "error");
  }
});

function renderLeakage(report, oldGen, candidate) {
  const box = $("#leakage-box");
  let html = "";
  if (report.leakage) {
    html += `<div class="conflict">
      <strong class="pill crit">发现潜在泄漏</strong>
      <p>${escapeHtml(report.summary)}</p>
      <p class="hint">旧版 <code>${escapeHtml(report.old_generation_id)}</code>（${escapeHtml(report.old_relation_version)}）的清单文件原样保留，
      以下冲突只标注在新候选 <code>${escapeHtml(candidate.generation_id)}</code>（${escapeHtml(report.new_relation_version)}）上。</p>`;
    html += report.cross_split_conflicts.map((c) => `
      <div class="conflict">
        <div><strong>新组</strong> <code>${escapeHtml(c.new_group_id)}</code> ·
        旧集合去向 <strong>${escapeHtml(c.old_splits.join(" / "))}</strong> ·
        涉及 ${c.sample_count} 个样本</div>
        <div class="hint">主体：${c.subjects.map(escapeHtml).join(", ")}</div>
        <div class="hint">连通边：${c.connecting_edges.map((e) => `<code>${escapeHtml(e)}</code>`).join(", ")}</div>
        <p>${escapeHtml(c.message)}</p>
      </div>`).join("");
    html += "</div>";
  } else {
    html += `<div class="ok-box"><span class="pill good">无跨集合冲突</span><p>${escapeHtml(report.summary)}</p></div>`;
  }
  html += `<h3>关系变更</h3>
    <p class="hint">新增 ${report.added_relations.length} 条 · 撤回 ${report.withdrawn_relations.length} 条</p>
    ${tableHtml(
      report.added_relations.map((r) => ({ 动作: "新增", ...r })),
      ["动作", "rel_id", "src", "dst", "rel_type", "active"]
    )}
    ${report.withdrawn_relations.length ? tableHtml(
      report.withdrawn_relations.map((r) => ({ 动作: "撤回", ...r })),
      ["动作", "rel_id", "src", "dst", "rel_type"]
    ) : ""}
    <div class="row" style="margin-top:12px">
      <button class="btn" id="btn-view-candidate">查看候选版分配解释</button>
    </div>`;
  box.innerHTML = html;
  $("#btn-view-candidate").addEventListener("click", () => {
    state.generation = candidate;
    renderGeneration(candidate);
    toast("已切换视图到候选版（旧版仍在生成历史中，未被覆盖）");
  });
}

/* ---------------- 版本历史、对比、发布 ---------------- */

async function refreshGenerations() {
  if (!state.dataset) return;
  const data = await api("GET", `/api/generations?dataset=${encodeURIComponent(state.dataset.name)}`);
  state.generations = data.generations;
  $("#gen-table tbody").innerHTML = data.generations.map((g) => `
    <tr>
      <td><code>${escapeHtml(g.generation_id)}</code></td>
      <td><code>${escapeHtml(g.freeze_id.slice(0, 16))}…</code></td>
      <td>${escapeHtml(g.relation_version)}</td>
      <td>${escapeHtml(g.strategy)}</td>
      <td class="mono">train ${pct(g.actual_ratios.train)} · val ${pct(g.actual_ratios.val)} · test ${pct(g.actual_ratios.test)}</td>
      <td>${g.published ? `<span class="pill good">已发布 ${escapeHtml(g.published_as)}</span>` : '<span class="pill muted">候选</span>'}</td>
      <td class="mono">${escapeHtml(g.created_at)}</td>
      <td class="row">
        <button class="btn small" data-action="view" data-id="${escapeHtml(g.generation_id)}">查看</button>
        ${g.published ? "" : `<button class="btn small btn-primary" data-action="publish" data-id="${escapeHtml(g.generation_id)}">确认发布</button>`}
      </td>
    </tr>`).join("");

  const options = data.generations
    .map((g) => `<option value="${escapeHtml(g.generation_id)}">${escapeHtml(g.generation_id)}（${escapeHtml(g.relation_version)}）</option>`)
    .join("");
  $("#compare-old").innerHTML = options;
  $("#compare-new").innerHTML = options;
  if (data.generations.length >= 2) {
    $("#compare-old").value = data.generations[data.generations.length - 1].generation_id;
    $("#compare-new").value = data.generations[0].generation_id;
  }
  await refreshPublications();
}

$("#gen-table").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-action]");
  if (!btn) return;
  const id = btn.dataset.id;
  if (btn.dataset.action === "view") {
    state.generation = await api("GET", `/api/generations/${encodeURIComponent(id)}`);
    renderGeneration(state.generation);
  } else if (btn.dataset.action === "publish") {
    if (!confirm(`确认发布 ${id}？发布后该版本不可变，旧发布版本继续保留。`)) return;
    try {
      const result = await api("POST", "/api/publish", { generation_id: id });
      toast(`已发布为 ${result.published_as}`, "success");
      await refreshGenerations();
    } catch (err) {
      toast(err.message, "error");
    }
  }
});

$("#btn-compare").addEventListener("click", async () => {
  const oldId = $("#compare-old").value;
  const newId = $("#compare-new").value;
  if (!oldId || !newId || oldId === newId) {
    toast("请选择两个不同的版本", "error");
    return;
  }
  const cmp = await api("GET", `/api/compare?old=${encodeURIComponent(oldId)}&new=${encodeURIComponent(newId)}`);
  const ratioRows = SPLITS.map((sp) => `
    <tr><td><span class="pill ${sp}">${sp}</span></td>
    <td class="mono">${pct(cmp.ratio_table.target[sp])}</td>
    <td class="mono">${pct(cmp.ratio_table.old_actual[sp])}（${cmp.sample_count_table.old[sp]} 个）</td>
    <td class="mono">${pct(cmp.ratio_table.new_actual[sp])}（${cmp.sample_count_table.new[sp]} 个）</td></tr>`).join("");
  $("#compare-result").innerHTML = `
    <p class="hint">旧版 <code>${escapeHtml(cmp.old_generation_id)}</code>（${escapeHtml(cmp.old_relation_version)}）
    → 新版 <code>${escapeHtml(cmp.new_generation_id)}</code>（${escapeHtml(cmp.new_relation_version)}），
    共 <strong>${cmp.samples_moved_count}</strong> 个样本跨集合移动。</p>
    <table><thead><tr><th>集合</th><th>目标</th><th>旧实际</th><th>新实际</th></tr></thead><tbody>${ratioRows}</tbody></table>
    <h3>移动明细</h3>
    ${tableHtml(cmp.samples_moved, ["sample_id", "from", "to"])}`;
});

async function refreshPublications() {
  const data = await api("GET", `/api/publications?dataset=${encodeURIComponent(state.dataset.name)}`);
  if (!data.publications.length) {
    $("#pub-list").innerHTML = '<p class="hint">还没有发布版本。候选版经人工确认后发布，发布目录只增不改。</p>';
    return;
  }
  $("#pub-list").innerHTML = data.publications.map((p) => `
    <div class="pub-item">
      <div><strong class="pill good">${escapeHtml(p.published_as)}</strong>
        <code>${escapeHtml(p.generation_id)}</code>
        <span class="hint">关系版 ${escapeHtml(p.relation_version)} · ${escapeHtml(p.created_at)}</span></div>
      <div class="hint mono">${escapeHtml(p.freeze_id.slice(0, 20))}…</div>
    </div>`).join("");
}

/* ---------------- 启动 ---------------- */

refreshDatasets().catch((err) => toast(err.message, "error"));
