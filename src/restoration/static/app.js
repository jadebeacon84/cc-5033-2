(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const logEl = $("log");
  const log = (...a) => {
    logEl.textContent += a.map(x => typeof x === "string" ? x : JSON.stringify(x)).join(" ") + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  };
  const api = async (url, opts) => {
    const res = await fetch(url, opts);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error || res.statusText);
    return body;
  };

  const state = {
    sid: null, meta: null,
    img: null,                 // HTMLImageElement of the working-resolution original
    maskCanvas: null,          // offscreen canvas at full native resolution
    maskCtx: null,
    evidenceImg: null,
    zoom: 1, ox: 0, oy: 0,
    fit: 1,
    viewW: 0, viewH: 0,
    drawing: false,
    pointerId: null,
    stroke: [],
    detections: null,
    origW: 0, origH: 0,
    workW: 0, workH: 0,      // display-resolution original (same aspect)
  };

  // ------------------------------------------------------------------ upload
  $("file").addEventListener("change", () => { $("btn-upload").disabled = !$("file").files.length; });
  $("btn-upload").addEventListener("click", async () => {
    const f = $("file").files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("image", f);
    try {
      const meta = await api("/api/sessions", { method: "POST", body: fd });
      state.sid = meta.id;
      state.meta = meta;
      $("panel-editor").classList.remove("hidden");
      $("panel-run").classList.remove("hidden");
      $("stage-empty").classList.add("hidden");
      await loadOriginal();
      clearMask();
      updateMaskInfo();
      log("载入", meta.source_name, `${meta.width}×${meta.height}`, meta.dtype, `${meta.channels}ch`);
      fitToView();
    } catch (e) { log("载入失败:", e.message); }
  });

  async function loadOriginal() {
    const img = new Image();
    img.src = `/media/${state.sid}/preview_original.png?ts=${Date.now()}`;
    await img.decode();
    state.img = img;
    state.workW = img.naturalWidth;
    state.workH = img.naturalHeight;
    state.origW = state.meta.width;
    state.origH = state.meta.height;
  }

  // ------------------------------------------------------------------- mask
  function clearMask() {
    state.maskCanvas = document.createElement("canvas");
    state.maskCanvas.width = state.origW;
    state.maskCanvas.height = state.origH;
    state.maskCtx = state.maskCanvas.getContext("2d", { willReadFrequently: true });
    state.maskCtx.fillStyle = "#000";
    state.maskCtx.fillRect(0, 0, state.origW, state.origH);
  }

  async function pushStrokes(strokes) {
    if (!state.sid) return;
    try {
      const res = await api(`/api/sessions/${state.sid}/mask`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strokes })
      });
      $("mask-info").textContent = `掩码 ${res.mask_pixels} px · ${res.mask_components} 连通块`;
    } catch (e) { log("掩码提交失败:", e.message); }
  }

  function updateMaskInfo() {
    if (!state.sid) return;
    api(`/api/sessions/${state.sid}/mask_stats`).then(r => {
      $("mask-info").textContent = `掩码 ${r.mask_pixels} px · ${r.mask_components} 连通块`;
    }).catch(() => {});
  }

  $("brush").addEventListener("input", e => { $("brush-out").value = e.target.value; });
  $("btn-clear").addEventListener("click", async () => {
    if (!state.sid) return;
    clearMask(); draw();
    await api(`/api/sessions/${state.sid}/mask`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clear: true })
    }).catch(e => log("清空失败:", e.message));
    updateMaskInfo();
  });
  $("btn-fit").addEventListener("click", fitToView);
  $("btn-zoomin").addEventListener("click", () => setZoom(state.zoom * 1.25));
  $("btn-zoomout").addEventListener("click", () => setZoom(state.zoom / 1.25));
  $("show-overlay").addEventListener("change", draw);
  $("show-evidence").addEventListener("change", draw);

  // ---------------------------------------------------------------- canvas
  const stage = $("stage");
  function resize() {
    const wrap = stage.parentElement;
    const dpr = window.devicePixelRatio || 1;
    stage.width = wrap.clientWidth * dpr;
    stage.height = wrap.clientHeight * dpr;
    stage.style.width = wrap.clientWidth + "px";
    stage.style.height = wrap.clientHeight + "px";
    state.viewW = wrap.clientWidth;
    state.viewH = wrap.clientHeight;
    state.dpr = dpr;
    if (!state.zoomManually) fitToView();
    else draw();
  }
  new ResizeObserver(resize).observe(stage.parentElement);

  function fitToView() {
    if (!state.workW) return;
    state.zoomManually = false;
    const pad = 20;
    const s = Math.min((state.viewW - pad) / state.workW, (state.viewH - pad) / state.workH);
    state.fit = s;
    state.zoom = s;
    state.ox = (state.viewW - state.workW * s) / 2;
    state.oy = (state.viewH - state.workH * s) / 2;
    $("zoom-out").value = Math.round((s / state.fit) * 100) + "%";
    draw();
  }

  function setZoom(z) {
    if (!state.workW) return;
    state.zoomManually = true;
    const cx = state.viewW / 2, cy = state.viewH / 2;
    const ix = (cx - state.ox) / state.zoom, iy = (cy - state.oy) / state.zoom;
    state.zoom = Math.max(state.fit * 0.5, Math.min(state.fit * 16, z));
    state.ox = cx - ix * state.zoom;
    state.oy = cy - iy * state.zoom;
    $("zoom-out").value = Math.round((state.zoom / state.fit) * 100) + "%";
    draw();
  }

  function toNative(clientX, clientY) {
    const r = stage.getBoundingClientRect();
    const x = (clientX - r.left - state.ox) / state.zoom;
    const y = (clientY - r.top - state.oy) / state.zoom;
    return { x: x * (state.origW / state.workW), y: y * (state.origH / state.workH) };
  }

  function draw() {
    if (!state.img) return;
    const ctx = stage.getContext("2d");
    const dpr = state.dpr || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, stage.width, stage.height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    const dx = state.ox, dy = state.oy;
    const dw = state.workW * state.zoom, dh = state.workH * state.zoom;
    ctx.drawImage(state.img, dx, dy, dw, dh);

    if ($("show-evidence").checked && state.evidenceImg) {
      ctx.globalAlpha = 0.65;
      ctx.drawImage(state.evidenceImg, dx, dy, dw, dh);
      ctx.globalAlpha = 1;
    }

    if ($("show-overlay").checked && state.maskCanvas) {
      // Draw the mask tinted, aligned to the displayed image.
      const tmp = document.createElement("canvas");
      tmp.width = state.workW; tmp.height = state.workH;
      const tctx = tmp.getContext("2d");
      tctx.drawImage(state.maskCanvas, 0, 0, state.workW, state.workH);
      // tint: white mask -> cyan, alpha
      const id = tctx.getImageData(0, 0, state.workW, state.workH);
      const d = id.data;
      for (let i = 0; i < d.length; i += 4) {
        const on = d[i] > 40;
        d[i] = 40; d[i + 1] = on ? 200 : 0; d[i + 2] = on ? 255 : 0;
        d[i + 3] = on ? 150 : 0;
      }
      tctx.putImageData(id, 0, 0);
      ctx.drawImage(tmp, dx, dy, dw, dh);
    }

    // scale bar / zoom text is DOM, not canvas
    $("zoom-out").value = Math.round((state.zoom / state.fit) * 100) + "%";
  }

  // -------------------------------------------------------------- painting
  stage.addEventListener("pointerdown", e => {
    if (!state.sid || !state.img) return;
    if (e.button === 1 || (e.button === 0 && e.altKey)) return; // (middle/alt reserved)
    state.drawing = true;
    state.pointerId = e.pointerId;
    stage.setPointerCapture(e.pointerId);
    const p = toNative(e.clientX, e.clientY);
    state.stroke = [p];
    paintSegment(p, p);
  });
  stage.addEventListener("pointermove", e => {
    if (!state.drawing || e.pointerId !== state.pointerId) return;
    const p = toNative(e.clientX, e.clientY);
    const last = state.stroke[state.stroke.length - 1];
    state.stroke.push(p);
    paintSegment(last, p);
  });
  stage.addEventListener("pointerup", async e => {
    if (!state.drawing) return;
    state.drawing = false;
    const strokes = [{ points: state.stroke.map(p => [Math.round(p.x), Math.round(p.y)]),
      radius: parseInt($("brush").value, 10), mode: $("brush-mode").value }];
    state.stroke = [];
    draw();
    await pushStrokes(strokes);
  });

  function paintSegment(a, b) {
    if (!state.maskCanvas) return;
    const ctx = state.maskCtx;
    const scale = state.origW / state.workW;
    const r = parseInt($("brush").value, 10) * scale;
    ctx.strokeStyle = $("brush-mode").value === "erase" ? "#000" : "#fff";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = r * 2;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    if (r <= 2) { // ensure at least a dot
      ctx.beginPath(); ctx.arc(b.x, b.y, r, 0, Math.PI * 2); ctx.fill();
    }
    draw();
  }

  // ---------------------------------------------------------------- detect
  $("btn-detect").addEventListener("click", async () => {
    if (!state.sid) return;
    try {
      const r = await api(`/api/sessions/${state.sid}/detect`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: "{}"
      });
      state.detections = r;
      renderCandidates(r);
      $("panel-cand").classList.remove("hidden");
      $("btn-apply-cand").disabled = false;
      $("show-evidence").disabled = false;
      $("show-evidence").checked = true;
      const ev = new Image();
      ev.src = `/media/${state.sid}/candidates_evidence.png?ts=${Date.now()}`;
      await ev.decode();
      state.evidenceImg = ev;
      $("evidence-img").src = ev.src;
      draw();
      $("detect-info").textContent =
        `候选 ${r.candidates.length} 个 · 接受 ${r.candidates.filter(c => c.accepted).length} · ` +
        `否定 ${r.candidates.filter(c => !c.accepted).length}`;
      log("检测:", JSON.stringify(r.diagnostics));
    } catch (e) { log("检测失败:", e.message); }
  });

  function renderCandidates(r) {
    const tb = $("cand-table").querySelector("tbody");
    tb.innerHTML = "";
    for (const c of r.candidates) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${c.id}</td><td>${c.kind === "line" ? "细线" : "灰尘"}</td>
        <td class="${c.accepted ? "acc" : "rej"}">${c.accepted ? "接受" : "否定"}</td>
        <td>${escapeHtml(c.reason)}</td><td>${c.length}</td><td>${c.line_contrast}</td>
        <td>${c.coverage}</td><td>${c.texture_ratio}</td>
        <td>${c.end_contrast[0]}/${c.end_contrast[1]}</td><td>${c.drift}</td>
        <td>${c.parallel_partners}</td>`;
      tb.appendChild(tr);
    }
  }
  function escapeHtml(s) { return s.replace(/[&<>"]/g, t => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[t])); }

  $("btn-apply-cand").addEventListener("click", async () => {
    if (!state.sid) return;
    try {
      const r = await api(`/api/sessions/${state.sid}/mask/apply_candidates`, { method: "POST" });
      // reload native mask into the canvas
      const img = new Image();
      img.src = `/media/${state.sid}/mask.png?ts=${Date.now()}`;
      await img.decode();
      clearMask();
      state.maskCtx.drawImage(img, 0, 0, state.origW, state.origH);
      draw();
      $("mask-info").textContent = `掩码 ${r.mask_pixels} px`;
      log("已将已接受候选并入掩码，请在画布上复核后再执行修复。");
    } catch (e) { log("并入候选失败:", e.message); }
  });

  // ------------------------------------------------------------------ run
  $("btn-run").addEventListener("click", async () => {
    if (!state.sid) return;
    const body = {
      method: $("method").value,
      band_px: parseInt($("band").value, 10),
      texture_threshold: parseFloat($("texth").value),
      patch: parseInt($("patch").value, 10),
      search_radius: parseInt($("search").value, 10),
      seed: parseInt($("seed").value, 10),
      speculative_width: parseFloat($("specw").value),
    };
    $("btn-run").disabled = true;
    log("执行修复…", JSON.stringify(body));
    try {
      const rep = await api(`/api/sessions/${state.sid}/restore`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
      });
      renderResult(rep);
      log("完成，用时", rep.timing_s, "秒");
    } catch (e) { log("修复失败:", e.message); }
    finally { $("btn-run").disabled = false; }
  });

  function bust(url) { return url + "?ts=" + Date.now(); }

  function renderResult(rep) {
    $("panel-result").classList.remove("hidden");
    $("panel-report").classList.remove("hidden");
    $("view-original").src = bust(rep.media.original);
    $("view-mask").src = bust(rep.media.mask);
    $("view-restored").src = bust(rep.media.restored);
    $("view-difference").src = bust(rep.media.difference);
    $("view-boundary").src = bust(rep.media.boundary);
    $("view-speculative").src = bust(rep.media.speculative);
    $("dl-restored").href = rep.downloads.restored;
    $("dl-mask").href = rep.downloads.mask;
    $("dl-report").href = rep.downloads.report;

    const p = rep.preservation;
    const facts = [
      ["掩码外改动样本", p.changed_outside_allowed, p.ok ? "ok" : "bad"],
      ["保留像素（逐样本核对）", p.outside_allowed_pixels, ""],
      ["核心改动样本", p.core_changed, ""],
      ["过渡带改动样本", p.band_changed, ""],
      ["过渡带宽", rep.zones.band_px_used + " px", ""],
      ["推测像素（宽缺口）", rep.speculation.speculative_pixels, rep.speculation.speculative_pixels ? "warn" : ""],
      ["缺口宽度下界估计", rep.speculation.gap_width_floor_px + " px", ""],
      ["方法", rep.parameters.method, ""],
      ["区域数", rep.regions.length, ""],
      ["用时", rep.timing_s + " s", ""],
    ];
    $("facts").innerHTML = facts.map(([k, v, cls]) =>
      `<div class="fact"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`).join("");

    const tb = $("region-table").querySelector("tbody");
    tb.innerHTML = rep.regions.map(r =>
      `<tr><td>${r.region_id}</td><td>${r.method}</td><td>${r.texture_energy.toFixed(5)}</td>
       <td>${r.area}</td><td>${r.bbox.join(", ")}</td></tr>`).join("");

    const notes = [];
    if (p.ok) notes.push({ cls: "ok", text: `保留核对通过：${p.note}` });
    else notes.push({ cls: "warn", text: `保留核对失败：${p.note}` });
    for (const n of rep.method_notes || []) notes.push({ cls: "", text: n });
    for (const n of rep.zones.notes || []) notes.push({ cls: "warn", text: "边界：" + n });
    notes.push({ cls: "warn", text: rep.speculation.statement });
    $("notes").innerHTML = notes.map(n =>
      `<div class="note ${n.cls}">${escapeHtml(n.text)}</div>`).join("");
  }

  // initial
  resize();
})();
