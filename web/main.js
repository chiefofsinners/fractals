import init, { render_mandelbrot, reference_orbit } from "./pkg/fractals.js";
import { createGpuRenderer } from "./gpu.js";
import { createWebGpuRenderer } from "./webgpu.js";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const gpuCanvas = document.getElementById("gpu");
const webgpuCanvas = document.getElementById("webgpu");
const axesCanvas = document.getElementById("axes");
const axesCtx = axesCanvas.getContext("2d");
const surface = canvas.parentElement; // <main>
const iterInput = document.getElementById("iter");
const iterRange = document.getElementById("iterRange");
const iterValue = document.getElementById("iterValue");
const showAxes = document.getElementById("showAxes");
const rendererSelect = document.getElementById("renderer");
const resetBtn = document.getElementById("reset");
const status = document.getElementById("status");

const webgl = createGpuRenderer(gpuCanvas);
let webgpu = null; // populated asynchronously by boot()

// Below this scale, f32 precision in the WebGL shader breaks down.
const WEBGL_MIN_SCALE = 1e-5;
// Perturbation does the per-pixel work in f32 deltas around an f64
// reference orbit, so its precision is bounded by f64 on the reference
// (ulp ~ 2e-16 near unit magnitude). Below ~1e-15 the reference orbit
// itself loses meaningful digits and we fall back to the f64 CPU path.
const WEBGPU_MIN_SCALE = 1e-15;

function activeBackend() {
  const want = rendererSelect.value;
  if (want === "webgpu") {
    if (webgpu) return "webgpu";
    if (webgl) return "webgl";
    return "cpu";
  }
  if (want === "webgl") {
    if (webgl) return "webgl";
    return "cpu";
  }
  return "cpu";
}

// Smallest view.scale we'll let the user zoom to with each backend. Past
// these values the per-pixel maths starts producing visibly wrong output
// (flat blobs of colour, banding) so we just refuse to zoom further rather
// than silently swap to a slower fallback.
function minScaleForBackend(backend) {
  if (backend === "webgl") return WEBGL_MIN_SCALE;
  if (backend === "webgpu") return WEBGPU_MIN_SCALE;
  return 0; // CPU is bounded by f64; no hard cap here.
}

// Clamp `scale` to whatever the *currently selected* backend can render
// without falling apart. Caller should compute scale unconditionally and
// then pass it through here, so a thwarted zoom just stops at the floor
// (no fallback to CPU).
function clampScale(scale) {
  return Math.max(scale, minScaleForBackend(activeBackend()));
}
function syncCanvasVisibility(backend) {
  canvas.style.display      = backend === "cpu"    ? "" : "none";
  gpuCanvas.style.display   = backend === "webgl"  ? "" : "none";
  webgpuCanvas.style.display = backend === "webgpu" ? "" : "none";
}

const DEFAULT_VIEW = { cx: -0.5, cy: 0.0, scale: 1.25 };
let view = { ...DEFAULT_VIEW };
let ready = false;
let drawScheduled = false;
let refineTimer = null;

// Render-resolution caps. Interactive draws use the low cap so dragging /
// wheel zoom stay smooth on 4K; we then refine to full device resolution on idle.
const MAX_PIXELS_HIGH = Infinity; // full devicePixelRatio on idle
const MAX_PIXELS_LOW  = 2_500_000; // CPU-preview cap (GPU paths ignore this)
const LOW_ITER_CAP    = 256;       // cheaper iteration budget while interacting (CPU only)

function targetPixelSize(maxPixels) {
  const main = canvas.parentElement;
  const cssW = main.clientWidth;
  const cssH = main.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  let w = Math.max(1, Math.floor(cssW * dpr));
  let h = Math.max(1, Math.floor(cssH * dpr));
  if (Number.isFinite(maxPixels)) {
    const total = w * h;
    if (total > maxPixels) {
      const k = Math.sqrt(maxPixels / total);
      w = Math.max(1, Math.floor(w * k));
      h = Math.max(1, Math.floor(h * k));
    }
  }
  return { w, h, cssW, cssH, dpr };
}

function resizeAxes() {
  const main = canvas.parentElement;
  const cssW = main.clientWidth;
  const cssH = main.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  axesCanvas.width = Math.max(1, Math.floor(cssW * dpr));
  axesCanvas.height = Math.max(1, Math.floor(cssH * dpr));
  axesCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function scheduleDraw(quality = "high") {
  if (drawScheduled) {
    // If a higher-quality pass is requested, upgrade the queued frame.
    if (quality === "high") drawScheduled = "high";
    return;
  }
  drawScheduled = quality;
  requestAnimationFrame(() => {
    const q = drawScheduled;
    drawScheduled = false;
    draw(q);
  });
}

// Re-request a high-res render after the user stops interacting.
function scheduleRefine() {
  if (refineTimer) clearTimeout(refineTimer);
  refineTimer = setTimeout(() => {
    refineTimer = null;
    scheduleDraw("high");
  }, 140);
}

// Cached reference orbit for the WebGPU perturbation renderer. We reuse
// it as long as the chosen reference point is still inside the current
// view; recomputing every frame would otherwise let the 9x9 grid search
// pick a different winner each time the view scrolled, making the image
// flicker darker mid-zoom (more pixels falling through the f32 fallback)
// and brighten again once motion settled on a stable choice.
//
// We store `absCx`/`absCy` as f64 so that on every frame we can compute
// `ref_off = absRef - view_centre` in f64 and only cast the (small) result
// to f32 — essential beyond ~scale 1e-7, where naively reading the f32
// reference coords back from the orbit buffer would jitter the perturbation
// centre by orders of magnitude more than a pixel.
let cachedRef = null; // { absCx, absCy, triedMaxIter, orbit }

function getReferenceOrbit(cx, cy, scale, aspect, maxIter) {
  const halfX = scale * aspect;
  const halfY = scale;
  const inView = cachedRef
    && cachedRef.absCx >= cx - halfX && cachedRef.absCx <= cx + halfX
    && cachedRef.absCy >= cy - halfY && cachedRef.absCy <= cy + halfY;
  // Reuse if the reference is still on screen *and* we've already searched
  // for the current iteration budget (or higher). If maxIter went up a
  // second search may turn up a longer reference, so it's worth one retry.
  if (!(inView && cachedRef.triedMaxIter >= maxIter)) {
    const orbit = reference_orbit(cx, cy, scale, aspect, maxIter);
    // Rust returns ref_off in the header (computed in f64 then cast). Add
    // it to the *current* view centre in f64 to recover the absolute
    // reference point; that's what we cache.
    cachedRef = {
      absCx: cx + orbit[0],
      absCy: cy + orbit[1],
      triedMaxIter: maxIter,
      orbit,
    };
  }
  return cachedRef;
}

function draw(quality = "high") {
  if (!ready) return;
  const backend = activeBackend();
  syncCanvasVisibility(backend);

  // GPU renders are cheap — always go full-resolution.
  const isGpu = backend === "webgl" || backend === "webgpu";
  const cap = isGpu
    ? MAX_PIXELS_HIGH
    : (quality === "high" ? MAX_PIXELS_HIGH : MAX_PIXELS_LOW);
  const { w, h } = targetPixelSize(cap);

  const userIter = Math.max(32, Math.min(10000, parseInt(iterInput.value, 10) || 512));
  const maxIter = (isGpu || quality === "high")
    ? userIter : Math.min(userIter, LOW_ITER_CAP);

  const t0 = performance.now();
  if (backend === "webgl") {
    webgl.render(w, h, view.cx, view.cy, view.scale, maxIter);
  } else if (backend === "webgpu") {
    // Compute reference orbit on the CPU in f64; the GPU iterates deltas
    // around it in f32 (perturbation theory). The reference is *cached*
    // and reused while it remains inside the current view and long enough,
    // because re-running the 9x9 grid search every frame would otherwise
    // pick different winners as the view scrolled — making the image flicker
    // darker mid-zoom (more pixels falling through the f32 fallback path)
    // and brighten back up once motion settled on a stable choice.
    const aspect = w / h;
    const ref = getReferenceOrbit(view.cx, view.cy, view.scale, aspect, maxIter);
    // Compute ref_off in f64 from the *current* view centre and the cached
    // absolute reference; the GPU only ever sees the (small) f32 result.
    const refOffX = ref.absCx - view.cx;
    const refOffY = ref.absCy - view.cy;
    webgpu.render(w, h, view.cx, view.cy, view.scale, maxIter, ref.orbit, refOffX, refOffY);
  } else {
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const pixels = render_mandelbrot(w, h, view.cx, view.cy, view.scale, maxIter);
    const img = new ImageData(new Uint8ClampedArray(pixels), w, h);
    ctx.putImageData(img, 0, 0);
  }
  const dt = performance.now() - t0;

  drawAxes();

  let tag = backend;
  if (backend === "webgpu" && webgpu && webgpu.lastPrecision) tag = `webgpu/${webgpu.lastPrecision()}`;
  if (backend === "cpu" && quality === "low") tag = "cpu preview";
  status.textContent =
    `${w}×${h} · ${tag} · iter=${maxIter} · scale=${view.scale.toExponential(2)} · ${dt.toFixed(1)} ms`;
}

function activeCanvas() {
  const b = activeBackend();
  if (b === "webgl") return gpuCanvas;
  if (b === "webgpu") return webgpuCanvas;
  return canvas;
}

function viewBounds() {
  const c = activeCanvas();
  const aspect = (c.width || 1) / (c.height || 1);
  return {
    xMin: view.cx - view.scale * aspect,
    xMax: view.cx + view.scale * aspect,
    yMin: view.cy - view.scale,
    yMax: view.cy + view.scale,
  };
}

// Map a CSS-pixel point on the surface to a complex-plane coordinate.
function pixelToComplex(px, py) {
  const rect = surface.getBoundingClientRect();
  const { xMin, xMax, yMin, yMax } = viewBounds();
  return {
    x: xMin + (px / rect.width) * (xMax - xMin),
    y: yMin + (py / rect.height) * (yMax - yMin),
  };
}

// "Nice" 1/2/5 × 10^k tick spacing.
function niceStep(range, target) {
  const raw = range / target;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / pow;
  const step = (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * pow;
  return step;
}

function formatTick(v, step) {
  if (Math.abs(v) < step * 0.5) return "0";
  const decimals = Math.max(0, -Math.floor(Math.log10(step)));
  if (decimals > 6 || Math.abs(v) >= 1e5) return v.toExponential(2);
  return v.toFixed(Math.min(10, decimals));
}

function drawAxes() {
  const cssW = axesCanvas.clientWidth;
  const cssH = axesCanvas.clientHeight;
  axesCtx.clearRect(0, 0, cssW, cssH);
  if (!showAxes.checked) return;

  const { xMin, xMax, yMin, yMax } = viewBounds();
  const xRange = xMax - xMin;
  const yRange = yMax - yMin;
  const xToPx = (x) => ((x - xMin) / xRange) * cssW;
  const yToPx = (y) => ((y - yMin) / yRange) * cssH;

  const xStep = niceStep(xRange, 10);
  const yStep = niceStep(yRange, 8);

  axesCtx.lineWidth = 1;
  axesCtx.font = "11px ui-monospace, Menlo, Consolas, monospace";
  axesCtx.textBaseline = "top";

  // Grid lines.
  axesCtx.strokeStyle = "rgba(255,255,255,0.08)";
  axesCtx.beginPath();
  const x0 = Math.ceil(xMin / xStep) * xStep;
  for (let x = x0; x <= xMax; x += xStep) {
    const px = Math.round(xToPx(x)) + 0.5;
    axesCtx.moveTo(px, 0); axesCtx.lineTo(px, cssH);
  }
  const y0 = Math.ceil(yMin / yStep) * yStep;
  for (let y = y0; y <= yMax; y += yStep) {
    const py = Math.round(yToPx(y)) + 0.5;
    axesCtx.moveTo(0, py); axesCtx.lineTo(cssW, py);
  }
  axesCtx.stroke();

  // Real / imaginary axes (when in view).
  axesCtx.strokeStyle = "rgba(255,255,255,0.55)";
  axesCtx.beginPath();
  if (yMin <= 0 && 0 <= yMax) {
    const py = Math.round(yToPx(0)) + 0.5;
    axesCtx.moveTo(0, py); axesCtx.lineTo(cssW, py);
  }
  if (xMin <= 0 && 0 <= xMax) {
    const px = Math.round(xToPx(0)) + 0.5;
    axesCtx.moveTo(px, 0); axesCtx.lineTo(px, cssH);
  }
  axesCtx.stroke();

  // Tick labels: real along bottom, imaginary along left.
  axesCtx.fillStyle = "rgba(255,255,255,0.85)";
  axesCtx.shadowColor = "rgba(0,0,0,0.9)";
  axesCtx.shadowBlur = 3;

  axesCtx.textAlign = "center";
  for (let x = x0; x <= xMax; x += xStep) {
    axesCtx.fillText(formatTick(x, xStep), xToPx(x), cssH - 14);
  }
  axesCtx.textAlign = "left";
  for (let y = y0; y <= yMax; y += yStep) {
    axesCtx.fillText(formatTick(y, yStep) + "i", 4, yToPx(y) + 2);
  }
  axesCtx.shadowBlur = 0;
}

// --- Pointer handling: 1-finger / mouse = pan + click-to-zoom,
//     2-finger = pinch-to-zoom-and-pan. Works on iOS, Android, trackpad.
const pointers = new Map(); // id -> {x, y}
let panAnchor = null;       // {clientX, clientY, viewCx, viewCy} for 1-finger pan
let pinchAnchor = null;     // {dist, midX, midY, viewCx, viewCy, scale} for 2-finger
let dragMoved = false;

function pointerCenter() {
  let sx = 0, sy = 0;
  for (const p of pointers.values()) { sx += p.x; sy += p.y; }
  const n = pointers.size;
  return { x: sx / n, y: sy / n };
}
function pointerDistance() {
  if (pointers.size < 2) return 0;
  const it = pointers.values();
  const a = it.next().value;
  const b = it.next().value;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

surface.addEventListener("pointerdown", (e) => {
  surface.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  dragMoved = false;
  reanchor();
});

// Re-snapshot the gesture state from the *current* pointers + view, so each
// new finger down/up doesn't cause a jump.
function reanchor() {
  panAnchor = null;
  pinchAnchor = null;
  if (pointers.size === 1) {
    const only = pointers.values().next().value;
    panAnchor = { clientX: only.x, clientY: only.y,
                  viewCx: view.cx, viewCy: view.cy };
  } else if (pointers.size >= 2) {
    pinchAnchor = {
      dist: pointerDistance(),
      mid: pointerCenter(),
      viewCx: view.cx,
      viewCy: view.cy,
      scale: view.scale,
    };
  }
}

surface.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId)) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

  const rect = surface.getBoundingClientRect();
  const c = activeCanvas();
  const aspect = (c.width || 1) / (c.height || 1);

  if (pointers.size === 1 && panAnchor) {
    const dxPx = e.clientX - panAnchor.clientX;
    const dyPx = e.clientY - panAnchor.clientY;
    if (Math.hypot(dxPx, dyPx) > 3) dragMoved = true;
    const dx = dxPx / rect.width  * 2 * view.scale * aspect;
    const dy = dyPx / rect.height * 2 * view.scale;
    view.cx = panAnchor.viewCx - dx;
    view.cy = panAnchor.viewCy - dy;
    scheduleDraw("low");
    scheduleRefine();
  } else if (pointers.size >= 2 && pinchAnchor) {
    dragMoved = true;
    const newDist = pointerDistance();
    if (newDist <= 0) return;

    // The point in complex space that was under the original pinch midpoint.
    // We pin it to wherever the current pinch midpoint is now, while scaling
    // the view by (originalDist / newDist).
    const a = pinchAnchor;
    const ax = (a.mid.x - rect.left) / rect.width  * 2 - 1; // [-1, 1]
    const ay = (a.mid.y - rect.top)  / rect.height * 2 - 1;
    const anchorCx = a.viewCx + ax * a.scale * aspect;
    const anchorCy = a.viewCy + ay * a.scale;

    const newScale = clampScale(a.scale * (a.dist / newDist));
    const cMid = pointerCenter();
    const nx = (cMid.x - rect.left) / rect.width  * 2 - 1;
    const ny = (cMid.y - rect.top)  / rect.height * 2 - 1;
    view.scale = newScale;
    view.cx = anchorCx - nx * newScale * aspect;
    view.cy = anchorCy - ny * newScale;
    scheduleDraw("low");
    scheduleRefine();
  }
});

const releasePointer = (e) => {
  pointers.delete(e.pointerId);
  if (pointers.size === 0) {
    panAnchor = null;
    pinchAnchor = null;
    scheduleRefine();
  } else {
    // Adding or removing a finger mid-gesture: re-snapshot so the next
    // move is relative to the current view, not the stale anchor.
    reanchor();
  }
};
surface.addEventListener("pointerup", releasePointer);
surface.addEventListener("pointercancel", releasePointer);

surface.addEventListener("click", (e) => {
  if (dragMoved) return; // don't zoom after a pan / pinch
  // Suppress click-zoom on touch devices — they have pinch instead, and a
  // tap-to-zoom would fight with double-tap behaviours.
  if (e.pointerType === "touch") return;
  const rect = surface.getBoundingClientRect();
  const p = pixelToComplex(e.clientX - rect.left, e.clientY - rect.top);
  view.cx = p.x;
  view.cy = p.y;
  const factor = e.shiftKey ? 2.5 : 0.4;
  view.scale = clampScale(view.scale * factor);
  scheduleDraw("high");
});

surface.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = surface.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  const before = pixelToComplex(px, py);
  const factor = Math.exp(e.deltaY * 0.0015);
  view.scale = clampScale(view.scale * factor);
  const after = pixelToComplex(px, py);
  view.cx += before.x - after.x;
  view.cy += before.y - after.y;
  scheduleDraw("low");
  scheduleRefine();
}, { passive: false });

iterInput.addEventListener("change", () => {
  // Sync the mobile slider whenever the number field changes (and clamp to
  // its range so dragging the slider afterwards doesn't snap unexpectedly).
  const v = Math.max(32, Math.min(10000, parseInt(iterInput.value, 10) || 512));
  iterRange.value = String(Math.min(parseInt(iterRange.max, 10), v));
  iterValue.textContent = String(v);
  scheduleDraw("high");
});
// Slider: live preview while dragging (low quality), commit on release.
iterRange.addEventListener("input", () => {
  iterInput.value = iterRange.value;
  iterValue.textContent = iterRange.value;
  scheduleDraw("low");
});
iterRange.addEventListener("change", () => scheduleDraw("high"));
showAxes.addEventListener("change", drawAxes);
rendererSelect.addEventListener("change", () => scheduleDraw("high"));
resetBtn.addEventListener("click", () => { view = { ...DEFAULT_VIEW }; scheduleDraw("high"); });

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { resizeAxes(); scheduleDraw("high"); }, 80);
});

async function boot() {
  // Initialize WASM and WebGPU in parallel.
  const [, wgpu] = await Promise.all([
    init(),
    createWebGpuRenderer(webgpuCanvas).catch((e) => {
      console.warn("WebGPU init failed:", e);
      return null;
    }),
  ]);
  webgpu = wgpu;

  // Disable unsupported renderer options.
  for (const opt of rendererSelect.options) {
    if (opt.value === "webgpu" && !webgpu) {
      opt.disabled = true;
      opt.textContent += " \u2014 unavailable";
    }
    if (opt.value === "webgl" && !webgl) {
      opt.disabled = true;
      opt.textContent += " \u2014 unavailable";
    }
  }
  // Default: WebGL (fast and good enough for most zooms). The HTML already
  // marks it as the selected option; only override if WebGL is unavailable.
  if (!webgl) rendererSelect.value = webgpu ? "webgpu" : "cpu";

  ready = true;
  resizeAxes();
  draw("high");
}

boot().catch((err) => {
  status.textContent = "failed to load wasm — see console";
  console.error(err);
});
