import init, { render_mandelbrot } from "./pkg/fractals.js";
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
const showAxes = document.getElementById("showAxes");
const rendererSelect = document.getElementById("renderer");
const resetBtn = document.getElementById("reset");
const status = document.getElementById("status");

const webgl = createGpuRenderer(gpuCanvas);
let webgpu = null; // populated asynchronously by boot()

// Below this scale, f32 precision in the WebGL shader breaks down.
const WEBGL_MIN_SCALE = 1e-5;
// df64 ~= 48 bits of mantissa; below this scale, df64 also breaks down.
const WEBGPU_MIN_SCALE = 1e-13;

function activeBackend() {
  const want = rendererSelect.value;
  if (want === "webgpu") {
    if (webgpu && view.scale > WEBGPU_MIN_SCALE) return "webgpu";
    if (webgl && view.scale > WEBGL_MIN_SCALE) return "webgl";
    return "cpu";
  }
  if (want === "webgl") {
    if (webgl && view.scale > WEBGL_MIN_SCALE) return "webgl";
    return "cpu";
  }
  return "cpu";
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
    webgpu.render(w, h, view.cx, view.cy, view.scale, maxIter);
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

let dragging = null;
let dragMoved = false;

surface.addEventListener("pointerdown", (e) => {
  dragging = { x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy };
  dragMoved = false;
  surface.setPointerCapture(e.pointerId);
});
surface.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  if (Math.hypot(e.clientX - dragging.x, e.clientY - dragging.y) > 3) dragMoved = true;
  const rect = surface.getBoundingClientRect();
  const c = activeCanvas();
  const aspect = (c.width || 1) / (c.height || 1);
  const dx = (e.clientX - dragging.x) / rect.width * 2 * view.scale * aspect;
  const dy = (e.clientY - dragging.y) / rect.height * 2 * view.scale;
  view.cx = dragging.cx - dx;
  view.cy = dragging.cy - dy;
  scheduleDraw("low");
  scheduleRefine();
});
const endDrag = () => {
  if (dragging) scheduleRefine();
  dragging = null;
};
surface.addEventListener("pointerup", endDrag);
surface.addEventListener("pointercancel", endDrag);

surface.addEventListener("click", (e) => {
  if (dragMoved) return; // don't zoom after a pan
  const rect = surface.getBoundingClientRect();
  const p = pixelToComplex(e.clientX - rect.left, e.clientY - rect.top);
  view.cx = p.x;
  view.cy = p.y;
  view.scale *= e.shiftKey ? 2.5 : 0.4;
  scheduleDraw("high");
});

surface.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = surface.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  const before = pixelToComplex(px, py);
  const factor = Math.exp(e.deltaY * 0.0015);
  view.scale *= factor;
  const after = pixelToComplex(px, py);
  view.cx += before.x - after.x;
  view.cy += before.y - after.y;
  scheduleDraw("low");
  scheduleRefine();
}, { passive: false });

iterInput.addEventListener("change", () => scheduleDraw("high"));
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
  // Default: prefer WebGPU when available, else WebGL, else CPU.
  if (webgpu) rendererSelect.value = "webgpu";
  else if (webgl) rendererSelect.value = "webgl";
  else rendererSelect.value = "cpu";

  ready = true;
  resizeAxes();
  draw("high");
}

boot().catch((err) => {
  status.textContent = "failed to load wasm — see console";
  console.error(err);
});
