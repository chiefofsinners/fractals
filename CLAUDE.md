# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```sh
# Build the WASM package into web/pkg/ (required before serving).
wasm-pack build --target web --release --out-dir web/pkg

# Serve the static site (the page loads pkg/fractals.js as an ES module
# and will not work over file://).
python3 -m http.server -d web 8080
# or: npx http-server web -p 8080
```

There are no tests, no linter config, and no JS toolchain — `web/` is plain
ES modules served as-is. Deployment is automated: pushes to `main` trigger
`.github/workflows/deploy.yml`, which runs the same `wasm-pack build` and
publishes `web/` to GitHub Pages.

## Architecture

Three rendering backends share one viewport model and one palette. The user
picks between them at runtime (`#renderer` select), and `web/main.js` is the
orchestrator that owns view state and dispatches to whichever backend is
active.

### Rust crate (`src/lib.rs`)

A `cdylib` exposed via `wasm-bindgen`. Three families of functions:

- **`render_*`** (`render_mandelbrot`, `render_julia`, `render_burning_ship`):
  full f64 CPU renderer that returns an `RGBA Vec<u8>` for `ImageData`.
- **`*_reference`** (`reference_orbit`, `julia_reference`, `burning_ship_reference`):
  produce a high-precision reference orbit consumed by the WebGPU
  perturbation shader. Buffer layout is documented inline:
  `[ref_off_x, ref_off_y, zx0, zy0, zx1, zy1, ...]` as f32, where
  `ref_off` is `(chosen_ref - view_centre)` computed in f64 then cast to
  f32 — the offset is always small (≤ scale), so f32 is sufficient even
  at deep zoom. Each function probes a 9×9 grid inside the view and picks
  the longest-lived starting point as the reference, preferring orbits
  that hit `max_iter` (i.e. land in the set).

The `palette` is a cosine-based "IQ palette" duplicated verbatim across
Rust, WGSL, and GLSL — keep them in sync if you change colours. Same for
the `0.5`-gamma compression applied to Julia and Burning Ship `t` values
(but **not** Mandelbrot).

### Backends and their precision regimes

`web/main.js` defines per-backend zoom floors and refuses to zoom past them
rather than silently falling back:

- **CPU (WASM, f64)**: no zoom floor. Slowest; used as fallback and for
  preview at low resolution while interacting (`MAX_PIXELS_LOW`,
  `LOW_ITER_CAP`).
- **WebGL2 (`web/gpu.js`, f32)**: a fragment shader does the iteration
  directly. Floor at `WEBGL_MIN_SCALE = 1e-5`; below that, f32 quantisation
  produces visible blockiness.
- **WebGPU (`web/webgpu.js`, perturbation theory)**: CPU computes a single
  reference orbit `Z_n` in f64; the GPU iterates a per-pixel delta `dz`
  using `dz_{n+1} = 2·Z_n·dz_n + dz_n² + dc`. Bailout uses the *true*
  orbit `Z + dz` so colouring matches the CPU renderer. Floor at
  `WEBGPU_MIN_SCALE = 1e-15`. Burning Ship has a custom perturbation rule
  with a sign-flip correction term to handle pixels straddling the axes.
  Includes 2×2 ordered-grid supersampling.

The WebGPU path **caches** the reference orbit (`cachedRef` in `main.js`)
keyed on fractal type / Julia c. It's invalidated when the cached point
falls outside the current view. Without this, the 9×9 grid search would
pick a different winner per frame during pans, causing flicker. Crucially,
the cache stores `absCx`/`absCy` as f64 so each frame can recompute
`ref_off = absRef - view_centre` in f64 — past ~scale 1e-7, reading the
baked f32 `ref_off` from the orbit header would jitter the perturbation
centre by orders of magnitude more than a pixel.

### Coordinate convention

All renderers use the same viewport: `cx`/`cy` are the centre in the
complex plane, `scale` is the half-height. The width is `scale * aspect`.
**Increasing screen-y maps to increasing complex-y** in every backend
(WebGL flips `gl_FragCoord.y` to match). Burning Ship's canonical "ship"
silhouette therefore appears upright when centred near `(-0.5, -0.5)`
(see `DEFAULT_VIEWS` in `main.js`).

### Interactive vs. idle rendering

`scheduleDraw(quality)` and `scheduleRefine()` implement a two-tier render
loop. While the user drags / wheels / pinches, the CPU backend renders at
`MAX_PIXELS_LOW` with `LOW_ITER_CAP`; on idle (140ms) it re-renders at full
device resolution and the user-set iteration cap. GPU backends ignore the
preview cap (they're cheap enough to always render full-res).

### Adding a new fractal type

The fractal index is shared across `currentFractal()` in `main.js`,
`u_mode` in `web/gpu.js`, and `mode` in `web/webgpu.js`:
`0 = Mandelbrot, 1 = Julia, 2 = Burning Ship`. Adding a new type means
touching all four code paths (Rust render fn, Rust reference fn, GLSL
fragment shader, WGSL shader) plus the `<select id="fractal">` options in
`web/index.html` and the `DEFAULT_VIEWS` entry in `main.js`.
