use wasm_bindgen::prelude::*;

/// Render the Mandelbrot set into an RGBA pixel buffer.
///
/// The viewport is described in complex-plane coordinates: the rectangle
/// [center_x - scale * aspect, center_x + scale * aspect] x
/// [center_y - scale,          center_y + scale].
///
/// Returns a `Vec<u8>` of length `width * height * 4` (RGBA, row-major,
/// top-to-bottom) suitable for `ImageData` on an HTML canvas.
#[wasm_bindgen]
pub fn render_mandelbrot(
    width: u32,
    height: u32,
    center_x: f64,
    center_y: f64,
    scale: f64,
    max_iter: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut buf = vec![0u8; w * h * 4];

    let aspect = width as f64 / height as f64;
    let x_min = center_x - scale * aspect;
    let x_max = center_x + scale * aspect;
    let y_min = center_y - scale;
    let y_max = center_y + scale;

    let dx = (x_max - x_min) / width as f64;
    let dy = (y_max - y_min) / height as f64;

    for py in 0..h {
        let cy = y_min + py as f64 * dy;
        for px in 0..w {
            let cx = x_min + px as f64 * dx;
            let (iter, zx, zy) = mandelbrot_iter(cx, cy, max_iter);

            let idx = (py * w + px) * 4;
            if iter >= max_iter {
                // Inside the set: black.
                buf[idx] = 0;
                buf[idx + 1] = 0;
                buf[idx + 2] = 0;
            } else {
                // Smooth (continuous) coloring.
                let log_zn = (zx * zx + zy * zy).ln() * 0.5;
                let nu = (log_zn / std::f64::consts::LN_2).log2();
                let smooth = iter as f64 + 1.0 - nu;
                let t = colour_t(smooth, max_iter);
                let (r, g, b) = palette(t);
                buf[idx] = r;
                buf[idx + 1] = g;
                buf[idx + 2] = b;
            }
            buf[idx + 3] = 255;
        }
    }

    buf
}

/// Render a Julia set z_{n+1} = z_n^2 + c (c fixed) into RGBA. Same
/// viewport convention as `render_mandelbrot`; `jcx`/`jcy` is the
/// constant complex parameter c.
#[wasm_bindgen]
pub fn render_julia(
    width: u32,
    height: u32,
    center_x: f64,
    center_y: f64,
    scale: f64,
    jcx: f64,
    jcy: f64,
    max_iter: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut buf = vec![0u8; w * h * 4];

    let aspect = width as f64 / height as f64;
    let x_min = center_x - scale * aspect;
    let y_min = center_y - scale;
    let dx = (scale * aspect * 2.0) / width as f64;
    let dy = (scale * 2.0) / height as f64;

    for py in 0..h {
        let y0 = y_min + py as f64 * dy;
        for px in 0..w {
            let x0 = x_min + px as f64 * dx;
            let (iter, zx, zy) = julia_iter(x0, y0, jcx, jcy, max_iter);

            let idx = (py * w + px) * 4;
            if iter >= max_iter {
                buf[idx] = 0; buf[idx + 1] = 0; buf[idx + 2] = 0;
            } else {
                let log_zn = (zx * zx + zy * zy).ln() * 0.5;
                let nu = (log_zn / std::f64::consts::LN_2).log2();
                let smooth = iter as f64 + 1.0 - nu;
                let t = colour_t(smooth, max_iter);
                let (r, g, b) = palette(t);
                buf[idx] = r; buf[idx + 1] = g; buf[idx + 2] = b;
            }
            buf[idx + 3] = 255;
        }
    }
    buf
}

/// Render the Burning Ship fractal into RGBA. Same viewport convention
/// as `render_mandelbrot`. Recurrence:
///
///     z_{n+1} = (|Re z_n| + i |Im z_n|)^2 + c,   z_0 = 0
///
/// Note that with our screen convention (increasing screen-y maps to
/// increasing complex-y) the canonical "ship" silhouette appears upright
/// when centred near (-0.5, -0.5).
#[wasm_bindgen]
pub fn render_burning_ship(
    width: u32,
    height: u32,
    center_x: f64,
    center_y: f64,
    scale: f64,
    max_iter: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut buf = vec![0u8; w * h * 4];

    let aspect = width as f64 / height as f64;
    let x_min = center_x - scale * aspect;
    let y_min = center_y - scale;
    let dx = (scale * aspect * 2.0) / width as f64;
    let dy = (scale * 2.0) / height as f64;

    for py in 0..h {
        let cy = y_min + py as f64 * dy;
        for px in 0..w {
            let cx = x_min + px as f64 * dx;
            let (iter, zx, zy) = burning_ship_iter(cx, cy, max_iter);

            let idx = (py * w + px) * 4;
            if iter >= max_iter {
                buf[idx] = 0; buf[idx + 1] = 0; buf[idx + 2] = 0;
            } else {
                let log_zn = (zx * zx + zy * zy).ln() * 0.5;
                let nu = (log_zn / std::f64::consts::LN_2).log2();
                let smooth = iter as f64 + 1.0 - nu;
                let t = colour_t(smooth, max_iter);
                let (r, g, b) = palette(t);
                buf[idx] = r; buf[idx + 1] = g; buf[idx + 2] = b;
            }
            buf[idx + 3] = 255;
        }
    }
    buf
}

#[inline]
fn burning_ship_iter(cx: f64, cy: f64, max_iter: u32) -> (u32, f64, f64) {
    let mut zx = 0.0f64;
    let mut zy = 0.0f64;
    let bailout = (1u64 << 16) as f64;
    let mut i = 0u32;
    while i < max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > bailout { return (i, zx, zy); }
        // Folded recurrence: take absolute values before the square.
        let nzy = 2.0 * zx.abs() * zy.abs() + cy;
        zx = zx2 - zy2 + cx;
        zy = nzy;
        i += 1;
    }
    (max_iter, zx, zy)
}

#[inline]
fn julia_iter(mut zx: f64, mut zy: f64, cx: f64, cy: f64, max_iter: u32) -> (u32, f64, f64) {
    let bailout = (1u64 << 16) as f64;
    let mut i = 0u32;
    while i < max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > bailout { return (i, zx, zy); }
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        i += 1;
    }
    (max_iter, zx, zy)
}

#[inline]
fn mandelbrot_iter(cx: f64, cy: f64, max_iter: u32) -> (u32, f64, f64) {
    // Cardioid / period-2 bulb early exit.
    let xm = cx - 0.25;
    let q = xm * xm + cy * cy;
    if q * (q + xm) <= 0.25 * cy * cy {
        return (max_iter, 0.0, 0.0);
    }
    let xp1 = cx + 1.0;
    if xp1 * xp1 + cy * cy <= 0.0625 {
        return (max_iter, 0.0, 0.0);
    }

    let mut zx = 0.0f64;
    let mut zy = 0.0f64;
    let bailout = 1u64 << 16; // large bailout for smoother coloring
    let bailout = bailout as f64;

    let mut i = 0u32;
    while i < max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > bailout {
            return (i, zx, zy);
        }
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        i += 1;
    }
    (max_iter, zx, zy)
}

/// Compute a high-precision reference orbit for perturbation-theory
/// rendering on the GPU. Returns header `[ref_off_x, ref_off_y]` (the
/// chosen reference point *expressed as an offset from the view centre*,
/// computed in f64 then cast to f32 — this is what the GPU needs and
/// preserves precision at deep zoom, where the absolute coords would lose
/// ~6e-8 to f32 quantisation) followed by `[zx0, zy0, zx1, zy1, ...]`,
/// all as f32. Iteration count = (len-2)/2.
///
/// We probe a grid of candidate centres inside the view and pick the
/// longest-lived orbit (preferring ones that hit `max_iter`, i.e. land
/// inside the set, since they make perfect perturbation references).
#[wasm_bindgen]
pub fn reference_orbit(cx: f64, cy: f64, scale: f64, aspect: f64, max_iter: u32) -> Vec<f32> {
    let mut best_cx = cx;
    let mut best_cy = cy;
    let mut best_n = 0u32;
    // 9x9 = 81 samples; cheap (~ms at max_iter=512) and finds in-set
    // references reliably for any view that contains some of the set.
    const GRID: i32 = 9;
    for gy in 0..GRID {
        for gx in 0..GRID {
            let ux = (gx as f64) / ((GRID - 1) as f64) * 2.0 - 1.0;
            let uy = (gy as f64) / ((GRID - 1) as f64) * 2.0 - 1.0;
            let pcx = cx + ux * scale * aspect;
            let pcy = cy + uy * scale;
            let n = orbit_length(pcx, pcy, max_iter);
            if n > best_n {
                best_n = n;
                best_cx = pcx;
                best_cy = pcy;
                if n >= max_iter { break; } // can't do better than full length
            }
        }
        if best_n >= max_iter { break; }
    }

    let mut out: Vec<f32> = Vec::with_capacity(ORBIT_HEADER_LEN + 2 * (max_iter as usize + 1));
    push_orbit_header(&mut out, best_cx - cx, best_cy - cy);

    let r2 = view_diag_sq(scale, aspect);
    let mut sa = SaTracker::mandelbrot();

    let mut zx = 0.0f64;
    let mut zy = 0.0f64;
    out.push(zx as f32);
    out.push(zy as f32);
    let bailout = (1u64 << 16) as f64;
    for n in 0..max_iter {
        sa.consider(n, r2);
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > bailout { break; }
        sa.advance(zx, zy);
        let nzy = 2.0 * zx * zy + best_cy;
        zx = zx2 - zy2 + best_cx;
        zy = nzy;
        out.push(zx as f32);
        out.push(zy as f32);
    }
    sa.finalize_into(&mut out);
    out
}

/// Julia reference orbit: same buffer layout as `reference_orbit`, but the
/// recurrence is z_{n+1} = z_n^2 + c with c fixed (= jcx, jcy) and z_0 set
/// to the chosen reference *pixel*. We pick the reference by sampling a
/// 9x9 grid inside the view and taking the longest-lived starting point.
#[wasm_bindgen]
pub fn julia_reference(cx: f64, cy: f64, scale: f64, aspect: f64, jcx: f64, jcy: f64, max_iter: u32) -> Vec<f32> {
    let mut best_zx = cx;
    let mut best_zy = cy;
    let mut best_n = 0u32;
    const GRID: i32 = 9;
    for gy in 0..GRID {
        for gx in 0..GRID {
            let ux = (gx as f64) / ((GRID - 1) as f64) * 2.0 - 1.0;
            let uy = (gy as f64) / ((GRID - 1) as f64) * 2.0 - 1.0;
            let pzx = cx + ux * scale * aspect;
            let pzy = cy + uy * scale;
            let n = julia_orbit_length(pzx, pzy, jcx, jcy, max_iter);
            if n > best_n {
                best_n = n;
                best_zx = pzx;
                best_zy = pzy;
                if n >= max_iter { break; }
            }
        }
        if best_n >= max_iter { break; }
    }

    let mut out: Vec<f32> = Vec::with_capacity(ORBIT_HEADER_LEN + 2 * (max_iter as usize + 1));
    push_orbit_header(&mut out, best_zx - cx, best_zy - cy);

    let r2 = view_diag_sq(scale, aspect);
    let mut sa = SaTracker::julia();

    let mut zx = best_zx;
    let mut zy = best_zy;
    out.push(zx as f32);
    out.push(zy as f32);
    let bailout = (1u64 << 16) as f64;
    for n in 0..max_iter {
        sa.consider(n, r2);
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > bailout { break; }
        sa.advance(zx, zy);
        let nzy = 2.0 * zx * zy + jcy;
        zx = zx2 - zy2 + jcx;
        zy = nzy;
        out.push(zx as f32);
        out.push(zy as f32);
    }
    sa.finalize_into(&mut out);
    out
}

/// Burning Ship reference orbit for the GPU perturbation renderer. Same
/// buffer layout as `reference_orbit`: `[ref_off_x, ref_off_y, zx0, zy0, ...]`
/// as f32. Recurrence is the folded Burning Ship one,
///     z_{n+1} = (|Re z_n| + i |Im z_n|)^2 + c,
/// computed in f64 for the reference, with z_0 = 0 so Z_1 = c (matching the
/// Mandelbrot convention; the shader reads `ref_orbit[1]` as the reference c
/// when extending past the stored orbit).
#[wasm_bindgen]
pub fn burning_ship_reference(cx: f64, cy: f64, scale: f64, aspect: f64, max_iter: u32) -> Vec<f32> {
    let mut best_cx = cx;
    let mut best_cy = cy;
    let mut best_n = 0u32;
    const GRID: i32 = 9;
    for gy in 0..GRID {
        for gx in 0..GRID {
            let ux = (gx as f64) / ((GRID - 1) as f64) * 2.0 - 1.0;
            let uy = (gy as f64) / ((GRID - 1) as f64) * 2.0 - 1.0;
            let pcx = cx + ux * scale * aspect;
            let pcy = cy + uy * scale;
            let n = burning_ship_orbit_length(pcx, pcy, max_iter);
            if n > best_n {
                best_n = n;
                best_cx = pcx;
                best_cy = pcy;
                if n >= max_iter { break; }
            }
        }
        if best_n >= max_iter { break; }
    }

    let mut out: Vec<f32> = Vec::with_capacity(ORBIT_HEADER_LEN + 2 * (max_iter as usize + 1));
    // SA fields stay zero — the abs() fold breaks the polynomial recurrence,
    // so Burning Ship can't use series approximation. Shader sees sa_skip=0
    // and runs the un-skipped perturbation loop.
    push_orbit_header(&mut out, best_cx - cx, best_cy - cy);

    let mut zx = 0.0f64;
    let mut zy = 0.0f64;
    out.push(zx as f32);
    out.push(zy as f32);
    let bailout = (1u64 << 16) as f64;
    for _ in 0..max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > bailout { break; }
        let nzy = 2.0 * zx.abs() * zy.abs() + best_cy;
        zx = zx2 - zy2 + best_cx;
        zy = nzy;
        out.push(zx as f32);
        out.push(zy as f32);
    }
    out
}

#[inline]
fn burning_ship_orbit_length(cx: f64, cy: f64, max_iter: u32) -> u32 {
    let mut zx = 0.0f64;
    let mut zy = 0.0f64;
    let bailout = (1u64 << 16) as f64;
    for i in 0..max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > bailout { return i; }
        let nzy = 2.0 * zx.abs() * zy.abs() + cy;
        zx = zx2 - zy2 + cx;
        zy = nzy;
    }
    max_iter
}

#[inline]
fn julia_orbit_length(mut zx: f64, mut zy: f64, cx: f64, cy: f64, max_iter: u32) -> u32 {
    let bailout = (1u64 << 16) as f64;
    for i in 0..max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > bailout { return i; }
        let nzy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zy = nzy;
    }
    max_iter
}

#[inline]
fn orbit_length(cx: f64, cy: f64, max_iter: u32) -> u32 {
    let mut zx = 0.0f64;
    let mut zy = 0.0f64;
    let bailout = (1u64 << 16) as f64;
    for i in 0..max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > bailout { return i; }
        let nzy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zy = nzy;
    }
    max_iter
}

/// Map a smooth iteration count to `t ∈ [0, 1]` for palette lookup.
///
/// Logarithmic so the palette spans the full range regardless of
/// `max_iter`: linear `smooth/max_iter` bunches everything at the start
/// of the palette when the user cranks the iteration cap up, since most
/// pixels still escape in tens of iterations.
#[inline]
fn colour_t(smooth: f64, max_iter: u32) -> f64 {
    let s = smooth.max(0.0);
    (s + 1.0).ln() / ((max_iter as f64) + 1.0).ln()
}

/// Reference-orbit buffer header layout (f32 slots):
///   [0,1] ref_off    — chosen reference - view_centre, computed in f64
///   [2]   sa_skip    — series-approximation skip count (u32 reinterpreted via f32 bits not used; we store it as a plain f32 cast)
///   [3]   _pad       — reserved
///   [4,5] sa_a       — series coefficient A (linear in dpix)
///   [6,7] sa_b       — series coefficient B (quadratic in dpix)
///   [8,9] sa_c       — series coefficient C (cubic in dpix)
/// Followed by [zx0, zy0, zx1, zy1, ...]. The JS side passes sa_skip via
/// `(refOrbit[2] | 0)`, so the f32 value just needs to round-trip integer
/// counts up to ~16M, which is plenty (max_iter cap is 10000).
const ORBIT_HEADER_LEN: usize = 10;

#[inline]
fn push_orbit_header(out: &mut Vec<f32>, off_x: f64, off_y: f64) {
    out.push(off_x as f32);
    out.push(off_y as f32);
    // SA fields zeroed; SaTracker::finalize_into patches them after the orbit.
    for _ in 2..ORBIT_HEADER_LEN { out.push(0.0); }
}

#[inline]
fn view_diag_sq(scale: f64, aspect: f64) -> f64 {
    // Worst-case |dpix| is the full diagonal of the view rectangle: the
    // chosen reference might sit at one corner of the 9x9 sample grid and
    // the worst pixel at the opposite corner.
    let half_x = scale * aspect;
    let half_y = scale;
    (2.0 * half_x).powi(2) + (2.0 * half_y).powi(2)
}

/// Tracks the series-approximation polynomial dz_n ≈ A_n·dpix + B_n·dpix² + C_n·dpix³
/// alongside the reference orbit. Recurrence (complex arithmetic):
///     A_{n+1} = 2·Z_n·A_n + plus_one
///     B_{n+1} = 2·Z_n·B_n + A_n²
///     C_{n+1} = 2·Z_n·C_n + 2·A_n·B_n
/// where `plus_one` = 1 for Mandelbrot (the +dc term in the recurrence) and 0
/// for Julia (c is constant; +dc contributes only via the initial dz_0 = dpix).
///
/// Validity: at iteration `n`, we accept (A_n, B_n, C_n) as a usable skip if
/// (a) all coefficient magnitudes fit safely in f32 (well below the 3.4e38
/// limit, with headroom for the per-pixel polynomial multiply) and (b) the
/// cubic term remains a small fraction of the linear+quadratic term at the
/// worst-case |dpix| in the view. The largest accepted n becomes `sa_skip`.
struct SaTracker {
    ax: f64, ay: f64,
    bx: f64, by: f64,
    cx: f64, cy: f64,
    plus_one: f64,
    skip: u32,
    saved_a: (f64, f64),
    saved_b: (f64, f64),
    saved_c: (f64, f64),
    invalid: bool,
}

impl SaTracker {
    fn mandelbrot() -> Self {
        Self {
            ax: 0.0, ay: 0.0,
            bx: 0.0, by: 0.0,
            cx: 0.0, cy: 0.0,
            plus_one: 1.0,
            skip: 0,
            saved_a: (0.0, 0.0),
            saved_b: (0.0, 0.0),
            saved_c: (0.0, 0.0),
            invalid: false,
        }
    }

    fn julia() -> Self {
        // dz_0 = dpix, so A_0 = 1.
        Self {
            ax: 1.0, ay: 0.0,
            bx: 0.0, by: 0.0,
            cx: 0.0, cy: 0.0,
            plus_one: 0.0,
            skip: 0,
            saved_a: (1.0, 0.0),
            saved_b: (0.0, 0.0),
            saved_c: (0.0, 0.0),
            invalid: false,
        }
    }

    /// At iteration `n` (before stepping past Z_n), check whether the current
    /// (A_n, B_n, C_n) is a valid SA skip and record it if so. Once invalid,
    /// later iterations are not re-checked — coefficient magnitudes grow
    /// monotonically once the orbit's chaotic phase begins.
    fn consider(&mut self, n: u32, r2: f64) {
        if self.invalid { return; }
        let am2 = self.ax * self.ax + self.ay * self.ay;
        let bm2 = self.bx * self.bx + self.by * self.by;
        let cm2 = self.cx * self.cx + self.cy * self.cy;
        // Hard cap: coefficient magnitudes must round-trip through f32. 1e30
        // leaves ~1e8 headroom before f32 max (3.4e38) for per-pixel evaluation.
        const COEF_MAX_SQ: f64 = 1.0e60;
        if am2 > COEF_MAX_SQ || bm2 > COEF_MAX_SQ || cm2 > COEF_MAX_SQ {
            self.invalid = true;
            return;
        }
        // Cubic term must be a small fraction of the polynomial total at
        // worst-case |dpix|. Compare squared magnitudes to skip a sqrt.
        // (cm² · r²³) vs tol² · max(am² · r², bm² · r²²) — algebraically:
        //   |C·R³|² < tol² · max(|A·R|², |B·R²|²)
        let cubic_sq = cm2 * r2 * r2 * r2;
        let lin_sq = am2 * r2;
        let quad_sq = bm2 * r2 * r2;
        const TOL_SQ: f64 = 1.0e-6; // tol = 1e-3
        // Below n=4 the polynomial is degenerate (B/C still 0) — accept any
        // small step; the coarse-skip ones get filtered out by the >=8 cutoff
        // in finalize_into.
        if n >= 4 && cubic_sq > TOL_SQ * lin_sq.max(quad_sq).max(r2 * r2) {
            self.invalid = true;
            return;
        }
        self.skip = n;
        self.saved_a = (self.ax, self.ay);
        self.saved_b = (self.bx, self.by);
        self.saved_c = (self.cx, self.cy);
    }

    /// Step (A_n, B_n, C_n) -> (A_{n+1}, B_{n+1}, C_{n+1}) using Z_n.
    fn advance(&mut self, zx: f64, zy: f64) {
        if self.invalid { return; }
        let twozx = 2.0 * zx;
        let twozy = 2.0 * zy;
        // 2Z·A
        let na_x = twozx * self.ax - twozy * self.ay + self.plus_one;
        let na_y = twozx * self.ay + twozy * self.ax;
        // A²
        let a2_x = self.ax * self.ax - self.ay * self.ay;
        let a2_y = 2.0 * self.ax * self.ay;
        // 2Z·B + A²
        let nb_x = twozx * self.bx - twozy * self.by + a2_x;
        let nb_y = twozx * self.by + twozy * self.bx + a2_y;
        // 2·A·B
        let ab_x = self.ax * self.bx - self.ay * self.by;
        let ab_y = self.ax * self.by + self.ay * self.bx;
        // 2Z·C + 2·A·B
        let nc_x = twozx * self.cx - twozy * self.cy + 2.0 * ab_x;
        let nc_y = twozx * self.cy + twozy * self.cx + 2.0 * ab_y;
        self.ax = na_x; self.ay = na_y;
        self.bx = nb_x; self.by = nb_y;
        self.cx = nc_x; self.cy = nc_y;
    }

    /// Patch SA fields into the orbit header. If skip < 8 the saved-up cost
    /// (per-pixel cubic evaluation) likely outweighs the saved iterations,
    /// so we write skip=0 and let the shader take the un-skipped path.
    fn finalize_into(&self, out: &mut [f32]) {
        let (skip, a, b, c) = if self.skip < 8 {
            (0u32, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        } else {
            (self.skip, self.saved_a, self.saved_b, self.saved_c)
        };
        out[2] = skip as f32;
        out[4] = a.0 as f32;
        out[5] = a.1 as f32;
        out[6] = b.0 as f32;
        out[7] = b.1 as f32;
        out[8] = c.0 as f32;
        out[9] = c.1 as f32;
    }
}

/// Cosine-based color palette ("IQ palette"). `t` in [0, 1].
#[inline]
fn palette(t: f64) -> (u8, u8, u8) {
    use std::f64::consts::TAU;
    let a = (0.5, 0.5, 0.5);
    let b = (0.5, 0.5, 0.5);
    let c = (1.0, 1.0, 1.0);
    let d = (0.0, 0.10, 0.20);

    let r = a.0 + b.0 * (TAU * (c.0 * t + d.0)).cos();
    let g = a.1 + b.1 * (TAU * (c.1 * t + d.1)).cos();
    let bl = a.2 + b.2 * (TAU * (c.2 * t + d.2)).cos();

    (
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (bl.clamp(0.0, 1.0) * 255.0) as u8,
    )
}
