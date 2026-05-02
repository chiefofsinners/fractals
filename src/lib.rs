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
                let t = (smooth / max_iter as f64).clamp(0.0, 1.0);
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
                let t = (smooth / max_iter as f64).clamp(0.0, 1.0);
                let (r, g, b) = palette(t);
                buf[idx] = r; buf[idx + 1] = g; buf[idx + 2] = b;
            }
            buf[idx + 3] = 255;
        }
    }
    buf
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

    let mut out: Vec<f32> = Vec::with_capacity(2 + 2 * (max_iter as usize + 1));
    // Header: offset from the view centre, subtracted in f64.
    out.push((best_cx - cx) as f32);
    out.push((best_cy - cy) as f32);

    let mut zx = 0.0f64;
    let mut zy = 0.0f64;
    out.push(zx as f32);
    out.push(zy as f32);
    let bailout = (1u64 << 16) as f64;
    for _ in 0..max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > bailout { break; }
        let nzy = 2.0 * zx * zy + best_cy;
        zx = zx2 - zy2 + best_cx;
        zy = nzy;
        out.push(zx as f32);
        out.push(zy as f32);
    }
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

    let mut out: Vec<f32> = Vec::with_capacity(2 + 2 * (max_iter as usize + 1));
    // Header: chosen reference *starting z*, expressed as offset from view centre.
    out.push((best_zx - cx) as f32);
    out.push((best_zy - cy) as f32);

    let mut zx = best_zx;
    let mut zy = best_zy;
    out.push(zx as f32);
    out.push(zy as f32);
    let bailout = (1u64 << 16) as f64;
    for _ in 0..max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > bailout { break; }
        let nzy = 2.0 * zx * zy + jcy;
        zx = zx2 - zy2 + jcx;
        zy = nzy;
        out.push(zx as f32);
        out.push(zy as f32);
    }
    out
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
