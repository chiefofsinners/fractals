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
