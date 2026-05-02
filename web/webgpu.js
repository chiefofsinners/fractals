// WebGPU Mandelbrot renderer.
//
// Uses "double-single" (df64) arithmetic in WGSL: each f64 value is represented
// as a pair of f32s (hi + lo) so we get ~48 bits of mantissa — enough to zoom
// well past where WebGL's f32 path gets pixelated. ~10× slower per iteration
// than plain f32 but still very fast on a discrete or Apple-silicon GPU.

const SHADER = /* wgsl */ `
struct Uniforms {
  resolution: vec2f,
  cx:    vec2f,  // (hi, lo)
  cy:    vec2f,
  scale: vec2f,
  max_iter: u32,
  use_df: u32,   // 0 = pure f32 fast path, 1 = double-single
  splitter: f32, // = 4097.0; passed via uniform to defeat compiler folding
  pad0: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

// ---- double-single arithmetic ----------------------------------------------
//
// Critical: WGSL allows the compiler to contract / reassociate fp ops, which
// silently destroys the error-tracking algorithms below (two_sum's e and
// two_prod's e both algebraically simplify to 0). We use bitcast<u32>
// roundtrips as opaque "fences" the optimizer cannot see through.

fn opaque(a: f32) -> f32 {
  // Add a runtime-zero uniform — compiler can't constant-fold a uniform load,
  // so x + 0_uniform isn't simplified. This is the only fence that survives
  // Tint→Metal's "fast math" optimization on Apple GPUs.
  return a + u.pad0;
}

fn two_sum(a: f32, b: f32) -> vec2f {
  let s = a + b;
  let bb = opaque(s) - a;
  let e = (a - opaque(opaque(s) - bb)) + (b - bb);
  return vec2f(s, e);
}
// Veltkamp split. The compiler tries hard to simplify (t - (t - a)) → a;
// we wrap each arithmetic step in opaque() so it can't see through.
fn split(a: f32) -> vec2f {
  let t = opaque(u.splitter * a);
  let t_minus_a = opaque(t - a);
  let hi = opaque(t - t_minus_a);
  let lo = a - hi;
  return vec2f(hi, lo);
}
fn two_prod(a: f32, b: f32) -> vec2f {
  let p = a * b;
  let aa = split(a);
  let bb = split(b);
  // Each of aa.x*bb.x, aa.x*bb.y, aa.y*bb.x, aa.y*bb.y is exactly representable
  // in f32 (split makes hi 12-bit, lo 12-bit). Sum them carefully.
  let hh = aa.x * bb.x;
  let hl = aa.x * bb.y;
  let lh = aa.y * bb.x;
  let ll = aa.y * bb.y;
  let e = ((opaque(hh) - p) + hl + lh) + ll;
  return vec2f(p, e);
}
// Quick two-sum: assumes |a| >= |b| (faster, no opaque needed for branchless).
fn quick_two_sum(a: f32, b: f32) -> vec2f {
  let s = a + b;
  let e = b - (opaque(s) - a);
  return vec2f(s, e);
}
fn df_add(a: vec2f, b: vec2f) -> vec2f {
  // IEEE-correct df + df from Hida/Li/Bailey "Library for Double-Double".
  let s1 = two_sum(a.x, b.x);
  let s2 = two_sum(a.y, b.y);
  let s = quick_two_sum(s1.x, s1.y + s2.x);
  return quick_two_sum(s.x, s.y + s2.y);
}
fn df_sub(a: vec2f, b: vec2f) -> vec2f { return df_add(a, vec2f(-b.x, -b.y)); }
// Multiply df by 2 (exact in IEEE: scaling by power of 2 introduces no error).
fn df_dbl(a: vec2f) -> vec2f { return vec2f(a.x + a.x, a.y + a.y); }
fn df_mul(a: vec2f, b: vec2f) -> vec2f {
  let p = two_prod(a.x, b.x);
  // p.y already holds the rounding error of a.x*b.x; add the cross terms.
  let e = p.y + (a.x * b.y + a.y * b.x);
  return quick_two_sum(p.x, e);
}
// Fence both components — call between iteration steps so the optimizer
// can't fuse / reassociate df_add(df_sub(zx²,zy²),cx) across iterations.
fn df_fence(a: vec2f) -> vec2f {
  return vec2f(opaque(a.x), opaque(a.y));
}
fn df_f32(a: f32) -> vec2f { return vec2f(a, 0.0); }
fn df_to_f32(a: vec2f) -> f32 { return a.x + a.y; }

// ---- color palette ---------------------------------------------------------

fn palette(t: f32) -> vec3f {
  let TAU = 6.2831853;
  let a = vec3f(0.5);
  let b = vec3f(0.5);
  let c = vec3f(1.0);
  let d = vec3f(0.0, 0.10, 0.20);
  return clamp(a + b * cos(TAU * (c * t + d)), vec3f(0.0), vec3f(1.0));
}

// ---- pipeline --------------------------------------------------------------

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  // Single full-screen triangle.
  let pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  return vec4f(pos[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  // 2x2 ordered-grid supersampling — cheap on the GPU, kills edge aliasing.
  var acc = vec3f(0.0);
  let offsets = array<vec2f, 4>(
    vec2f(-0.25, -0.25),
    vec2f( 0.25, -0.25),
    vec2f(-0.25,  0.25),
    vec2f( 0.25,  0.25),
  );
  for (var s: u32 = 0u; s < 4u; s = s + 1u) {
    acc = acc + sample_at(frag.xy + offsets[s]);
  }
  return vec4f(acc * 0.25, 1.0);
}

fn sample_at(fragxy: vec2f) -> vec3f {
  let aspect = u.resolution.x / u.resolution.y;
  let uv = (fragxy / u.resolution) * 2.0 - 1.0;

  let sx_df = df_mul(df_f32(uv.x * aspect), u.scale);
  let sy_df = df_mul(df_f32(uv.y),          u.scale);
  let cxd = df_add(u.cx, sx_df);
  let cyd = df_add(u.cy, sy_df);
  let cxf0 = df_to_f32(cxd);
  let cyf0 = df_to_f32(cyd);

  let xm = cxf0 - 0.25;
  let q = xm * xm + cyf0 * cyf0;
  if (q * (q + xm) <= 0.25 * cyf0 * cyf0
      || (cxf0 + 1.0) * (cxf0 + 1.0) + cyf0 * cyf0 <= 0.0625) {
    return vec3f(0.0);
  }
  var iter: u32 = u.max_iter;
  var final_mag2: f32 = 0.0;

  if (u.use_df == 0u) {
    let cx = cxf0;
    let cy = cyf0;
    var zx: f32 = 0.0;
    var zy: f32 = 0.0;
    var zx2: f32 = 0.0;
    var zy2: f32 = 0.0;
    for (var k: u32 = 0u; k < u.max_iter; k = k + 1u) {
      zx2 = zx * zx;
      zy2 = zy * zy;
      if (zx2 + zy2 > 65536.0) { iter = k; break; }
      let nzy = 2.0 * zx * zy + cy;
      zx = zx2 - zy2 + cx;
      zy = nzy;
    }
    final_mag2 = zx2 + zy2;
  } else {
    let cx = cxd;
    let cy = cyd;
    var zx = vec2f(0.0);
    var zy = vec2f(0.0);
    var zx2 = vec2f(0.0);
    var zy2 = vec2f(0.0);
    let two = df_f32(2.0);
    for (var k: u32 = 0u; k < u.max_iter; k = k + 1u) {
      zx2 = df_mul(zx, zx);
      zy2 = df_mul(zy, zy);
      let mag = df_to_f32(df_add(zx2, zy2));
      if (mag > 65536.0) { iter = k; break; }
      let two_zxzy = df_dbl(df_mul(zx, zy));
      zy = df_fence(df_add(two_zxzy, cy));
      zx = df_fence(df_add(df_sub(zx2, zy2), cx));
    }
    final_mag2 = df_to_f32(df_add(zx2, zy2));
  }

  if (iter >= u.max_iter) { return vec3f(0.0); }

  let logZn = 0.5 * log(final_mag2);
  let nu = log(logZn / log(2.0)) / log(2.0);
  let smoothI = f32(iter) + 1.0 - nu;
  let t = clamp(smoothI / f32(u.max_iter), 0.0, 1.0);
  return palette(t);
}

fn _disabled_iterate(cxd: vec2f, cyd: vec2f, cxf0: f32, cyf0: f32) -> vec3f {
  let xm = cxf0 - 0.25;
  let q = xm * xm + cyf0 * cyf0;
  if (q * (q + xm) <= 0.25 * cyf0 * cyf0
      || (cxf0 + 1.0) * (cxf0 + 1.0) + cyf0 * cyf0 <= 0.0625) {
    return vec3f(0.0);
  }

  var iter: u32 = u.max_iter;
  var final_mag2: f32 = 0.0;

  if (u.use_df == 0u) {
    let cx = cxf0;
    let cy = cyf0;
    var zx: f32 = 0.0;
    var zy: f32 = 0.0;
    var zx2: f32 = 0.0;
    var zy2: f32 = 0.0;
    for (var k: u32 = 0u; k < u.max_iter; k = k + 1u) {
      zx2 = zx * zx;
      zy2 = zy * zy;
      if (zx2 + zy2 > 65536.0) { iter = k; break; }
      let nzy = 2.0 * zx * zy + cy;
      zx = zx2 - zy2 + cx;
      zy = nzy;
    }
    final_mag2 = zx2 + zy2;
  } else {
    let cx = cxd;
    let cy = cyd;
    var zx = vec2f(0.0);
    var zy = vec2f(0.0);
    var zx2 = vec2f(0.0);
    var zy2 = vec2f(0.0);
    let two = df_f32(2.0);
    for (var k: u32 = 0u; k < u.max_iter; k = k + 1u) {
      zx2 = df_mul(zx, zx);
      zy2 = df_mul(zy, zy);
      let mag = df_to_f32(df_add(zx2, zy2));
      if (mag > 65536.0) { iter = k; break; }
      let two_zxzy = df_mul(df_mul(two, zx), zy);
      zy = df_add(two_zxzy, cy);
      zx = df_add(df_sub(zx2, zy2), cx);
    }
    final_mag2 = df_to_f32(df_add(zx2, zy2));
  }

  if (iter >= u.max_iter) { return vec3f(0.0); }

  let logZn = 0.5 * log(final_mag2);
  let nu = log(logZn / log(2.0)) / log(2.0);
  let smoothI = f32(iter) + 1.0 - nu;
  let t = clamp(smoothI / f32(u.max_iter), 0.0, 1.0);
  return palette(t);
}
`;

function splitDouble(d) {
  // Veltkamp split into hi+lo f32 so hi+lo == d (to f32 precision).
  const hi = Math.fround(d);
  const lo = d - hi;
  return [hi, Math.fround(lo)];
}

export async function createWebGpuRenderer(canvas) {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return null;
  const device = await adapter.requestDevice();

  const ctx = canvas.getContext("webgpu");
  if (!ctx) return null;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const module = device.createShaderModule({ code: SHADER });
  // Surface WGSL compile diagnostics in the console.
  if (module.getCompilationInfo) {
    module.getCompilationInfo().then(info => {
      for (const m of info.messages) {
        const where = `WGSL ${m.type} (line ${m.lineNum}:${m.linePos}): ${m.message}`;
        if (m.type === "error") console.error(where);
        else if (m.type === "warning") console.warn(where);
        else console.log(where);
      }
    });
  }
  device.addEventListener?.("uncapturederror", (ev) => {
    console.error("WebGPU uncaptured error:", ev.error?.message || ev.error);
  });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const uniformSize = 64; // see WGSL Uniforms layout (multiple of 16)
  const uniformBuf = device.createBuffer({
    size: uniformSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuf } }],
  });

  const cpuUniform = new ArrayBuffer(uniformSize);
  const fView = new Float32Array(cpuUniform);
  const uView = new Uint32Array(cpuUniform);

  let lastUseDf = 0;
  return {
    lastPrecision() { return lastUseDf ? "df64" : "f32"; },
    render(width, height, cx, cy, scale, maxIter) {
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      // Always use df64 — f32 aliases too easily even at moderate zooms.
      const useDf = 1;
      lastUseDf = useDf;

      fView[0] = width;
      fView[1] = height;
      const [cxh, cxl] = splitDouble(cx);
      const [cyh, cyl] = splitDouble(cy);
      const [scH, scL] = splitDouble(scale);
      fView[2] = cxh; fView[3] = cxl;
      fView[4] = cyh; fView[5] = cyl;
      fView[6] = scH; fView[7] = scL;
      uView[8] = maxIter | 0;
      uView[9] = useDf;
      fView[10] = 4097.0; // splitter
      fView[11] = 0.0;    // pad0 — used as opaque-zero fence in WGSL
      device.queue.writeBuffer(uniformBuf, 0, cpuUniform);

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: ctx.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3, 1, 0, 0);
      pass.end();
      device.queue.submit([encoder.finish()]);
    },
  };
}
