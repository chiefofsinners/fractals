// WebGPU Mandelbrot renderer — perturbation theory.
//
// The CPU computes one high-precision (f64) "reference orbit" Z_n at the
// view center. The GPU then iterates a per-pixel delta orbit dz around it
// using the linearised recurrence
//
//   dz_{n+1} = 2 * Z_n * dz_n + dz_n^2 + dc          (complex)
//
// where dc is the per-pixel offset from the reference point in the complex
// plane. Because Z_n is fixed for every pixel and dc is small, all the
// per-pixel work fits in plain f32 with no precision loss — even at zooms
// far past where df64 collapses on Apple's Metal compiler.
//
// Bailout uses the *true* orbit Z + dz so the colouring matches the CPU
// renderer exactly. We don't yet do glitch correction (rebasing onto a new
// reference when |dz| approaches |Z|), so extreme zooms over highly chaotic
// regions can still show artefacts; for the zoom range users actually
// navigate to, this is good for many orders of magnitude beyond f32.

const SHADER = /* wgsl */ `
struct Uniforms {
  resolution: vec2f,
  scale_x:    f32,   // half-width of the view in complex units (= scale * aspect)
  scale_y:    f32,   // half-height of the view in complex units (= scale)
  max_iter:   u32,
  ref_iters:  u32,   // length of the reference orbit (<= max_iter+1)
  // (ref - view_centre) in complex coords. Subtracted from per-pixel dc
  // so the perturbation is taken around the chosen reference, not the view
  // centre. Computed in f64 on the CPU (Rust) before being cast to f32 —
  // f32 is plenty for this small offset (|ref_off| <= scale).
  ref_off:    vec2f,
  // 0 = Mandelbrot (z0 = 0, c varies per pixel), 1 = Julia (z0 varies
  // per pixel, c is the constant 'jc'), 2 = Burning Ship (folded square,
  // z0 = 0, c varies per pixel; uses sign(Z) to track abs perturbation).
  mode:       u32,
  _pad0:      u32,
  jc:         vec2f,
  _pad1:      vec2f,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> ref_orbit: array<vec2f>;

fn palette(t: f32) -> vec3f {
  let TAU = 6.2831853;
  let a = vec3f(0.5);
  let b = vec3f(0.5);
  let c = vec3f(1.0);
  let d = vec3f(0.0, 0.10, 0.20);
  return clamp(a + b * cos(TAU * (c * t + d)), vec3f(0.0), vec3f(1.0));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  return vec4f(pos[vi], 0.0, 1.0);
}

fn sample_at(fragxy: vec2f) -> vec3f {
  // dPix = pixel position relative to the chosen reference point. For
  // Mandelbrot this is 'dc' (per-pixel offset of c, since z0 = 0); for
  // Julia it's 'dz0' (per-pixel offset of the starting z, since c is
  // constant). Either way the per-pixel state we iterate is dz, and the
  // recurrence differs only by whether we add dPix each step.
  let uv = (fragxy / u.resolution) * 2.0 - 1.0;
  let dpx = uv.x * u.scale_x - u.ref_off.x;
  let dpy = uv.y * u.scale_y - u.ref_off.y;

  // Julia: dz starts at the pixel offset; Mandelbrot/Burning Ship: dz starts at 0.
  var dzx: f32 = select(0.0, dpx, u.mode == 1u);
  var dzy: f32 = select(0.0, dpy, u.mode == 1u);
  // For Mandelbrot/Burning Ship, dPix is the per-iteration '+dc'; for Julia, no add.
  let addx: f32 = select(dpx, 0.0, u.mode == 1u);
  let addy: f32 = select(dpy, 0.0, u.mode == 1u);

  var iter: u32 = u.max_iter;
  var final_zx: f32 = 0.0;
  var final_zy: f32 = 0.0;

  let limit = min(u.max_iter, u.ref_iters);
  for (var n: u32 = 0u; n < limit; n = n + 1u) {
    let Z = ref_orbit[n];
    // True orbit at this pixel = Z + dz. Bail out using its magnitude.
    let zx = Z.x + dzx;
    let zy = Z.y + dzy;
    let mag2 = zx * zx + zy * zy;
    if (mag2 > 65536.0) {
      iter = n;
      final_zx = zx;
      final_zy = zy;
      break;
    }
    if (u.mode == 2u) {
      // Burning Ship perturbation. Reference recurrence is
      //   X' = X^2 - Y^2 + Cx,   Y' = 2|X||Y| + Cy.
      // dx' is straightforward (the squares are analytic):
      //   dx' = 2X dx - 2Y dy + dx^2 - dy^2 + dcx.
      // For dy' the absolute values give an exact identity
      //   2|x||y| - 2|X||Y|
      //     = 2 sxy (X dy + Y dx + dx dy)        (no cancellation when small)
      //     + 2 (sxy - SXY) X Y                  (exactly 0 unless a sign flips)
      // where SXY = sgn(X) sgn(Y) is the reference sign and sxy = sgn(x) sgn(y)
      // is the *true* pixel sign (we already have x = X+dx, y = Y+dy from the
      // bailout test). The correction term cleanly handles glitches near the
      // axes, where the naive linearisation otherwise produces bright streaks.
      let SX = select(1.0, -1.0, Z.x < 0.0);
      let SY = select(1.0, -1.0, Z.y < 0.0);
      let SXY = SX * SY;
      let sx = select(1.0, -1.0, zx < 0.0);
      let sy = select(1.0, -1.0, zy < 0.0);
      let sxy = sx * sy;
      let new_dzx = 2.0 * (Z.x * dzx - Z.y * dzy) + (dzx * dzx - dzy * dzy) + addx;
      let new_dzy = 2.0 * sxy * (Z.x * dzy + Z.y * dzx + dzx * dzy)
                  + 2.0 * (sxy - SXY) * Z.x * Z.y
                  + addy;
      dzx = new_dzx;
      dzy = new_dzy;
    } else {
      // Mandelbrot/Julia: dz_{n+1} = 2*Z*dz + dz^2  (+ dc for Mandelbrot)
      let new_dzx = 2.0 * (Z.x * dzx - Z.y * dzy) + (dzx * dzx - dzy * dzy) + addx;
      let new_dzy = 2.0 * (Z.x * dzy + Z.y * dzx) + 2.0 * dzx * dzy        + addy;
      dzx = new_dzx;
      dzy = new_dzy;
    }
  }

  // If the reference orbit ran out before this pixel escaped, keep going
  // in plain f32 from the current true orbit position.
  // Mandelbrot: c = ref_c + dc, where ref_c = ref_orbit[1] (Z_1 = c since
  //   Z_0 = 0). Julia: c is constant = u.jc.
  if (iter >= u.max_iter && u.ref_iters < u.max_iter) {
    let lastZ = ref_orbit[u.ref_iters - 1u];
    var zx = lastZ.x + dzx;
    var zy = lastZ.y + dzy;
    var cx: f32;
    var cy: f32;
    if (u.mode == 1u) {
      cx = u.jc.x;
      cy = u.jc.y;
    } else {
      let refC = ref_orbit[1u];
      cx = refC.x + dpx;
      cy = refC.y + dpy;
    }
    for (var n: u32 = u.ref_iters; n < u.max_iter; n = n + 1u) {
      let zx2 = zx * zx;
      let zy2 = zy * zy;
      if (zx2 + zy2 > 65536.0) {
        iter = n;
        final_zx = zx;
        final_zy = zy;
        break;
      }
      if (u.mode == 2u) {
        // Burning Ship: fold to |Re|, |Im| before squaring.
        let nzy = 2.0 * abs(zx) * abs(zy) + cy;
        zx = zx2 - zy2 + cx;
        zy = nzy;
      } else {
        let nzy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zy = nzy;
      }
    }
  }

  if (iter >= u.max_iter) { return vec3f(0.0); }

  let logZn = 0.5 * log(final_zx * final_zx + final_zy * final_zy);
  let nu = log(logZn / log(2.0)) / log(2.0);
  let smoothI = f32(iter) + 1.0 - nu;
  var t = clamp(smoothI / f32(u.max_iter), 0.0, 1.0);
  // Julia and Burning Ship escape fast across most pixels; gamma-compress
  // so the narrow iteration range still spans the palette. (Mandelbrot
  // keeps the linear mapping it's always had.)
  if (u.mode == 1u || u.mode == 2u) { t = sqrt(t); }
  return palette(t);
}

@fragment
fn fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  // 2x2 ordered-grid supersampling.
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
`;

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

  // Uniforms layout (see WGSL): 32B base + mode/_pad0 (8B) + jc/_pad1 (16B) = 56B,
  // padded up to a 16B multiple = 64B.
  const uniformSize = 64;
  const uniformBuf = device.createBuffer({
    size: uniformSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Storage buffer for the reference orbit. Sized once for the max iteration
  // count we'd ever request; we just write the active prefix per draw.
  const REF_MAX = 10001; // max_iter cap is 10000 in the UI
  const refBuf = device.createBuffer({
    size: REF_MAX * 8, // vec2<f32>
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: refBuf } },
    ],
  });

  const cpuUniform = new ArrayBuffer(uniformSize);
  const fView = new Float32Array(cpuUniform);
  const uView = new Uint32Array(cpuUniform);

  return {
    lastPrecision() { return "perturb"; },
    render(width, height, cx, cy, scale, maxIter, refOrbit, refOffX, refOffY,
           mode = 0, jcx = 0, jcy = 0) {
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      const aspect = width / height;
      // refOffX/refOffY are computed by the caller in f64 from the cached
      // absolute reference point and the current view centre; doing it
      // there (instead of reading the f32 ref_off baked into the orbit
      // header) keeps precision when reusing a cached orbit across pans
      // and zooms, which matters past ~scale 1e-7.
      const orbitData = refOrbit.subarray(2);
      const refIters = Math.min(REF_MAX, orbitData.length / 2);

      fView[0] = width;
      fView[1] = height;
      fView[2] = scale * aspect; // scale_x
      fView[3] = scale;          // scale_y
      uView[4] = maxIter | 0;
      uView[5] = refIters | 0;
      fView[6] = refOffX;        // ref_off.x
      fView[7] = refOffY;        // ref_off.y
      uView[8] = mode | 0;       // 0 = Mandelbrot, 1 = Julia, 2 = Burning Ship
      // uView[9] = _pad0
      fView[10] = jcx;           // jc.x
      fView[11] = jcy;           // jc.y
      // fView[12..15] = _pad1
      device.queue.writeBuffer(uniformBuf, 0, cpuUniform);

      // Upload reference orbit (clipped to REF_MAX).
      const bytes = refIters * 8;
      device.queue.writeBuffer(
        refBuf, 0, orbitData.buffer, orbitData.byteOffset, bytes
      );

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
