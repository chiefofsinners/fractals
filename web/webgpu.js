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
  _pad:       vec2f,
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
  // dc = pixel position relative to the *reference* point (not view centre).
  let uv = (fragxy / u.resolution) * 2.0 - 1.0;
  let dcx = uv.x * u.scale_x - u.ref_off.x;
  let dcy = uv.y * u.scale_y - u.ref_off.y;

  var dzx: f32 = 0.0;
  var dzy: f32 = 0.0;
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
    // dz_{n+1} = 2*Z*dz + dz^2 + dc
    let new_dzx = 2.0 * (Z.x * dzx - Z.y * dzy) + (dzx * dzx - dzy * dzy) + dcx;
    let new_dzy = 2.0 * (Z.x * dzy + Z.y * dzx) + 2.0 * dzx * dzy        + dcy;
    dzx = new_dzx;
    dzy = new_dzy;
  }

  // If the reference orbit ran out before this pixel escaped, keep going
  // in plain f32 from the current true orbit position. c is reconstructed
  // as ref_c + dc, where ref_c = ref_orbit[1] (because Z_1 = Z_0^2 + c = c
  // since Z_0 = 0). This is f32-precise for moderate zooms and avoids the
  // black-blob glitches that would otherwise appear wherever the reference
  // dies young; at extreme zoom the f32 ref_c is imprecise, but in that
  // regime a good in-set reference normally prevents this branch firing.
  if (iter >= u.max_iter && u.ref_iters < u.max_iter) {
    let lastZ = ref_orbit[u.ref_iters - 1u];
    let refC = ref_orbit[1u];
    var zx = lastZ.x + dzx;
    var zy = lastZ.y + dzy;
    let cx = refC.x + dcx;
    let cy = refC.y + dcy;
    for (var n: u32 = u.ref_iters; n < u.max_iter; n = n + 1u) {
      let zx2 = zx * zx;
      let zy2 = zy * zy;
      if (zx2 + zy2 > 65536.0) {
        iter = n;
        final_zx = zx;
        final_zy = zy;
        break;
      }
      let nzy = 2.0 * zx * zy + cy;
      zx = zx2 - zy2 + cx;
      zy = nzy;
    }
  }

  if (iter >= u.max_iter) { return vec3f(0.0); }

  let logZn = 0.5 * log(final_zx * final_zx + final_zy * final_zy);
  let nu = log(logZn / log(2.0)) / log(2.0);
  let smoothI = f32(iter) + 1.0 - nu;
  let t = clamp(smoothI / f32(u.max_iter), 0.0, 1.0);
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

  const uniformSize = 48; // see WGSL Uniforms layout (multiple of 16)
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
    render(width, height, cx, cy, scale, maxIter, refOrbit, refOffX, refOffY) {
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
      fView[6] = refOffX;
      fView[7] = refOffY;
      // fView[8..11] = padding
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
