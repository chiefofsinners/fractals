// WebGL2 Mandelbrot renderer — runs the iteration on the GPU as a fragment shader.
// Single-precision (f32) only; for deep zoom (scale < ~1e-5) callers should fall
// back to the WASM/f64 renderer.

const VERT = `#version 300 es
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }
`;

const FRAG = `#version 300 es
precision highp float;
out vec4 outColor;
uniform vec2  u_resolution;
uniform vec2  u_center;     // complex-plane center
uniform float u_scale;      // half-height of viewport in complex plane
uniform int   u_maxIter;
// 0 = Mandelbrot, 1 = Julia, 2 = Burning Ship.
//  Mandelbrot:   z0 = 0,    c = pixel,  z := z^2 + c
//  Julia:        z0 = pixel, c = u_jc,   z := z^2 + c
//  Burning Ship: z0 = 0,    c = pixel,  z := (|Re z| + i|Im z|)^2 + c
uniform int   u_mode;
uniform vec2  u_jc;

vec3 palette(float t) {
  const float TAU = 6.2831853;
  vec3 a = vec3(0.5);
  vec3 b = vec3(0.5);
  vec3 c = vec3(1.0);
  vec3 d = vec3(0.0, 0.10, 0.20);
  return clamp(a + b * cos(TAU * (c * t + d)), 0.0, 1.0);
}

void main() {
  float aspect = u_resolution.x / u_resolution.y;
  vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0; // [-1,1]
  // gl_FragCoord.y is bottom-up; negate so increasing screen-y → increasing
  // complex-y, matching the CPU and WebGPU backends.
  vec2 p = vec2(u_center.x + uv.x * u_scale * aspect,
                u_center.y - uv.y * u_scale);

  // For Mandelbrot/Burning Ship c is the pixel and z0 = 0; for Julia z0
  // is the pixel and c is the constant u_jc.
  vec2 c, z;
  if (u_mode == 1) {
    c = u_jc;
    z = p;
  } else {
    c = p;
    z = vec2(0.0);
    if (u_mode == 0) {
      // Cardioid / period-2 bulb early exit (Mandelbrot only).
      float xm = c.x - 0.25;
      float q = xm*xm + c.y*c.y;
      if (q * (q + xm) <= 0.25 * c.y * c.y ||
          (c.x + 1.0)*(c.x + 1.0) + c.y*c.y <= 0.0625) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
      }
    }
  }

  const float BAILOUT = 65536.0;
  int i = 0;
  float zx2 = 0.0, zy2 = 0.0;
  for (int k = 0; k < 100000; ++k) {
    if (k >= u_maxIter) { i = u_maxIter; break; }
    zx2 = z.x * z.x;
    zy2 = z.y * z.y;
    if (zx2 + zy2 > BAILOUT) { i = k; break; }
    if (u_mode == 2) {
      // Burning Ship: fold to |Re|, |Im| before squaring. Squared terms
      // are unaffected; only the cross term picks up the abs.
      z = vec2(zx2 - zy2 + c.x, 2.0 * abs(z.x) * abs(z.y) + c.y);
    } else {
      z = vec2(zx2 - zy2 + c.x, 2.0 * z.x * z.y + c.y);
    }
  }

  if (i >= u_maxIter) {
    outColor = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    float logZn = 0.5 * log(zx2 + zy2);
    float nu = log(logZn / log(2.0)) / log(2.0);
    float smooth_i = float(i) + 1.0 - nu;
    float t = clamp(smooth_i / float(u_maxIter), 0.0, 1.0);
    // Burning Ship and Julia escape very fast; gamma-compress so the few
    // low iteration counts still span the palette.
    if (u_mode == 1 || u_mode == 2) t = sqrt(t);
    outColor = vec4(palette(t), 1.0);
  }
}
`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error("Shader compile failed: " + log);
  }
  return sh;
}

export function createGpuRenderer(canvas) {
  const gl = canvas.getContext("webgl2", {
    antialias: false, alpha: false, preserveDrawingBuffer: false,
    powerPreference: "high-performance",
  });
  if (!gl) return null;

  // Some mobile GPUs expose WebGL2 but only low/mediump-like precision in
  // fragment highp float. Mandelbrot maths then collapses into a blob.
  const highp = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
  if (!highp || highp.precision < 23 || highp.rangeMax < 127) {
    return null;
  }

  const vs = compile(gl, gl.VERTEX_SHADER, VERT);
  const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    return null;
  }

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

  const aPos = gl.getAttribLocation(prog, "a_pos");
  const uRes = gl.getUniformLocation(prog, "u_resolution");
  const uCenter = gl.getUniformLocation(prog, "u_center");
  const uScale = gl.getUniformLocation(prog, "u_scale");
  const uMaxIter = gl.getUniformLocation(prog, "u_maxIter");
  const uMode = gl.getUniformLocation(prog, "u_mode");
  const uJc = gl.getUniformLocation(prog, "u_jc");

  gl.useProgram(prog);
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  // Runtime correctness checks using an offscreen 4×4 framebuffer.
  //
  // Check 1 — the iteration loop must run at all:
  //   c = (2, 0) escapes in 5 iterations → pixels must be coloured.
  //
  // Check 2 — iteration must be numerically accurate:
  //   c = (-1, 0.3) is inside the Mandelbrot set and lies outside the
  //   cardioid/bulb early-exit zone, so the loop must run many times and
  //   decide the orbit is bounded → pixels must be black.
  //   GPUs that secretly execute highp shaders at mediump precision (common
  //   on Adreno/Mali in Android browsers) accumulate rounding errors and
  //   incorrectly escape this orbit, returning coloured pixels instead.
  {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 4, 4, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

    const probe = (cx, cy) => {
      gl.viewport(0, 0, 4, 4);
      gl.uniform2f(uRes, 4, 4);
      gl.uniform2f(uCenter, cx, cy);
      gl.uniform1f(uScale, 0.001);  // tiny scale → all pixels ≈ same point
      gl.uniform1i(uMaxIter, 100);
      gl.uniform1i(uMode, 0);
      gl.uniform2f(uJc, 0, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      const px = new Uint8Array(4 * 4 * 4);
      gl.readPixels(0, 0, 4, 4, gl.RGBA, gl.UNSIGNED_BYTE, px);
      // Returns true if any RGB channel (not alpha) is non-zero.
      return px.some((v, i) => i % 4 !== 3 && v > 0);
    };

    const escapingIsColoured = probe(2.0, 0.0);   // must be true
    const interiorIsBlack    = !probe(-1.0, 0.3);  // must be true

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(tex);

    if (!escapingIsColoured || !interiorIsBlack) return null;
  }

  return {
    render(width, height, cx, cy, scale, maxIter, mode = 0, jcx = 0, jcy = 0) {
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      gl.viewport(0, 0, width, height);
      gl.uniform2f(uRes, width, height);
      gl.uniform2f(uCenter, cx, cy);
      gl.uniform1f(uScale, scale);
      gl.uniform1i(uMaxIter, maxIter | 0);
      gl.uniform1i(uMode, mode | 0);
      gl.uniform2f(uJc, jcx, jcy);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
  };
}
