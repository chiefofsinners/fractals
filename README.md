# Mandelbrot · Rust + WebAssembly

A Mandelbrot set explorer written in Rust, compiled to WebAssembly, and rendered
to an HTML canvas. Interactive: click to zoom in, Shift+click to zoom out,
mouse-wheel to zoom around the cursor, drag to pan.

## Prerequisites

Install the Rust toolchain and `wasm-pack`:

```sh
# Rust (rustup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
# or: cargo install wasm-pack
```

## Build

Compile the Rust crate to a web-ready WASM package in `web/pkg/`:

```sh
wasm-pack build --target web --release --out-dir web/pkg
```

## Run locally

The page loads `pkg/fractals.js` as an ES module, so it must be served over
HTTP (not `file://`). Any static server works:

```sh
# Python
python3 -m http.server -d web 8080

# or Node
npx http-server web -p 8080
```

Then open <http://localhost:8080>.

## Deploy

After `wasm-pack build`, the entire `web/` directory is a self-contained static
site. Upload it to GitHub Pages, Netlify, Cloudflare Pages, S3, etc. Make sure
your host serves `.wasm` files with `Content-Type: application/wasm` (most do
by default).

## Project layout

- [Cargo.toml](Cargo.toml) – crate manifest (`cdylib` for wasm).
- [src/lib.rs](src/lib.rs) – `render_mandelbrot` exported via `wasm-bindgen`,
  smooth coloring, cardioid/bulb early exit.
- [web/index.html](web/index.html) – page shell and controls.
- [web/main.js](web/main.js) – loads the wasm module, renders to canvas,
  handles zoom/pan input.
