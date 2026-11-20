# ReefShader

Underwater diving video automatic colour correction and index-of-refraction-aware Gyroflow stabilization, rewritten in Rust (`wgpu` + `gyroflow-core` + `ffmpeg-next`, with an upcoming `egui` desktop UI).

## Workspace Layout

- `crates/reefshader-core`: Core processing pipeline (`wgpu` compute shaders, `gyroflow-core` stabilization, FFmpeg video I/O).
- `crates/reefshader-cli`: Command-line interface (`reefshader`).
- `legacy/`: Original Python / JAX / PySide6 prototype.

## Building (macOS)

Install FFmpeg and `pkg-config` via Homebrew, then build with Cargo:

```bash
brew install ffmpeg pkg-config
cargo build --release --bin reefshader
```

## Usage (CLI)

```bash
cargo run --bin reefshader -- --input dive.mp4 --output dive_corrected.mp4
```

Run `cargo run --bin reefshader -- --help` for all options.
