pub mod config;
pub mod error;
pub mod gpu;
pub mod stabilize;
pub mod video;

use std::path::Path;

use config::ProcessingConfig;
use error::Result;
use gpu::GpuContext;
use stabilize::Stabilizer;

pub fn init() -> Result<()> {
    video::init_ffmpeg()?;
    Ok(())
}

pub fn process_video(input: &Path, output: &Path, config: &ProcessingConfig) -> Result<()> {
    let _gpu = GpuContext::new()?;
    let meta = video::probe_video(input)?;
    log::info!(
        "Loaded {}: {}x{} @ {:.2} fps ({:.2}s, {} frames)",
        input.display(),
        meta.width,
        meta.height,
        meta.fps,
        meta.duration_ms / 1000.0,
        meta.frame_count
    );

    if config.gyroflow.enabled {
        let stabilizer = Stabilizer::new(&config.gyroflow);
        stabilizer.load_video_metadata(
            input,
            meta.duration_ms,
            meta.fps,
            meta.frame_count,
            (meta.width as usize, meta.height as usize),
        )?;
        log::info!(
            "Gyroflow stabilization ready (refraction={:.2})",
            if config.gyroflow.underwater {
                config.gyroflow.light_refraction_coefficient
            } else {
                1.0
            }
        );
    }

    log::info!("Target output path: {}", output.display());
    Err(error::ReefShaderError::Message(
        "Frame processing pipeline not implemented yet".to_string(),
    ))
}
