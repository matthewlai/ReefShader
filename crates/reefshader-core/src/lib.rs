pub mod colour;
pub mod config;
pub mod error;
pub mod gpu;
pub mod stabilize;
pub mod video;

use std::path::Path;

use colour::ColourPipeline;
use config::ProcessingConfig;
use error::Result;
use stabilize::Stabilizer;
use video::ProcessSummary;

pub fn init() -> Result<()> {
    video::init_ffmpeg()?;
    Ok(())
}

pub fn process_video(
    input: &Path,
    output: &Path,
    config: &ProcessingConfig,
) -> Result<ProcessSummary> {
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
        match stabilizer.load_video_metadata(
            input,
            meta.duration_ms,
            meta.fps,
            meta.frame_count,
            (meta.width as usize, meta.height as usize),
        ) {
            Ok(()) => {
                log::info!(
                    "Gyroflow metadata loaded (refraction={:.2}); per-frame stabilization pass not wired yet",
                    if config.gyroflow.underwater {
                        config.gyroflow.light_refraction_coefficient
                    } else {
                        1.0
                    }
                );
            }
            Err(err) => {
                log::warn!("Skipping Gyroflow metadata ({err})");
            }
        }
    }

    let mut pipeline = ColourPipeline::new(config.force_cpu);
    let summary = video::transcode_video(input, output, config, &mut pipeline)?;

    log::info!(
        "Finished {} -> {} ({} frames, {}x{}, backend={}, encoder={}, {:.2}s)",
        input.display(),
        output.display(),
        summary.frames_processed,
        summary.width,
        summary.height,
        summary.colour_backend.name(),
        summary.encoder_name,
        summary.elapsed_secs
    );

    Ok(summary)
}
