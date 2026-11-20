use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use reefshader_core::config::ProcessingConfig;

#[derive(Debug, Parser)]
#[command(
    name = "reefshader",
    version,
    about = "Underwater diving video colour correction and refraction-aware stabilization"
)]
struct Cli {
    /// Input video file
    #[arg(short, long)]
    input: PathBuf,

    /// Output video file
    #[arg(short, long)]
    output: PathBuf,

    /// Optional output width (preserves aspect ratio; omitted keeps source resolution)
    #[arg(long)]
    width: Option<u32>,

    /// Disable automatic colour normalisation
    #[arg(long)]
    no_colour_norm: bool,

    /// Maximum per-channel normalisation gain
    #[arg(long, default_value_t = 10.0)]
    max_gain: f32,

    /// Temporal smoothing factor in [0.0, 1.0)
    #[arg(long, default_value_t = 0.95)]
    temporal_smoothing: f32,

    /// Disable gamma correction
    #[arg(long)]
    no_gamma: bool,

    /// Gamma exponent
    #[arg(long, default_value_t = 1.1)]
    gamma: f32,

    /// Disable Gyroflow stabilisation
    #[arg(long)]
    no_gyroflow: bool,

    /// Disable underwater index-of-refraction compensation (uses 1.0 instead of --refraction)
    #[arg(long)]
    no_underwater: bool,

    /// Water index of refraction for underwater lens correction
    #[arg(long, default_value_t = 1.33)]
    refraction: f64,

    /// Render processed and original video side by side
    #[arg(long)]
    side_by_side: bool,

    /// Output video encoder name (e.g. hevc_videotoolbox, libx265, libx264)
    #[arg(long)]
    codec: Option<String>,

    /// Target output bitrate in Mbps
    #[arg(long, default_value_t = 20)]
    bitrate: u32,

    /// Force CPU colour processing instead of wgpu GPU compute
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    reefshader_core::init()?;

    let mut config = ProcessingConfig::default();
    if let Some(width) = cli.width {
        config.scaling.enabled = true;
        config.scaling.width = Some(width);
    }
    config.colour_norm.enabled = !cli.no_colour_norm;
    config.colour_norm.max_gain = cli.max_gain;
    config.colour_norm.temporal_smoothing = cli.temporal_smoothing;
    config.gamma.enabled = !cli.no_gamma;
    config.gamma.gamma = cli.gamma;
    config.gyroflow.enabled = !cli.no_gyroflow;
    config.gyroflow.underwater = !cli.no_underwater;
    config.gyroflow.light_refraction_coefficient = cli.refraction;
    config.output.side_by_side = cli.side_by_side;
    config.encode.codec = cli.codec;
    config.encode.bitrate_mbps = cli.bitrate;
    config.force_cpu = cli.cpu;

    let summary = reefshader_core::process_video(&cli.input, &cli.output, &config)?;
    println!(
        "Done: {} frames ({}x{}) via {} [{}] -> {} ({:.2}s)",
        summary.frames_processed,
        summary.width,
        summary.height,
        summary.encoder_name,
        summary.colour_backend.name(),
        cli.output.display(),
        summary.elapsed_secs
    );
    Ok(())
}
