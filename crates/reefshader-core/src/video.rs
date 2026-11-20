use std::path::Path;

use crate::error::Result;

#[derive(Debug, Clone)]
pub struct VideoMetadata {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration_ms: f64,
    pub frame_count: usize,
}

#[cfg(feature = "ffmpeg")]
pub fn init_ffmpeg() -> Result<()> {
    ffmpeg_next::init()?;
    Ok(())
}

#[cfg(feature = "ffmpeg")]
pub fn probe_video(path: &Path) -> Result<VideoMetadata> {
    let ictx = ffmpeg_next::format::input(path)?;
    let stream = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or_else(|| {
            crate::error::ReefShaderError::Message(format!(
                "No video stream found in {}",
                path.display()
            ))
        })?;

    let rate = stream.avg_frame_rate();
    let fps = if rate.denominator() != 0 {
        f64::from(rate.numerator()) / f64::from(rate.denominator())
    } else {
        30.0
    };

    let codec_params = stream.parameters();
    let decoder = ffmpeg_next::codec::context::Context::from_parameters(codec_params)?;
    let video = decoder.decoder().video()?;

    let duration_secs = if ictx.duration() > 0 {
        ictx.duration() as f64 / f64::from(ffmpeg_next::ffi::AV_TIME_BASE)
    } else {
        0.0
    };
    let frame_count = if stream.frames() > 0 {
        stream.frames() as usize
    } else {
        (duration_secs * fps).round() as usize
    };

    Ok(VideoMetadata {
        width: video.width(),
        height: video.height(),
        fps,
        duration_ms: duration_secs * 1000.0,
        frame_count,
    })
}

#[cfg(not(feature = "ffmpeg"))]
pub fn init_ffmpeg() -> Result<()> {
    Ok(())
}

#[cfg(not(feature = "ffmpeg"))]
pub fn probe_video(_path: &Path) -> Result<VideoMetadata> {
    Err(crate::error::ReefShaderError::Message(
        "Built without ffmpeg feature".to_string(),
    ))
}
