use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReefShaderError {
    #[error("No compatible wgpu GPU adapter found")]
    NoGpuAdapter,

    #[error("Failed to request wgpu device: {0}")]
    GpuDeviceRequest(#[from] wgpu::RequestDeviceError),

    #[error("Gyroflow error: {0}")]
    Gyroflow(#[from] gyroflow_core::GyroflowCoreError),

    #[cfg(feature = "ffmpeg")]
    #[error("FFmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, ReefShaderError>;
