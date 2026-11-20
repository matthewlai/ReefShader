use crate::error::{ReefShaderError, Result};

/// Shared wgpu GPU context used by ReefShader's color-correction compute shaders
/// (and later shared directly with egui).
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    pub async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|_| ReefShaderError::NoGpuAdapter)?;

        let info = adapter.get_info();
        log::info!(
            "Using wgpu adapter: {} ({:?}, {:?})",
            info.name,
            info.device_type,
            info.backend
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("reefshader-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}
