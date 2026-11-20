use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::config::{ColourNormConfig, ProcessingConfig};
use crate::error::{ReefShaderError, Result};
use crate::gpu::GpuContext;

const HIST_BINS: usize = 4096;
const HIST_LEN: usize = 3 * HIST_BINS;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ShaderParams {
    width: u32,
    height: u32,
    downsample_win: u32,
    ds_width: u32,
    ds_height: u32,
    colour_norm_enabled: u32,
    gamma_enabled: u32,
    side_by_side: u32,
    gamma: f32,
    blackpoint: f32,
    _pad0: [f32; 2],
    mins: [f32; 4],
    gains: [f32; 4],
}

#[derive(Debug, Default, Clone)]
pub struct ColourState {
    prev_mins: Option<[f32; 3]>,
    prev_maxs: Option<[f32; 3]>,
}

impl ColourState {
    pub fn reset(&mut self) {
        self.prev_mins = None;
        self.prev_maxs = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColourBackend {
    Wgpu,
    Cpu,
}

impl ColourBackend {
    pub fn name(self) -> &'static str {
        match self {
            ColourBackend::Wgpu => "wgpu",
            ColourBackend::Cpu => "cpu",
        }
    }
}

pub struct ColourPipeline {
    backend: PipelineBackend,
    state: ColourState,
}

enum PipelineBackend {
    Gpu(GpuColourEngine),
    Cpu,
}

impl ColourPipeline {
    pub fn new(force_cpu: bool) -> Self {
        if force_cpu {
            log::info!("Using CPU colour pipeline (--cpu requested)");
            return Self {
                backend: PipelineBackend::Cpu,
                state: ColourState::default(),
            };
        }

        match GpuContext::new().and_then(GpuColourEngine::new) {
            Ok(engine) => Self {
                backend: PipelineBackend::Gpu(engine),
                state: ColourState::default(),
            },
            Err(err) => {
                log::warn!(
                    "Falling back to CPU colour pipeline (wgpu unavailable: {err})"
                );
                Self {
                    backend: PipelineBackend::Cpu,
                    state: ColourState::default(),
                }
            }
        }
    }

    pub fn backend(&self) -> ColourBackend {
        match &self.backend {
            PipelineBackend::Gpu(_) => ColourBackend::Wgpu,
            PipelineBackend::Cpu => ColourBackend::Cpu,
        }
    }

    pub fn reset_temporal_state(&mut self) {
        self.state.reset();
    }

    /// Processes a tightly-packed `RGBA64LE` frame (`width * height * 4` `u16`s) in place.
    pub fn process_rgba16(
        &mut self,
        pixels: &mut [u16],
        width: u32,
        height: u32,
        config: &ProcessingConfig,
    ) -> Result<()> {
        let expected_len = (width as usize) * (height as usize) * 4;
        if pixels.len() != expected_len {
            return Err(ReefShaderError::Message(format!(
                "Pixel buffer length mismatch: got {}, expected {} ({}x{} RGBA16)",
                pixels.len(),
                expected_len,
                width,
                height
            )));
        }
        if width == 0 || height == 0 {
            return Ok(());
        }

        match &mut self.backend {
            PipelineBackend::Gpu(engine) => {
                engine.process_rgba16(pixels, width, height, config, &mut self.state)
            }
            PipelineBackend::Cpu => {
                process_rgba16_cpu(pixels, width, height, config, &mut self.state);
                Ok(())
            }
        }
    }
}

fn quantile_from_hist(hist: &[u32], total: u32, q: f32) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let target = ((total as f64 - 1.0) * f64::from(q.clamp(0.0, 1.0))).round() as u32;
    let mut cumulative = 0u32;
    for (bin, &count) in hist.iter().enumerate() {
        cumulative = cumulative.saturating_add(count);
        if cumulative > target {
            return bin as f32 / (HIST_BINS - 1) as f32;
        }
    }
    1.0
}

fn compute_mins_and_gains(
    hist: &[u32; HIST_LEN],
    total_samples: u32,
    cfg: &ColourNormConfig,
    state: &mut ColourState,
) -> ([f32; 3], [f32; 3]) {
    let mut curr_mins = [0.0f32; 3];
    let mut curr_maxs = [1.0f32; 3];

    for c in 0..3 {
        let channel_hist = &hist[c * HIST_BINS..(c + 1) * HIST_BINS];
        curr_mins[c] = quantile_from_hist(channel_hist, total_samples, cfg.quantile_low);
        curr_maxs[c] = quantile_from_hist(channel_hist, total_samples, cfg.quantile_high);
    }

    let smoothing = cfg.temporal_smoothing.clamp(0.0, 0.9999);
    let alpha = 1.0 - smoothing;
    if let (Some(prev_mins), Some(prev_maxs)) = (state.prev_mins, state.prev_maxs) {
        for c in 0..3 {
            curr_mins[c] = curr_mins[c] * alpha + prev_mins[c] * smoothing;
            curr_maxs[c] = curr_maxs[c] * alpha + prev_maxs[c] * smoothing;
        }
    }
    state.prev_mins = Some(curr_mins);
    state.prev_maxs = Some(curr_maxs);

    let span = (cfg.whitepoint - cfg.blackpoint).max(0.0);
    let mut gains = [1.0f32; 3];
    for c in 0..3 {
        let range = (curr_maxs[c] - curr_mins[c]).max(1e-6);
        gains[c] = (span / range).min(cfg.max_gain);
    }

    (curr_mins, gains)
}

pub fn process_rgba16_cpu(
    pixels: &mut [u16],
    width: u32,
    height: u32,
    config: &ProcessingConfig,
    state: &mut ColourState,
) {
    let w = width as usize;
    let h = height as usize;
    let norm = &config.colour_norm;

    let (mins, gains) = if norm.enabled {
        let win = norm.downsample_win.max(1) as usize;
        let ds_w = w.div_ceil(win);
        let ds_h = h.div_ceil(win);
        let mut hist = [0u32; HIST_LEN];
        let block_area = (win * win) as f32;
        let inv_65535 = 1.0 / 65535.0;
        let bin_scale = (HIST_BINS - 1) as f32;

        for by in 0..ds_h {
            let base_y = by * win;
            for bx in 0..ds_w {
                let base_x = bx * win;
                let mut sum = [0.0f32; 3];
                for dy in 0..win {
                    let cy = (base_y + dy).min(h - 1);
                    for dx in 0..win {
                        let cx = (base_x + dx).min(w - 1);
                        let off = (cy * w + cx) * 4;
                        sum[0] += pixels[off] as f32 * inv_65535;
                        sum[1] += pixels[off + 1] as f32 * inv_65535;
                        sum[2] += pixels[off + 2] as f32 * inv_65535;
                    }
                }
                for c in 0..3 {
                    let avg = (sum[c] / block_area).clamp(0.0, 1.0);
                    let bin = (avg * bin_scale).round() as usize;
                    hist[c * HIST_BINS + bin.min(HIST_BINS - 1)] += 1;
                }
            }
        }

        compute_mins_and_gains(&hist, (ds_w * ds_h) as u32, norm, state)
    } else {
        ([0.0; 3], [1.0; 3])
    };

    let inv_65535 = 1.0 / 65535.0;
    let half_w = w / 2;
    let gamma = config.gamma.gamma;
    let gamma_enabled = config.gamma.enabled;
    let norm_enabled = norm.enabled;
    let blackpoint = norm.blackpoint;
    let side_by_side = config.output.side_by_side;

    for y in 0..h {
        let row_off = y * w * 4;
        for x in 0..w {
            if side_by_side && x >= half_w {
                continue;
            }
            let off = row_off + x * 4;
            for c in 0..3 {
                let mut v = pixels[off + c] as f32 * inv_65535;
                if norm_enabled {
                    v = ((v - mins[c]) * gains[c] + blackpoint).clamp(0.0, 1.0);
                }
                if gamma_enabled {
                    v = v.max(0.0).powf(gamma);
                }
                pixels[off + c] = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
            }
        }
    }
}

struct GpuBuffers {
    width: u32,
    height: u32,
    in_pixels: wgpu::Buffer,
    out_pixels: wgpu::Buffer,
    out_staging: wgpu::Buffer,
    hist_buf: wgpu::Buffer,
    hist_staging: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct GpuColourEngine {
    ctx: GpuContext,
    hist_pipeline: wgpu::ComputePipeline,
    apply_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    buffers: Option<GpuBuffers>,
}

impl GpuColourEngine {
    fn new(ctx: GpuContext) -> Result<Self> {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("reefshader-colour-correct"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/colour_correct.wgsl").into(),
                ),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("colour-correct-bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("colour-correct-layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let hist_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("colour-histogram-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("histogram_pass"),
                compilation_options: Default::default(),
                cache: None,
            });

        let apply_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("colour-apply-pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("apply_pass"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Ok(Self {
            ctx,
            hist_pipeline,
            apply_pipeline,
            bind_group_layout,
            buffers: None,
        })
    }

    fn ensure_buffers(&mut self, width: u32, height: u32) -> &GpuBuffers {
        let needs_alloc = match &self.buffers {
            Some(b) => b.width != width || b.height != height,
            None => true,
        };

        if needs_alloc {
            let pixel_bytes = (width as u64) * (height as u64) * 8;
            let hist_bytes = (HIST_LEN * std::mem::size_of::<u32>()) as u64;
            let params_bytes = std::mem::size_of::<ShaderParams>() as u64;

            let in_pixels = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("in-pixels"),
                size: pixel_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let out_pixels = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("out-pixels"),
                size: pixel_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let out_staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("out-staging"),
                size: pixel_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let hist_buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hist-buf"),
                size: hist_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let hist_staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hist-staging"),
                size: hist_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let params_buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("colour-params"),
                size: params_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("colour-correct-bg"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: in_pixels.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_pixels.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: hist_buf.as_entire_binding(),
                        },
                    ],
                });

            self.buffers = Some(GpuBuffers {
                width,
                height,
                in_pixels,
                out_pixels,
                out_staging,
                hist_buf,
                hist_staging,
                params_buf,
                bind_group,
            });
        }

        self.buffers.as_ref().unwrap()
    }

    fn read_buffer<T: Pod>(&self, staging: &wgpu::Buffer, dst: &mut [T]) -> Result<()> {
        let slice = staging.slice(..);
        let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ready_cb = Arc::clone(&ready);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                ready_cb.store(true, std::sync::atomic::Ordering::Release);
            }
        });
        let _ = self.ctx.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        if !ready.load(std::sync::atomic::Ordering::Acquire) {
            return Err(ReefShaderError::Message(
                "Failed to map GPU staging buffer".to_string(),
            ));
        }
        {
            let data = slice.get_mapped_range().map_err(|e| {
                ReefShaderError::Message(format!("Failed to get mapped GPU buffer range: {e}"))
            })?;
            let src: &[T] = bytemuck::cast_slice(&data);
            dst.copy_from_slice(src);
        }
        staging.unmap();
        Ok(())
    }

    fn process_rgba16(
        &mut self,
        pixels: &mut [u16],
        width: u32,
        height: u32,
        config: &ProcessingConfig,
        state: &mut ColourState,
    ) -> Result<()> {
        let win = config.colour_norm.downsample_win.max(1);
        let ds_width = width.div_ceil(win);
        let ds_height = height.div_ceil(win);

        let mut params = ShaderParams {
            width,
            height,
            downsample_win: win,
            ds_width,
            ds_height,
            colour_norm_enabled: u32::from(config.colour_norm.enabled),
            gamma_enabled: u32::from(config.gamma.enabled),
            side_by_side: u32::from(config.output.side_by_side),
            gamma: config.gamma.gamma,
            blackpoint: config.colour_norm.blackpoint,
            _pad0: [0.0; 2],
            mins: [0.0; 4],
            gains: [1.0; 4],
        };

        let _ = self.ensure_buffers(width, height);
        let bufs = self.buffers.as_ref().unwrap();

        self.ctx
            .queue
            .write_buffer(&bufs.in_pixels, 0, bytemuck::cast_slice(pixels));

        if config.colour_norm.enabled {
            self.ctx
                .queue
                .write_buffer(&bufs.params_buf, 0, bytemuck::bytes_of(&params));

            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("hist-encoder"),
                    });
            encoder.clear_buffer(&bufs.hist_buf, 0, None);
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hist-pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.hist_pipeline);
                cpass.set_bind_group(0, &bufs.bind_group, &[]);
                cpass.dispatch_workgroups(ds_width.div_ceil(16), ds_height.div_ceil(16), 1);
            }
            encoder.copy_buffer_to_buffer(
                &bufs.hist_buf,
                0,
                &bufs.hist_staging,
                0,
                (HIST_LEN * std::mem::size_of::<u32>()) as u64,
            );
            self.ctx.queue.submit(Some(encoder.finish()));

            let mut hist = [0u32; HIST_LEN];
            self.read_buffer(&bufs.hist_staging, &mut hist)?;
            let (mins, gains) =
                compute_mins_and_gains(&hist, ds_width * ds_height, &config.colour_norm, state);
            params.mins = [mins[0], mins[1], mins[2], 0.0];
            params.gains = [gains[0], gains[1], gains[2], 1.0];
        }

        self.ctx
            .queue
            .write_buffer(&bufs.params_buf, 0, bytemuck::bytes_of(&params));

        let mut encoder =
            self.ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("apply-encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apply-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.apply_pipeline);
            cpass.set_bind_group(0, &bufs.bind_group, &[]);
            cpass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
        }
        encoder.copy_buffer_to_buffer(
            &bufs.out_pixels,
            0,
            &bufs.out_staging,
            0,
            (width as u64) * (height as u64) * 8,
        );
        self.ctx.queue.submit(Some(encoder.finish()));

        self.read_buffer(&bufs.out_staging, pixels)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_colour_norm_recovers_underwater_red_and_preserves_sbs() {
        let w = 16u32;
        let h = 16u32;
        let mut pixels = vec![0u16; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let t = x as f32 / (w - 1) as f32;
                let off = ((y * w + x) * 4) as usize;
                // Simulated underwater tint: very weak red [0.02..0.10], strong green/blue.
                pixels[off] = ((0.02 + 0.08 * t) * 65535.0) as u16;
                pixels[off + 1] = ((0.20 + 0.50 * t) * 65535.0) as u16;
                pixels[off + 2] = ((0.30 + 0.60 * t) * 65535.0) as u16;
                pixels[off + 3] = 65535;
            }
        }
        let original = pixels.clone();

        let mut cfg = ProcessingConfig::default();
        cfg.force_cpu = true;
        cfg.gamma.enabled = false;
        cfg.output.side_by_side = true;
        cfg.colour_norm.downsample_win = 1;

        let mut pipeline = ColourPipeline::new(true);
        pipeline.process_rgba16(&mut pixels, w, h, &cfg).unwrap();

        // Left half should be stretched: red gain is clamped at max_gain (10.0), boosting red.
        let left_bright_off = (7 * 4) as usize;
        assert!(pixels[left_bright_off] > original[left_bright_off] * 4);

        // Right half (x >= w / 2) should be untouched in side-by-side mode.
        let right_off = (8 * 4) as usize;
        assert_eq!(
            &pixels[right_off..right_off + 4],
            &original[right_off..right_off + 4]
        );
    }

    #[test]
    fn temporal_smoothing_interpolates_gains() {
        let mut cfg = ColourNormConfig::default();
        cfg.temporal_smoothing = 0.5;
        let mut state = ColourState::default();

        let mut hist1 = [0u32; HIST_LEN];
        hist1[100] = 10;
        hist1[1000] = 10;
        hist1[HIST_BINS + 100] = 20;
        hist1[2 * HIST_BINS + 100] = 20;
        let (mins1, _) = compute_mins_and_gains(&hist1, 20, &cfg, &mut state);

        let mut hist2 = [0u32; HIST_LEN];
        hist2[2000] = 20;
        hist2[HIST_BINS + 2000] = 20;
        hist2[2 * HIST_BINS + 2000] = 20;
        let (mins2, _) = compute_mins_and_gains(&hist2, 20, &cfg, &mut state);

        assert!(mins2[0] > mins1[0]);
        assert!(mins2[0] < (2000.0 / 4095.0));
    }
}
