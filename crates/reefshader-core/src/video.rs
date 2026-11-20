use std::path::Path;
use std::time::Instant;

use crate::colour::{ColourBackend, ColourPipeline};
use crate::config::ProcessingConfig;
use crate::error::{ReefShaderError, Result};

#[derive(Debug, Clone)]
pub struct VideoMetadata {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration_ms: f64,
    pub frame_count: usize,
}

#[derive(Debug, Clone)]
pub struct ProcessSummary {
    pub frames_processed: usize,
    pub width: u32,
    pub height: u32,
    pub encoder_name: String,
    pub colour_backend: ColourBackend,
    pub elapsed_secs: f64,
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
            ReefShaderError::Message(format!("No video stream found in {}", path.display()))
        })?;

    let rate = stream.avg_frame_rate();
    let fps = if rate.denominator() != 0 && rate.numerator() != 0 {
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

#[cfg(feature = "ffmpeg")]
fn compute_output_dimensions(src_w: u32, src_h: u32, config: &ProcessingConfig) -> (u32, u32) {
    let (w, h) = if config.scaling.enabled {
        if let Some(target_w) = config.scaling.width.filter(|&w| w > 0) {
            let ratio = f64::from(target_w) / f64::from(src_w.max(1));
            let target_h = (f64::from(src_h) * ratio).round() as u32;
            (target_w, target_h)
        } else {
            (src_w, src_h)
        }
    } else {
        (src_w, src_h)
    };
    // YUV420P requires even dimensions.
    let even_w = ((w.max(2)) + 1) & !1;
    let even_h = ((h.max(2)) + 1) & !1;
    (even_w, even_h)
}

#[cfg(feature = "ffmpeg")]
fn open_video_encoder(
    requested: Option<&str>,
    width: u32,
    height: u32,
    fps_rational: ffmpeg_next::Rational,
    enc_time_base: ffmpeg_next::Rational,
    bitrate_mbps: u32,
    global_header: bool,
) -> Result<(String, ffmpeg_next::encoder::Video)> {
    let candidates: Vec<&str> = if let Some(name) = requested {
        vec![name]
    } else if cfg!(target_os = "macos") {
        vec![
            "hevc_videotoolbox",
            "h264_videotoolbox",
            "libx265",
            "libx264",
            "mpeg4",
        ]
    } else {
        vec!["libx265", "libx264", "libvpx-vp9", "mpeg4"]
    };

    let mut last_err: Option<String> = None;
    for name in candidates {
        let Some(codec) = ffmpeg_next::encoder::find_by_name(name) else {
            last_err = Some(format!("Encoder '{name}' not found in FFmpeg"));
            continue;
        };

        let ctx = ffmpeg_next::codec::context::Context::new_with_codec(codec);
        let mut video = match ctx.encoder().video() {
            Ok(v) => v,
            Err(e) => {
                last_err = Some(format!("Failed to create video encoder '{name}': {e}"));
                continue;
            }
        };

        video.set_width(width);
        video.set_height(height);
        video.set_format(ffmpeg_next::format::Pixel::YUV420P);
        video.set_frame_rate(Some(fps_rational));
        video.set_time_base(enc_time_base);
        video.set_bit_rate((bitrate_mbps as usize) * 1_000_000);
        if global_header {
            video.set_flags(ffmpeg_next::codec::Flags::GLOBAL_HEADER);
        }

        let mut opts = ffmpeg_next::Dictionary::new();
        if name == "libx264" || name == "libx265" {
            opts.set("preset", "medium");
        }
        if name == "libx265" {
            opts.set("x265-params", "log-level=error");
        }

        match video.open_with(opts) {
            Ok(opened) => return Ok((name.to_string(), opened)),
            Err(e) => {
                last_err = Some(format!("Failed to open encoder '{name}': {e}"));
            }
        }
    }

    Err(ReefShaderError::Message(last_err.unwrap_or_else(|| {
        "No usable video encoder found".to_string()
    })))
}

#[cfg(feature = "ffmpeg")]
fn copy_frame_to_rgba16_buf(
    frame: &ffmpeg_next::frame::Video,
    width: u32,
    height: u32,
    dst: &mut Vec<u16>,
) {
    let w = width as usize;
    let h = height as usize;
    let row_u16s = w * 4;
    dst.resize(h * row_u16s, 0);
    let stride = frame.stride(0);
    let src_bytes = frame.data(0);
    for y in 0..h {
        let src_row = &src_bytes[y * stride..y * stride + row_u16s * 2];
        let dst_row = &mut dst[y * row_u16s..(y + 1) * row_u16s];
        bytemuck::cast_slice_mut(dst_row).copy_from_slice(src_row);
    }
}

#[cfg(feature = "ffmpeg")]
fn copy_rgba16_buf_to_frame(
    src: &[u16],
    width: u32,
    height: u32,
    frame: &mut ffmpeg_next::frame::Video,
) {
    let w = width as usize;
    let h = height as usize;
    let row_u16s = w * 4;
    let stride = frame.stride(0);
    let dst_bytes = frame.data_mut(0);
    for y in 0..h {
        let src_row = &src[y * row_u16s..(y + 1) * row_u16s];
        let dst_row = &mut dst_bytes[y * stride..y * stride + row_u16s * 2];
        dst_row.copy_from_slice(bytemuck::cast_slice(src_row));
    }
}

#[cfg(feature = "ffmpeg")]
pub fn transcode_video(
    input: &Path,
    output: &Path,
    config: &ProcessingConfig,
    pipeline: &mut ColourPipeline,
) -> Result<ProcessSummary> {
    use ffmpeg_next::{Packet, Rational, Rescale, codec, encoder, format, frame, media, picture, software::scaling};

    let start = Instant::now();
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }

    let mut ictx = format::input(input)?;
    let mut octx = format::output(output)?;
    let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

    let best_video_idx = ictx
        .streams()
        .best(media::Type::Video)
        .map(|s| s.index())
        .ok_or_else(|| {
            ReefShaderError::Message(format!("No video stream found in {}", input.display()))
        })?;

    let mut stream_mapping: Vec<isize> = vec![-1; ictx.nb_streams() as usize];
    let mut ist_time_bases: Vec<Rational> = vec![Rational(0, 1); ictx.nb_streams() as usize];
    let mut ost_time_bases: Vec<Rational> = Vec::new();

    let mut decoder: Option<ffmpeg_next::decoder::Video> = None;
    let mut encoder: Option<ffmpeg_next::encoder::Video> = None;
    let mut encoder_name = String::new();
    let mut out_video_idx: usize = 0;
    let mut in_video_tb = Rational(1, 30);
    let mut enc_time_base = Rational(1, 30);
    let mut out_w = 0u32;
    let mut out_h = 0u32;

    let mut ost_index = 0usize;
    for (ist_idx, ist) in ictx.streams().enumerate() {
        ist_time_bases[ist_idx] = ist.time_base();
        let medium = ist.parameters().medium();

        if ist_idx == best_video_idx {
            let dec = codec::context::Context::from_parameters(ist.parameters())?
                .decoder()
                .video()?;
            let (tw, th) = compute_output_dimensions(dec.width(), dec.height(), config);
            out_w = tw;
            out_h = th;

            let avg_rate = ist.avg_frame_rate();
            let fps_rational = if avg_rate.numerator() > 0 && avg_rate.denominator() > 0 {
                avg_rate
            } else {
                Rational(30, 1)
            };
            enc_time_base = fps_rational.invert();
            in_video_tb = ist.time_base();

            let (name, opened_enc) = open_video_encoder(
                config.encode.codec.as_deref(),
                out_w,
                out_h,
                fps_rational,
                enc_time_base,
                config.encode.bitrate_mbps,
                global_header,
            )?;

            let mut ost = octx.add_stream(encoder::find_by_name(&name))?;
            ost.set_parameters(&opened_enc);
            if name.contains("hevc") || name.contains("265") {
                unsafe {
                    (*ost.parameters().as_mut_ptr()).codec_tag = u32::from_le_bytes(*b"hvc1");
                }
            }

            stream_mapping[ist_idx] = ost_index as isize;
            out_video_idx = ost_index;
            ost_index += 1;
            decoder = Some(dec);
            encoder = Some(opened_enc);
            encoder_name = name;
        } else if medium == media::Type::Audio {
            let mut ost = octx.add_stream(encoder::find(codec::Id::None))?;
            ost.set_parameters(ist.parameters());
            unsafe {
                (*ost.parameters().as_mut_ptr()).codec_tag = 0;
            }
            stream_mapping[ist_idx] = ost_index as isize;
            ost_index += 1;
        }
    }

    let mut decoder = decoder.expect("video decoder initialized");
    let mut encoder = encoder.expect("video encoder initialized");

    octx.set_metadata(ictx.metadata().to_owned());
    let ext = output
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "mp4" || ext == "mov" || ext == "m4v" {
        let mut mux_opts = ffmpeg_next::Dictionary::new();
        mux_opts.set("movflags", "faststart");
        octx.write_header_with(mux_opts)?;
    } else {
        octx.write_header()?;
    }

    for ost in octx.streams() {
        ost_time_bases.push(ost.time_base());
    }
    let out_video_tb = ost_time_bases[out_video_idx];

    let mut to_rgba_scaler: Option<scaling::Context> = None;
    let mut to_rgba_src: Option<(ffmpeg_next::format::Pixel, u32, u32)> = None;
    let mut to_yuv_scaler = scaling::Context::get(
        ffmpeg_next::format::Pixel::RGBA64LE,
        out_w,
        out_h,
        ffmpeg_next::format::Pixel::YUV420P,
        out_w,
        out_h,
        scaling::Flags::BILINEAR,
    )?;

    let mut decoded = frame::Video::empty();
    let mut rgba_frame = frame::Video::new(ffmpeg_next::format::Pixel::RGBA64LE, out_w, out_h);
    let mut yuv_frame = frame::Video::new(ffmpeg_next::format::Pixel::YUV420P, out_w, out_h);
    let mut rgba_buf: Vec<u16> = Vec::with_capacity((out_w as usize) * (out_h as usize) * 4);

    let mut frames_processed = 0usize;
    let mut last_pts: Option<i64> = None;

    let drain_encoded = |enc: &mut ffmpeg_next::encoder::Video,
                         octx: &mut format::context::Output|
     -> Result<()> {
        let mut encoded = Packet::empty();
        while enc.receive_packet(&mut encoded).is_ok() {
            encoded.set_stream(out_video_idx);
            encoded.rescale_ts(enc_time_base, out_video_tb);
            encoded.write_interleaved(octx)?;
        }
        Ok(())
    };

    let mut process_decoded = |dec: &mut ffmpeg_next::decoder::Video,
                               enc: &mut ffmpeg_next::encoder::Video,
                               octx: &mut format::context::Output|
     -> Result<()> {
        while dec.receive_frame(&mut decoded).is_ok() {
            let src_fmt = decoded.format();
            let src_w = decoded.width();
            let src_h = decoded.height();
            if to_rgba_src != Some((src_fmt, src_w, src_h)) {
                to_rgba_scaler = Some(scaling::Context::get(
                    src_fmt,
                    src_w,
                    src_h,
                    ffmpeg_next::format::Pixel::RGBA64LE,
                    out_w,
                    out_h,
                    scaling::Flags::BILINEAR,
                )?);
                to_rgba_src = Some((src_fmt, src_w, src_h));
            }

            to_rgba_scaler
                .as_mut()
                .unwrap()
                .run(&decoded, &mut rgba_frame)?;

            copy_frame_to_rgba16_buf(&rgba_frame, out_w, out_h, &mut rgba_buf);
            pipeline.process_rgba16(&mut rgba_buf, out_w, out_h, config)?;
            copy_rgba16_buf_to_frame(&rgba_buf, out_w, out_h, &mut rgba_frame);

            to_yuv_scaler.run(&rgba_frame, &mut yuv_frame)?;

            let mut pts = decoded
                .timestamp()
                .map(|ts| ts.rescale(in_video_tb, enc_time_base))
                .unwrap_or(frames_processed as i64);
            if let Some(prev) = last_pts {
                if pts <= prev {
                    pts = prev + 1;
                }
            }
            last_pts = Some(pts);

            yuv_frame.set_pts(Some(pts));
            yuv_frame.set_kind(picture::Type::None);

            enc.send_frame(&yuv_frame)?;
            drain_encoded(enc, octx)?;

            frames_processed += 1;
            if frames_processed % 100 == 0 {
                log::info!("Processed {} frames...", frames_processed);
            }
        }
        Ok(())
    };

    for (stream, mut packet) in ictx.packets() {
        let ist_idx = stream.index();
        let ost_idx = stream_mapping[ist_idx];
        if ost_idx < 0 {
            continue;
        }
        if ist_idx == best_video_idx {
            decoder.send_packet(&packet)?;
            process_decoded(&mut decoder, &mut encoder, &mut octx)?;
        } else {
            let ost_tb = ost_time_bases[ost_idx as usize];
            packet.rescale_ts(ist_time_bases[ist_idx], ost_tb);
            packet.set_position(-1);
            packet.set_stream(ost_idx as usize);
            packet.write_interleaved(&mut octx)?;
        }
    }

    decoder.send_eof()?;
    process_decoded(&mut decoder, &mut encoder, &mut octx)?;

    encoder.send_eof()?;
    drain_encoded(&mut encoder, &mut octx)?;

    octx.write_trailer()?;

    Ok(ProcessSummary {
        frames_processed,
        width: out_w,
        height: out_h,
        encoder_name,
        colour_backend: pipeline.backend(),
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}

#[cfg(not(feature = "ffmpeg"))]
pub fn init_ffmpeg() -> Result<()> {
    Ok(())
}

#[cfg(not(feature = "ffmpeg"))]
pub fn probe_video(_path: &Path) -> Result<VideoMetadata> {
    Err(ReefShaderError::Message(
        "Built without ffmpeg feature".to_string(),
    ))
}

#[cfg(not(feature = "ffmpeg"))]
pub fn transcode_video(
    _input: &Path,
    _output: &Path,
    _config: &ProcessingConfig,
    _pipeline: &mut ColourPipeline,
) -> Result<ProcessSummary> {
    Err(ReefShaderError::Message(
        "Built without ffmpeg feature".to_string(),
    ))
}
