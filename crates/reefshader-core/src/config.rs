use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub scaling: ScalingConfig,
    pub gamma: GammaConfig,
    pub colour_norm: ColourNormConfig,
    pub gyroflow: GyroflowConfig,
    pub output: OutputConfig,
    pub encode: EncodeConfig,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            scaling: ScalingConfig::default(),
            gamma: GammaConfig::default(),
            colour_norm: ColourNormConfig::default(),
            gyroflow: GyroflowConfig::default(),
            output: OutputConfig::default(),
            encode: EncodeConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    pub enabled: bool,
    /// Target output width in pixels (`None` preserves source resolution).
    pub width: Option<u32>,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            width: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaConfig {
    pub enabled: bool,
    pub gamma: f32,
}

impl Default for GammaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            gamma: 1.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColourNormConfig {
    pub enabled: bool,
    pub max_gain: f32,
    pub temporal_smoothing: f32,
    pub downsample_win: u32,
    pub quantile_low: f32,
    pub quantile_high: f32,
    pub blackpoint: f32,
    pub whitepoint: f32,
}

impl Default for ColourNormConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_gain: 10.0,
            temporal_smoothing: 0.95,
            downsample_win: 4,
            quantile_low: 0.0005,
            quantile_high: 0.995,
            blackpoint: 0.1,
            whitepoint: 0.9,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GyroflowConfig {
    pub enabled: bool,
    pub underwater: bool,
    /// Index of refraction relative to air (1.33 for water, 1.0 for air).
    pub light_refraction_coefficient: f64,
}

impl Default for GyroflowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            underwater: true,
            light_refraction_coefficient: 1.33,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputConfig {
    pub side_by_side: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeConfig {
    pub codec: String,
    pub bitrate_mbps: u32,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self {
            codec: default_codec().to_string(),
            bitrate_mbps: 20,
        }
    }
}

fn default_codec() -> &'static str {
    if cfg!(target_os = "macos") {
        "hevc_videotoolbox"
    } else {
        "libx265"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_underwater_defaults() {
        let cfg = ProcessingConfig::default();
        assert!(cfg.colour_norm.enabled);
        assert!(cfg.gamma.enabled);
        assert!(cfg.gyroflow.enabled);
        assert!(cfg.gyroflow.underwater);
        assert!((cfg.gyroflow.light_refraction_coefficient - 1.33).abs() < 1e-6);
        assert_eq!(cfg.encode.bitrate_mbps, 20);
    }
}
