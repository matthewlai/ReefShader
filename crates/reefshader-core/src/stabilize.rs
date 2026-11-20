use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use gyroflow_core::StabilizationManager;
use gyroflow_core::gyro_source::FileLoadOptions;

use crate::config::GyroflowConfig;
use crate::error::Result;

/// Wrapper around `gyroflow_core::StabilizationManager` configured for underwater refraction.
pub struct Stabilizer {
    manager: StabilizationManager,
}

impl Stabilizer {
    pub fn new(config: &GyroflowConfig) -> Self {
        let manager = StabilizationManager::default();
        let refraction = if config.underwater {
            config.light_refraction_coefficient
        } else {
            1.0
        };
        manager.params.write().light_refraction_coefficient = refraction;
        Self { manager }
    }

    pub fn load_video_metadata(
        &self,
        video_path: &Path,
        duration_ms: f64,
        fps: f64,
        frame_count: usize,
        size: (usize, usize),
    ) -> Result<()> {
        self.manager
            .init_from_video_data(duration_ms, fps, frame_count, size);

        let mut file = File::open(video_path)?;
        let filesize = file.metadata()?.len();
        let url = video_path.to_string_lossy();
        let cancel = Arc::new(AtomicBool::new(false));
        let options = FileLoadOptions::default();

        self.manager
            .load_gyro_data(&mut file, filesize, &url, true, &options, |_| {}, cancel)?;
        self.manager.recompute_blocking();

        Ok(())
    }

    pub fn inner(&self) -> &StabilizationManager {
        &self.manager
    }
}
