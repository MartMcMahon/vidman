/// Maps station frequencies to hallway-meter positions, computes per-station
/// (gain, pan) given a listener position, and produces interleaved stereo.
pub struct SpatialMixer {
    pub camera_x: f32,     // listener position in hallway meters
    pub hz_per_meter: f32, // freq <-> world conversion (e.g. 50_000)
    pub center_hz: f64,    // freq at hallway x = 0
    pub pan_full_m: f32,   // distance at which pan saturates to ±1 (e.g. 5.0)
    pub dist_k: f32,       // distance falloff constant in 1/(1+k·d²) (e.g. 0.25)
    pub dist_floor_m: f32, // minimum distance to avoid 0-div blasts (e.g. 0.5)
}

impl SpatialMixer {
    pub fn new(center_hz: f64) -> Self {
        Self {
            camera_x: 0.0,
            hz_per_meter: 50_000.0,
            center_hz,
            pan_full_m: 5.0,
            dist_k: 1.0,
            dist_floor_m: 0.5,
        }
    }

    pub fn station_x(&self, freq_hz: f64) -> f32 {
        ((freq_hz - self.center_hz) / self.hz_per_meter as f64) as f32
    }

    /// (left_gain, right_gain) for a station given the current camera_x.
    /// Currently mono: L = R = distance-based gain. No stereo pan — position
    /// in the hallway maps purely to volume, not to L/R balance.
    pub fn channel_gains(&self, freq_hz: f64) -> (f32, f32) {
        let dx = self.station_x(freq_hz) - self.camera_x;
        let dist = dx.abs().max(self.dist_floor_m);
        let g = (1.0 / (1.0 + self.dist_k * dist * dist)).min(1.0);
        (g, g)
    }

    /// Pick the single station nearest the listener and write its audio
    /// (scaled by distance gain) into interleaved stereo. All other streams
    /// are muted — strictly one station at a time. Returns the winner's
    /// freq_hz so callers can log/visualize what's playing.
    pub fn mix<'a, I>(&self, streams: I, stereo_out: &mut Vec<f32>) -> Option<f64>
    where
        I: IntoIterator<Item = (&'a [f32], f64)>,
    {
        let collected: Vec<(&'a [f32], f64)> = streams.into_iter().collect();
        stereo_out.clear();
        let (winner_samples, winner_freq) = collected
            .iter()
            .min_by(|(_, fa), (_, fb)| {
                let da = (self.station_x(*fa) - self.camera_x).abs();
                let db = (self.station_x(*fb) - self.camera_x).abs();
                da.partial_cmp(&db).unwrap()
            })
            .copied()?;
        let (lg, rg) = self.channel_gains(winner_freq);
        stereo_out.reserve(winner_samples.len() * 2);
        for &v in winner_samples {
            stereo_out.push(v * lg);
            stereo_out.push(v * rg);
        }
        Some(winner_freq)
    }
}
