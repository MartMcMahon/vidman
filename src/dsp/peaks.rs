// Top-K peak detector over an fftshifted dB spectrum, with hysteresis,
// hang-time, and stable per-station IDs across frames.

#[derive(Clone, Debug)]
pub struct Station {
    pub id: u32,
    pub freq_hz: f64,
    pub power_db: f32,
    pub opened_at_frame: u32,
}

#[derive(Clone, Debug)]
pub enum Event {
    Open(Station),
    Close { id: u32, freq_hz: f64 },
}

pub struct PeakDetector {
    open_threshold_db: f32,
    close_threshold_db: f32,
    hang_frames: u32,
    min_separation_bins: usize,
    dc_skip_bins: usize, // half-width of center exclusion zone
    sample_rate_hz: f32,
    fft_size: usize,

    open: Vec<OpenStation>,
    next_id: u32,
    frame: u32,
}

struct OpenStation {
    id: u32,
    bin: usize,
    hang: u32, // frames since last seen above close threshold
    last_power_db: f32,
    opened_at_frame: u32,
}

pub struct Params {
    pub sample_rate_hz: f32,
    pub fft_size: usize,
    pub open_threshold_db: f32,
    pub close_threshold_db: f32,
    pub hang_seconds: f32,
    pub rows_per_second: f32,
    pub min_separation_hz: f32,
    pub dc_skip_hz: f32,
}

impl PeakDetector {
    pub fn new(p: Params) -> Self {
        let bin_hz = p.sample_rate_hz / p.fft_size as f32;
        let min_separation_bins = (p.min_separation_hz / bin_hz).ceil().max(1.0) as usize;
        let dc_skip_bins = (p.dc_skip_hz / bin_hz).ceil() as usize;
        let hang_frames = (p.hang_seconds * p.rows_per_second).ceil().max(1.0) as u32;
        Self {
            open_threshold_db: p.open_threshold_db,
            close_threshold_db: p.close_threshold_db,
            hang_frames,
            min_separation_bins,
            dc_skip_bins,
            sample_rate_hz: p.sample_rate_hz,
            fft_size: p.fft_size,
            open: Vec::new(),
            next_id: 1,
            frame: 0,
        }
    }

    /// `spectrum`: one fftshifted dB row (e.g. from `Spectrum::take()`).
    /// `center_hz`: the radio's *actual* tuned center at capture time
    /// (i.e. include the +SHIFT_HZ offset already; this is what bin = fft_size/2 maps to).
    /// `events`: appended-to.
    pub fn step(
        &mut self,
        spectrum: &[f32],
        center_hz: f64,
        events: &mut Vec<Event>,
    ) {
        self.frame += 1;
        debug_assert_eq!(spectrum.len(), self.fft_size);

        let floor = median(spectrum);
        let open_db = floor + self.open_threshold_db;
        let close_db = floor + self.close_threshold_db;
        let mid = self.fft_size / 2;
        let dc_lo = mid.saturating_sub(self.dc_skip_bins);
        let dc_hi = (mid + self.dc_skip_bins + 1).min(self.fft_size);

        // 1) refresh / age out existing stations
        let frame = self.frame;
        let sep = self.min_separation_bins;
        let hang_limit = self.hang_frames;
        let n = spectrum.len();

        let mut closed_ids: Vec<(u32, f64)> = Vec::new();
        self.open.retain_mut(|st| {
            let lo = st.bin.saturating_sub(sep);
            let hi = (st.bin + sep + 1).min(n);
            let (new_bin, peak_db) = argmax(&spectrum[lo..hi])
                .map(|(i, v)| (lo + i, v))
                .unwrap_or((st.bin, f32::NEG_INFINITY));
            st.bin = new_bin;
            st.last_power_db = peak_db;
            if peak_db < close_db {
                st.hang += 1;
            } else {
                st.hang = 0;
            }
            let alive = st.hang < hang_limit;
            if !alive {
                let freq = bin_to_freq(st.bin, self.fft_size, self.sample_rate_hz, center_hz);
                closed_ids.push((st.id, freq));
            }
            alive
        });
        for (id, freq_hz) in closed_ids {
            events.push(Event::Close { id, freq_hz });
        }

        // 2) scan for new peaks above open threshold, skipping DC zone and existing stations
        for k in 1..n - 1 {
            if k >= dc_lo && k < dc_hi {
                continue;
            }
            let v = spectrum[k];
            if v < open_db {
                continue;
            }
            // strict local maximum: >= both neighbors, > at least one
            let l = spectrum[k - 1];
            let r = spectrum[k + 1];
            if !(v >= l && v >= r && v > l.min(r)) {
                continue;
            }
            if self.open.iter().any(|st| st.bin.abs_diff(k) < sep) {
                continue;
            }
            let id = self.next_id;
            self.next_id += 1;
            self.open.push(OpenStation {
                id,
                bin: k,
                hang: 0,
                last_power_db: v,
                opened_at_frame: frame,
            });
            events.push(Event::Open(Station {
                id,
                freq_hz: bin_to_freq(k, self.fft_size, self.sample_rate_hz, center_hz),
                power_db: v,
                opened_at_frame: frame,
            }));
        }
    }

    /// Drop all currently-open stations. Use after the radio retunes — the
    /// detector's stored bin numbers now point at different absolute Hz, and
    /// keeping them would briefly report wrong frequencies until they age out.
    pub fn reset(&mut self) {
        self.open.clear();
    }

    pub fn snapshot(&self, center_hz: f64) -> Vec<Station> {
        self.open
            .iter()
            .map(|st| Station {
                id: st.id,
                freq_hz: bin_to_freq(st.bin, self.fft_size, self.sample_rate_hz, center_hz),
                power_db: st.last_power_db,
                opened_at_frame: st.opened_at_frame,
            })
            .collect()
    }
}

fn bin_to_freq(bin: usize, fft_size: usize, sample_rate_hz: f32, center_hz: f64) -> f64 {
    let offset = bin as f32 - (fft_size as f32 / 2.0);
    center_hz + (offset * sample_rate_hz / fft_size as f32) as f64
}

fn argmax(s: &[f32]) -> Option<(usize, f32)> {
    s.iter()
        .enumerate()
        .fold(None, |acc, (i, &v)| match acc {
            None => Some((i, v)),
            Some((_, best)) if v > best => Some((i, v)),
            x => x,
        })
}

fn median(s: &[f32]) -> f32 {
    let mut buf: Vec<f32> = s.to_vec();
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    buf[buf.len() / 2]
}
