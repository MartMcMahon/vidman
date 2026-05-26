use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use num_complex::Complex32;
use seify_hackrfone::{Config, HackRf};

use crate::dsp::peaks::{Params as PeakParams, PeakDetector};

pub const RADIO_SAMPLE_RATE_HZ: u32 = 2_400_000;
pub const SHIFT_HZ: u64 = 200_000;
pub const KMFA: u64 = 89_500_000;

pub const FFT_SIZE: usize = 1024;
pub const AVG_FFTS: usize = 64;
pub const ROWS_PER_SECOND: f32 =
    RADIO_SAMPLE_RATE_HZ as f32 / (FFT_SIZE as f32 * AVG_FFTS as f32);

const READ_BUF_BYTES: usize = 262_144;

#[derive(Clone, Copy)]
pub struct Gains {
    pub vga_db: u16,
    pub lna_db: u16,
}

impl Gains {
    /// Default for the wideband scanner / multi-station demod chains.
    pub const SCAN: Self = Self { vga_db: 16, lna_db: 16 };
    /// A few dB hotter for single-station listening (the `play` command).
    pub const PLAY: Self = Self { vga_db: 20, lna_db: 16 };
}

pub fn parse_freq(input: &str) -> u64 {
    let trimmed = input.trim();
    let (num_str, multiplier) = if let Some(rest) = trimmed.strip_suffix(['M', 'm']) {
        (rest, 1_000_000.0)
    } else if let Some(rest) = trimmed.strip_suffix(['K', 'k']) {
        (rest, 1_000.0)
    } else {
        (trimmed, 1.0)
    };
    let parsed: f64 = num_str.parse().expect("freq is a number");
    (parsed * multiplier) as u64
}

pub fn samples_to_iq(bytes: &[u8], iq: &mut Vec<Complex32>) {
    iq.clear();
    iq.extend(bytes.chunks_exact(2).map(|chunk| {
        Complex32::new(
            chunk[0] as i8 as f32 / 128.0,
            chunk[1] as i8 as f32 / 128.0,
        )
    }));
}

pub fn default_peak_detector(narrow: bool) -> PeakDetector {
    PeakDetector::new(PeakParams {
        sample_rate_hz: RADIO_SAMPLE_RATE_HZ as f32,
        fft_size: FFT_SIZE,
        open_threshold_db: 10.0,
        close_threshold_db: 6.0,
        hang_seconds: 0.5,
        rows_per_second: ROWS_PER_SECOND,
        min_separation_hz: if narrow { 12_500.0 } else { 150_000.0 },
        // skip the LO leakage zone plus a bit of slop; the tuned signal sits
        // SHIFT_HZ away from DC so it's not hidden by this exclusion
        dc_skip_hz: 50_000.0,
    })
}

/// Owns the HackRF, a stop-on-ctrl-c flag, and reusable read buffers.
/// `open()` opens the device + starts RX in one shot; `retune()` changes
/// frequency in-flight without restarting the stream.
pub struct Session {
    hackrf: HackRf,
    stop: Arc<AtomicBool>,
    /// LO frequency (the requested freq + SHIFT_HZ offset).
    pub tuned_to: u64,
    read_buf: Vec<u8>,
    iq: Vec<Complex32>,
}

impl Session {
    pub fn open(freq_hz: u64, gains: Gains) -> anyhow::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_handler = stop.clone();
        ctrlc::set_handler(move || {
            stop_handler.store(true, Ordering::Relaxed);
        })?;

        let hackrf = HackRf::open_first()?;
        let tuned_to = freq_hz + SHIFT_HZ;
        hackrf.start_rx(&Config {
            txvga_db: 0,
            vga_db: gains.vga_db,
            lna_db: gains.lna_db,
            amp_enable: false,
            antenna_enable: false,
            frequency_hz: tuned_to,
            sample_rate_hz: RADIO_SAMPLE_RATE_HZ,
            sample_rate_div: 1,
        })?;

        Ok(Self {
            hackrf,
            stop,
            tuned_to,
            read_buf: vec![0u8; READ_BUF_BYTES],
            iq: Vec::with_capacity(READ_BUF_BYTES / 2),
        })
    }

    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// One read + i8 IQ → Complex32. Returns `None` on a non-fatal read error
    /// (after logging it) so the caller can simply skip the block.
    pub fn read_block(&mut self) -> Option<&[Complex32]> {
        match self.hackrf.read(&mut self.read_buf) {
            Ok(n) => {
                samples_to_iq(&self.read_buf[..n], &mut self.iq);
                Some(&self.iq)
            }
            Err(e) => {
                eprintln!("hackrf read error: {e:#}");
                None
            }
        }
    }

    /// Change LO frequency while RX is running. The arg is the *target* freq;
    /// the LO is set to `target + SHIFT_HZ` to keep the standard offset.
    pub fn retune(&mut self, target_hz: u64) -> anyhow::Result<()> {
        let new_lo = target_hz + SHIFT_HZ;
        self.hackrf.set_freq(new_lo)?;
        self.tuned_to = new_lo;
        Ok(())
    }

    pub fn stop(self) -> anyhow::Result<()> {
        self.hackrf.stop()?;
        Ok(())
    }
}

/// Free-standing retune for callers that own a `HackRf` directly rather than
/// a [`Session`] (e.g. the `play`/`waterfall`/`hallway` paths that predate it).
pub fn retune(hackrf: &HackRf, target_rf_hz: u64) -> anyhow::Result<()> {
    hackrf.set_freq(target_rf_hz + SHIFT_HZ)?;
    Ok(())
}
