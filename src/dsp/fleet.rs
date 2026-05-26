use num_complex::Complex32;

use crate::dsp::{
    deemph::Deemph,
    demod::FmDemod,
    fir::{Fir, RealFir},
    mixer::Mixer,
    peaks::Station,
};

const SAMPLE_RATE_HZ: f32 = 2_400_000.0;
const TOL_HZ: f64 = 5_000.0;

pub struct DemodSlot {
    pub freq_hz: f64, // 0.0 = idle
    mixer: Mixer,
    fir1: Fir,
    fm: FmDemod,
    fir2: RealFir,
    deemph: Deemph,
    pub out: Vec<f32>,
}

impl DemodSlot {
    pub fn new() -> Self {
        Self {
            freq_hz: 0.0,
            mixer: Mixer::new(0.0, SAMPLE_RATE_HZ),
            fir1: Fir::new(63, 100_000.0, SAMPLE_RATE_HZ, 10),
            fm: FmDemod::new(),
            fir2: RealFir::new(63, 15_000.0, 240_000.0, 5),
            deemph: Deemph::new(75e-6, 48_000.0),
            out: Vec::with_capacity(4096),
        }
    }

    pub fn retune(&mut self, freq_hz: f64, tuned_center_hz: f64) {
        // positive shift brings a NEGATIVE-baseband signal to DC; the HackRF
        // sits SHIFT_HZ above target, so a station at freq_hz lives at
        // baseband (freq_hz - tuned_center_hz), and we need to mix by the
        // negative of that to land it on DC.
        let shift_hz = (tuned_center_hz - freq_hz) as f32;
        self.mixer.retune(shift_hz, SAMPLE_RATE_HZ);
        self.freq_hz = freq_hz;
    }

    pub fn process(
        &mut self,
        iq: &[Complex32],
        scratch_c: &mut Vec<Complex32>,
        scratch_decimated: &mut Vec<Complex32>,
        scratch_audio_hi: &mut Vec<f32>,
        scratch_audio: &mut Vec<f32>,
    ) {
        scratch_c.clear();
        scratch_decimated.clear();
        scratch_audio_hi.clear();
        scratch_audio.clear();
        self.out.clear();

        for &z in iq {
            scratch_c.push(self.mixer.mix(z));
        }
        self.fir1.process(scratch_c, scratch_decimated);
        self.fm.process(scratch_decimated, scratch_audio_hi);
        self.fir2.process(scratch_audio_hi, scratch_audio);
        self.deemph.process(scratch_audio, &mut self.out);
    }
}

pub struct Fleet {
    pub slots: Vec<DemodSlot>,
    pub tuned_center_hz: f64,
}

impl Fleet {
    /// HackRF was retuned to a new absolute center — bring all active slots
    /// along, phase-continuously, so the audio doesn't go to noise.
    pub fn set_tuned_center(&mut self, new_center_hz: f64) {
        self.tuned_center_hz = new_center_hz;
        for slot in &mut self.slots {
            if slot.freq_hz != 0.0 {
                slot.retune(slot.freq_hz, new_center_hz);
            }
        }
    }
}

impl Fleet {
    pub fn new(k: usize, tuned_center_hz: f64) -> Self {
        Self {
            slots: (0..k).map(|_| DemodSlot::new()).collect(),
            tuned_center_hz,
        }
    }

    /// keep-then-fill matching: existing tunings within TOL_HZ of a current
    /// station survive untouched (phase-continuous, no click); freed slots
    /// take the strongest unclaimed stations.
    pub fn update(&mut self, stations: &[Station]) {
        let mut claimed = vec![false; stations.len()];

        for slot in &mut self.slots {
            if slot.freq_hz == 0.0 {
                continue;
            }
            let m = stations.iter().enumerate().find(|(i, s)| {
                !claimed[*i] && (s.freq_hz - slot.freq_hz).abs() < TOL_HZ
            });
            match m {
                Some((i, _)) => claimed[i] = true,
                None => slot.freq_hz = 0.0,
            }
        }

        let mut unclaimed: Vec<&Station> = stations
            .iter()
            .enumerate()
            .filter(|(i, _)| !claimed[*i])
            .map(|(_, s)| s)
            .collect();
        unclaimed.sort_by(|a, b| b.power_db.partial_cmp(&a.power_db).unwrap());

        let center = self.tuned_center_hz;
        for slot in self.slots.iter_mut().filter(|s| s.freq_hz == 0.0) {
            if let Some(st) = unclaimed.pop() {
                slot.retune(st.freq_hz, center);
            }
        }
    }
}
