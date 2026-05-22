// Finite Impulse Resposne filter

use std::f32::consts::PI;

use num_complex::Complex32;

pub struct Fir {
    taps: Vec<f32>,
    history: Vec<Complex32>, // carried tail: last num_taps-1 samples of prev block
    decimation_factor: usize,
    sample_count: usize,
}

impl Fir {
    pub fn new(
        num_taps: usize,
        cutoff_hz: f32,
        sample_rate_hz: f32,
        decimation_factor: usize,
    ) -> Self {
        let taps = windowed_sinc(num_taps, cutoff_hz, sample_rate_hz);
        let history = vec![Complex32::new(0.0, 0.0); num_taps - 1];
        Self {
            taps,
            history,
            decimation_factor,
            sample_count: 0,
        }
    }

    pub fn process(&mut self, input: &[Complex32], output: &mut Vec<Complex32>) {
        let n = self.taps.len();
        let decim = self.decimation_factor;

        // first block-local index that lands on the decimation phase
        let r = (self.sample_count + 1) % decim;
        let j0 = (decim - r) % decim;

        // history becomes: carried tail (n-1 samples) ++ this input block
        self.history.extend_from_slice(input);

        let mut j = j0;
        while j < input.len() {
            // n-sample window ending at block sample j; contiguous → vectorizes
            let w = &self.history[j..j + n];
            let mut acc = Complex32::new(0.0, 0.0);
            for (&x, &t) in w.iter().zip(&self.taps) {
                acc += x * t;
            }
            output.push(acc);
            j += decim;
        }

        self.sample_count += input.len();
        // keep only the n-1 samples needed to start the next block
        let drop = self.history.len() - (n - 1);
        self.history.drain(..drop);
    }
}

// some maths
fn windowed_sinc(n: usize, cutoff_hz: f32, fs: f32) -> Vec<f32> {
    let fc = cutoff_hz / fs; // normalized cutoff in cycles/sample
    let center = (n - 1) as f32 / 2.0;
    let mut taps = Vec::with_capacity(n);
    let mut sum = 0.0f32;

    for i in 0..n {
        let x = i as f32 - center;
        let sinc = if x.abs() < 1e-6 {
            2.0 * fc
        } else {
            (2.0 * PI * fc * x).sin() / (PI * x)
        };

        let w = 0.54 - 0.46 * (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos();
        let tap = sinc * w;
        taps.push(tap);
        sum += tap;
    }

    for t in &mut taps {
        *t /= sum;
    }
    taps
}

pub struct RealFir {
    taps: Vec<f32>,
    history: Vec<f32>, // carried tail: last num_taps-1 samples of prev block
    decimation_factor: usize,
    sample_count: usize,
}

impl RealFir {
    pub fn new(
        num_taps: usize,
        cutoff_hz: f32,
        sample_rate_hz: f32,
        decimation_factor: usize,
    ) -> Self {
        let taps = windowed_sinc(num_taps, cutoff_hz, sample_rate_hz);
        let history = vec![0.0f32; num_taps - 1];
        Self {
            taps,
            history,
            decimation_factor,
            sample_count: 0,
        }
    }

    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let n = self.taps.len();
        let decim = self.decimation_factor;

        let r = (self.sample_count + 1) % decim;
        let j0 = (decim - r) % decim;

        self.history.extend_from_slice(input);

        let mut j = j0;
        while j < input.len() {
            let w = &self.history[j..j + n];
            let mut acc = 0.0f32;
            for (&x, &t) in w.iter().zip(&self.taps) {
                acc += x * t;
            }
            output.push(acc);
            j += decim;
        }

        self.sample_count += input.len();
        let drop = self.history.len() - (n - 1);
        self.history.drain(..drop);
    }
}
