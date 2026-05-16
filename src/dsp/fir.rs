// Finite Impulse Resposne filter

use std::collections::VecDeque;
use std::f32::consts::PI;

use num_complex::Complex32;

pub struct Fir {
    taps: Vec<f32>,
    history: VecDeque<Complex32>,
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
        let history = VecDeque::from(vec![Complex32::new(0.0, 0.0); num_taps]);
        Self {
            taps,
            history,
            decimation_factor,
            sample_count: 0,
        }
    }

    pub fn process(&mut self, input: &[Complex32], output: &mut Vec<Complex32>) {
        for &x in input {
            self.history.pop_front();
            self.history.push_back(x);
            self.sample_count += 1;
            if self.sample_count.is_multiple_of(self.decimation_factor) {
                let mut acc = Complex32::new(0.0, 0.0);
                for (k, &h) in self.taps.iter().enumerate() {
                    acc += self.history[k] * h;
                }
                output.push(acc);
            }
        }
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
    history: VecDeque<f32>,
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
        let history = VecDeque::from(vec![0.0f32; num_taps]);
        Self {
            taps,
            history,
            decimation_factor,
            sample_count: 0,
        }
    }

    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        for &x in input {
            self.history.pop_front();
            self.history.push_back(x);
            self.sample_count += 1;
            if self.sample_count.is_multiple_of(self.decimation_factor) {
                let mut acc = 0.0f32;
                for (k, &h) in self.taps.iter().enumerate() {
                    acc += self.history[k] * h;
                }
                output.push(acc);
            }
        }
    }
}
