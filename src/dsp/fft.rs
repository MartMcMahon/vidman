use std::{f32::consts::PI, sync::Arc};

use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

pub struct Spectrum {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    buf: Vec<Complex32>,
    accum: Vec<f32>,
    accum_count: usize,
    size: usize,
}

impl Spectrum {
    pub fn new(size: usize) -> Self {
        let fft = FftPlanner::new().plan_fft_forward(size);
        let window = hann(size);
        Self {
            fft,
            window,
            buf: vec![Complex32::new(0.0, 0.0); size],
            accum: vec![0.0; size],
            accum_count: 0,
            size,
        }
    }

    pub fn push(&mut self, chunk: &[Complex32]) {
        assert_eq!(chunk.len(), self.size);
        for i in 0..self.size {
            self.buf[i] = chunk[i] * self.window[i];
        }
        self.fft.process(&mut self.buf);
        for i in 0..self.size {
            self.accum[i] += self.buf[i].norm_sqr();
        }
        self.accum_count += 1;
    }

    pub fn take(&mut self) -> Vec<f32> {
        let n = self.accum_count.max(1) as f32;
        let mut out = vec![0.0; self.size];
        // avg power -> magnitude -> dB
        for k in 0..self.size {
            let power = self.accum[k] / n;
            out[k] = 10.0 * (power + 1e-12).log10();
        }

        self.accum.fill(0.0);
        self.accum_count = 0;
        fftshift(&mut out);
        out
    }
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos()))
        .collect()
}

fn fftshift(v: &mut [f32]) {
    let half = v.len() / 2;
    v.rotate_left(half);
}
