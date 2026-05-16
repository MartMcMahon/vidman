use std::f32::consts::PI;

use num_complex::Complex32;

pub struct Mixer {
    phase_inc: f32,
    phase: f32,
}

impl Mixer {
    pub fn new(shift_hz: f32, sample_rate_hz: f32) -> Self {
        let phase_inc = 2.0 * PI * shift_hz / sample_rate_hz;
        Self {
            phase_inc,
            phase: 0.0,
        }
    }

    pub fn mix(&mut self, sample: Complex32) -> Complex32 {
        let (sin, cos) = self.phase.sin_cos();
        let lo = Complex32::new(cos, sin);
        self.phase += self.phase_inc;

        // wrap to [-pi, pi] so precesion doesn't decay
        if self.phase > PI {
            self.phase -= 2.0 * PI;
        } else if self.phase < -PI {
            self.phase += 2.0 * PI;
        }

        sample * lo
    }
}
