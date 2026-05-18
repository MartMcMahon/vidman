pub struct Deemph {
    alpha: f32,
    y_prev: f32,
}

impl Deemph {
    pub fn new(tau_seconds: f32, sample_rate_hz: f32) -> Self {
        let dt = 1.0 / sample_rate_hz;
        let alpha = dt / (tau_seconds + dt);
        Self { alpha, y_prev: 0.0 }
    }

    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        for &x in input {
            let y = self.alpha * x + (1.0 - self.alpha) * self.y_prev;
            output.push(y);
            self.y_prev = y;
        }
    }
}
