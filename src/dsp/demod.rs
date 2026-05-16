use num_complex::Complex32;

pub struct FmDemod {
    prev: Complex32,
}
impl FmDemod {
    pub fn new() -> Self {
        Self {
            prev: Complex32::new(0.0, 0.0),
        }
    }

    pub fn process(&mut self, input: &[Complex32], output: &mut Vec<f32>) {
        for &z in input {
            let diff = z * self.prev.conj();
            output.push(diff.arg());
            self.prev = z;
        }
    }
}
