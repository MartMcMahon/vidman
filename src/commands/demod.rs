use std::io::Write;

use num_complex::Complex32;

use crate::{
    dsp::{
        demod::FmDemod,
        fir::{Fir, RealFir},
        mixer::Mixer,
    },
    radio::samples_to_iq,
};

pub fn run() -> anyhow::Result<()> {
    let bytes = std::fs::read("cap.cs8")?;
    eprintln!(
        "read {} bytes ({:.2} s @ 2.4 Msps)",
        bytes.len(),
        bytes.len() as f32 / 4_800_000.0
    );

    let mut iq: Vec<Complex32> = Vec::with_capacity(bytes.len() / 2);
    samples_to_iq(&bytes, &mut iq);

    // shift station from -200 kHz to DC
    let mut mixer = Mixer::new(200_000.0, 2_400_000.0);
    let mixed: Vec<Complex32> = iq.into_iter().map(|s| mixer.mix(s)).collect();

    // first FIR: low-pass @ 100 kHz, 2,400 ksps -> 240 ksps
    let mut fir1 = Fir::new(63, 100_000.0, 2_400_000.0, 10);
    let mut decimated = Vec::with_capacity(mixed.len() / 10 + 1);
    fir1.process(&mixed, &mut decimated);

    // FM demod: complex -> audio (240 ksps)
    let mut fm = FmDemod::new();
    let mut audio_hi = Vec::with_capacity(decimated.len());
    fm.process(&decimated, &mut audio_hi);

    // audio FIR: low-pass @ 15 kHz, 240 ksps -> 48 ksps
    let mut fir2 = RealFir::new(63, 15_000.0, 240_000.0, 5);
    let mut audio: Vec<f32> = Vec::with_capacity(audio_hi.len() / 5 + 1);
    fir2.process(&audio_hi, &mut audio);

    let mut out = std::fs::File::create("demod.f32")?;
    for sample in &audio {
        out.write_all(&sample.to_le_bytes())?;
    }
    eprintln!(
        "wrote {} audio samples ({:.2} s @ 48 ksps)",
        audio.len(),
        audio.len() as f32 / 48_000.0
    );

    Ok(())
}
