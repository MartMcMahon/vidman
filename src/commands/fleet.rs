use num_complex::Complex32;
use rtrb::RingBuffer;

use crate::{
    audio,
    dsp::{
        fleet::Fleet,
        fft::Spectrum,
        peaks::Event,
    },
    radio::{
        AVG_FFTS, FFT_SIZE, Gains, SHIFT_HZ, Session, default_peak_detector,
    },
};

pub fn run(freq_hz: u64, k: usize) -> anyhow::Result<()> {
    // mono audio sink
    let (mut producer, consumer) = RingBuffer::<f32>::new(16_384);
    for _ in 0..8_192 {
        let _ = producer.push(0.0);
    }
    let _audio_out = audio::start(consumer)?;

    let mut session = Session::open(freq_hz, Gains::SCAN)?;
    let tuned_to = session.tuned_to;
    eprintln!(
        "fleet: K={} tuned to {:.3} MHz (center {:.3} MHz + {} kHz LO offset)",
        k,
        tuned_to as f64 / 1e6,
        freq_hz as f64 / 1e6,
        SHIFT_HZ / 1000,
    );

    let mut detector = default_peak_detector(false);
    let mut peak_events: Vec<Event> = Vec::new();
    let mut fleet = Fleet::new(k, tuned_to as f64);

    let mut spectrum = Spectrum::new(FFT_SIZE);
    let mut scratch_c: Vec<Complex32> = Vec::new();
    let mut scratch_decim: Vec<Complex32> = Vec::new();
    let mut scratch_audio_hi: Vec<f32> = Vec::new();
    let mut scratch_audio: Vec<f32> = Vec::new();
    let mut mono_mix: Vec<f32> = Vec::new();
    let mut fft_pushed = 0usize;
    let mut iters: u64 = 0;
    let mut drops: u64 = 0;

    let inv_k = 1.0 / k as f32;

    while !session.stopped() {
        let Some(iq) = session.read_block() else {
            continue;
        };

        // process every active slot, then sample-wise sum into mono_mix
        let mut mix_len = 0usize;
        for slot in fleet.slots.iter_mut().filter(|s| s.freq_hz != 0.0) {
            slot.process(
                iq,
                &mut scratch_c,
                &mut scratch_decim,
                &mut scratch_audio_hi,
                &mut scratch_audio,
            );
            mix_len = mix_len.max(slot.out.len());
        }
        mono_mix.clear();
        mono_mix.resize(mix_len, 0.0);
        for slot in fleet.slots.iter().filter(|s| s.freq_hz != 0.0) {
            for (mix_sample, &slot_sample) in mono_mix.iter_mut().zip(&slot.out) {
                *mix_sample += slot_sample * inv_k;
            }
        }
        for &sample in &mono_mix {
            if producer.push(sample).is_err() {
                drops += 1;
            }
        }

        // FFT path: feed bins, detect peaks every AVG_FFTS, update fleet
        let mut fft_start = 0;
        while fft_start + FFT_SIZE <= iq.len() {
            spectrum.push(&iq[fft_start..fft_start + FFT_SIZE]);
            fft_pushed += 1;
            fft_start += FFT_SIZE;
            if fft_pushed == AVG_FFTS {
                let row = spectrum.take();
                peak_events.clear();
                detector.step(&row, tuned_to as f64, &mut peak_events);
                let stations = detector.snapshot(tuned_to as f64);
                fleet.update(&stations);
                fft_pushed = 0;
            }
        }

        iters += 1;
        if iters.is_multiple_of(20) {
            let parts: Vec<String> = fleet
                .slots
                .iter()
                .enumerate()
                .map(|(i, slot)| {
                    if slot.freq_hz == 0.0 {
                        format!("slot{i}=idle")
                    } else {
                        format!("slot{i}={:.3}MHz", slot.freq_hz / 1e6)
                    }
                })
                .collect();
            eprintln!("{} drops={drops}", parts.join(" "));
        }
    }

    session.stop()?;
    eprintln!("\nfleet stopped cleanly. drops={drops}");
    Ok(())
}
