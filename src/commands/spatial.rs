use num_complex::Complex32;
use rtrb::RingBuffer;

use crate::{
    audio,
    dsp::{
        fft::Spectrum,
        fleet::Fleet,
        peaks::Event,
        spatial::SpatialMixer,
    },
    radio::{
        AVG_FFTS, FFT_SIZE, Gains, RADIO_SAMPLE_RATE_HZ, Session, default_peak_detector,
    },
};

// One read returns ~131k IQ samples (262k bytes / 2). At 2.4 Msps that's ~54.6 ms.
const READ_SAMPLES_PER_BLOCK: f32 = (262_144 / 2) as f32;
const BLOCK_SECONDS: f32 = READ_SAMPLES_PER_BLOCK / RADIO_SAMPLE_RATE_HZ as f32;

pub fn run(freq_hz: u64, k: usize, speed_mps: f32) -> anyhow::Result<()> {
    // stereo ring + sink
    let (mut producer, consumer) = RingBuffer::<f32>::new(32_768);
    for _ in 0..16_384 {
        let _ = producer.push(0.0);
    }
    let _audio_out = audio::start_stereo(consumer)?;

    let mut session = Session::open(freq_hz, Gains::SCAN)?;
    let tuned_to = session.tuned_to;

    let mut spatial = SpatialMixer::new(tuned_to as f64);
    // band edges in hallway meters: ±(sample_rate/2) / hz_per_meter
    let edge_m = (RADIO_SAMPLE_RATE_HZ as f32 / 2.0) / spatial.hz_per_meter;
    let mut walk_dir: f32 = 1.0;
    spatial.camera_x = -edge_m;

    eprintln!(
        "spatial: K={} tuned to {:.3} MHz, hallway ±{:.1}m, walking {} m/s",
        k,
        tuned_to as f64 / 1e6,
        edge_m,
        speed_mps,
    );

    let mut detector = default_peak_detector(false);
    let mut peak_events: Vec<Event> = Vec::new();
    let mut fleet = Fleet::new(k, tuned_to as f64);

    let mut spectrum = Spectrum::new(FFT_SIZE);
    let mut scratch_c: Vec<Complex32> = Vec::new();
    let mut scratch_decim: Vec<Complex32> = Vec::new();
    let mut scratch_audio_hi: Vec<f32> = Vec::new();
    let mut scratch_audio: Vec<f32> = Vec::new();
    let mut stereo_mix: Vec<f32> = Vec::new();
    let mut fft_pushed = 0usize;
    let mut iters: u64 = 0;
    let mut drops: u64 = 0;

    while !session.stopped() {
        let Some(iq) = session.read_block() else {
            continue;
        };

        // advance the camera, ping-pong at edges
        spatial.camera_x += walk_dir * speed_mps * BLOCK_SECONDS;
        if spatial.camera_x > edge_m {
            spatial.camera_x = edge_m;
            walk_dir = -1.0;
        } else if spatial.camera_x < -edge_m {
            spatial.camera_x = -edge_m;
            walk_dir = 1.0;
        }

        for slot in fleet.slots.iter_mut().filter(|s| s.freq_hz != 0.0) {
            slot.process(
                iq,
                &mut scratch_c,
                &mut scratch_decim,
                &mut scratch_audio_hi,
                &mut scratch_audio,
            );
        }

        let streams = fleet
            .slots
            .iter()
            .filter(|s| s.freq_hz != 0.0)
            .map(|s| (s.out.as_slice(), s.freq_hz));
        spatial.mix(streams, &mut stereo_mix);

        for &sample in &stereo_mix {
            if producer.push(sample).is_err() {
                drops += 1;
            }
        }

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
                        let (left_gain, right_gain) = spatial.channel_gains(slot.freq_hz);
                        format!(
                            "slot{i}={:.2}MHz L{:.2}R{:.2}",
                            slot.freq_hz / 1e6,
                            left_gain,
                            right_gain
                        )
                    }
                })
                .collect();
            eprintln!(
                "cam={:+.1}m {} drops={}",
                spatial.camera_x,
                parts.join(" "),
                drops
            );
        }
    }

    session.stop()?;
    eprintln!("\nspatial stopped cleanly. drops={drops}");
    Ok(())
}
