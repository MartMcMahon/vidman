use rtrb::RingBuffer;

use crate::{
    audio,
    control::{ControlMsg, Mode, default_freq_for_mode, mode_name, stdin_reader_channel},
    dsp::pipeline::Pipeline,
    radio::{Gains, KMFA, Session},
};

pub fn run() -> anyhow::Result<()> {
    // ring buffer; bridge from radio to audio. Half-fill to absorb jitter.
    let (mut producer, consumer) = RingBuffer::<f32>::new(16_384);
    for _ in 0..8_192 {
        let _ = producer.push(0.0);
    }
    let _audio_out = audio::start(consumer)?;

    let control_rx = stdin_reader_channel("play");
    let mut session = Session::open(KMFA, Gains::PLAY)?;
    let mut current_freq_hz: u64 = KMFA;
    let mut pipeline = Pipeline::fm();

    let mut mixed = Vec::new();
    let mut decimated = Vec::new();
    let mut audio_hi = Vec::new();
    let mut deemphed = Vec::new();
    let mut drops: u64 = 0;

    while !session.stopped() {
        while let Ok(msg) = control_rx.try_recv() {
            let new_freq = match msg {
                ControlMsg::SetMode(mode) => {
                    pipeline = match mode {
                        Mode::Fm => Pipeline::fm(),
                        Mode::Gmrs => Pipeline::gmrs(),
                    };
                    eprintln!("switched to {}", mode_name(mode));
                    default_freq_for_mode(mode)
                }
                ControlMsg::RetuneAbs(f) => f,
                ControlMsg::RetuneRel(delta) => {
                    (current_freq_hz as i64 + delta).max(1) as u64
                }
            };
            match session.retune(new_freq) {
                Ok(()) => {
                    current_freq_hz = new_freq;
                    eprintln!("tuned to {:.3} MHz", new_freq as f64 / 1e6);
                }
                Err(e) => eprintln!("retune error: {e}"),
            }
        }

        let Some(iq) = session.read_block() else {
            continue;
        };

        mixed.clear();
        decimated.clear();
        audio_hi.clear();
        deemphed.clear();
        pipeline.process(iq, &mut mixed, &mut decimated, &mut audio_hi, &mut deemphed);

        for &sample in &deemphed {
            if producer.push(sample).is_err() {
                drops += 1;
            }
        }
    }

    session.stop()?;
    eprintln!("\nstopped cleanly. drops={drops}");
    Ok(())
}
