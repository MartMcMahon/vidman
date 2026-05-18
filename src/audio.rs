use std::sync::{
    Arc,
    atomic::{AtomicI16, AtomicU64},
};

use cpal::{
    StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use rtrb::Consumer;

pub struct AudioOut {
    _stream: cpal::Stream,
    pub underruns: Arc<AtomicU64>,
    pub pops: Arc<AtomicU64>,
}

pub fn start(mut consumer: Consumer<f32>) -> anyhow::Result<AudioOut> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("no default output device"))?;

    eprintln!("device: {:?}", device.description());
    eprintln!("default config: {:?}", device.default_output_config());

    let config = StreamConfig {
        channels: 2,
        sample_rate: 48_000,
        buffer_size: cpal::BufferSize::Default,
    };

    let underruns = Arc::new(AtomicU64::new(0));
    let underruns_cb = underruns.clone();
    let pops = Arc::new(AtomicU64::new(0));
    let pops_cb = pops.clone();

    let stream = device.build_output_stream(
        &config,
        // data callback
        move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for frame in out.chunks_exact_mut(2) {
                match consumer.pop() {
                    Ok(s) => {
                        frame[0] = s;
                        frame[1] = s;
                        pops_cb.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    Err(_) => {
                        frame[0] = 0.0;
                        frame[1] = 0.0;
                        underruns_cb.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        },
        // error callback
        |err| eprintln!("audio stream error: {err}"),
        None, // no timeout
    )?;

    stream.play()?;

    Ok(AudioOut {
        _stream: stream,
        underruns,
        pops,
    })
}
