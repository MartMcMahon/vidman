use std::io::{self, Write};

use crate::{
    dsp::{
        fft::Spectrum,
        peaks::Event,
    },
    radio::{
        AVG_FFTS, FFT_SIZE, Gains, ROWS_PER_SECOND, SHIFT_HZ, Session, default_peak_detector,
    },
};

pub fn run(freq_hz: u64, narrow: bool) -> anyhow::Result<()> {
    let mut session = Session::open(freq_hz, Gains::SCAN)?;
    let tuned_to = session.tuned_to;
    eprintln!(
        "scan: tuned to {:.3} MHz (center {:.3} MHz + {} kHz LO offset), {} rows/s",
        tuned_to as f64 / 1e6,
        freq_hz as f64 / 1e6,
        SHIFT_HZ / 1000,
        ROWS_PER_SECOND,
    );

    let mut detector = default_peak_detector(narrow);
    let mut spectrum = Spectrum::new(FFT_SIZE);
    let mut events: Vec<Event> = Vec::new();
    let mut fft_pushed = 0usize;
    let started = std::time::Instant::now();

    let stdout = io::stdout();
    let mut out = stdout.lock();

    while !session.stopped() {
        let Some(iq) = session.read_block() else {
            continue;
        };

        let mut fft_start = 0;
        while fft_start + FFT_SIZE <= iq.len() {
            spectrum.push(&iq[fft_start..fft_start + FFT_SIZE]);
            fft_pushed += 1;
            fft_start += FFT_SIZE;
            if fft_pushed == AVG_FFTS {
                let row = spectrum.take();
                events.clear();
                detector.step(&row, tuned_to as f64, &mut events);
                let t_ms = started.elapsed().as_millis() as u64;
                for event in &events {
                    match event {
                        Event::Open(station) => {
                            writeln!(
                                out,
                                r#"{{"t_ms":{},"event":"open","id":{},"freq_hz":{:.0},"power_db":{:.2}}}"#,
                                t_ms, station.id, station.freq_hz, station.power_db
                            )?;
                        }
                        Event::Close { id, freq_hz } => {
                            writeln!(
                                out,
                                r#"{{"t_ms":{},"event":"close","id":{},"freq_hz":{:.0}}}"#,
                                t_ms, id, freq_hz
                            )?;
                        }
                    }
                }
                out.flush()?;
                fft_pushed = 0;
            }
        }
    }

    session.stop()?;
    eprintln!("\nscan stopped cleanly.");
    Ok(())
}
