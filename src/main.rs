use clap::{Parser, Subcommand};
use num_complex::Complex32;
use rtrb::RingBuffer;
use seify_hackrfone::{Config, HackRf};
use std::{
    fs::File,
    io::{self, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::dsp::{
    deemph::Deemph,
    demod::FmDemod,
    fir::{Fir, RealFir},
    mixer::Mixer,
};
mod audio;
mod dsp;
mod viz;

const BLOCK_SIZE: usize = 480_000;
const BAR_WIDTH: usize = 50;
const BAR_FULL_SCALE: f32 = 0.5;

const KMFA: u64 = 89_500_000; // KMFA 89.5
const SHIFT_HZ: u64 = 200_000;
const RADIO_SAMPLE_RATE_HZ: u32 = 2_400_000;

#[derive(Parser)]
#[command()]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Capture,
    Meter,
    Demod,
    #[command(alias = "viz")]
    Visualize,
    Play,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Capture => {
            capture();
        }
        Commands::Meter => meter().expect("meter"),
        Commands::Demod => demod().expect("demod"),
        Commands::Visualize => viz::run().expect("visualization"),
        Commands::Play => play().expect("play"),
    }
}

fn capture() {
    let hackrf = HackRf::open_first().expect("opening hackrf device");
    hackrf
        .start_rx(&Config {
            txvga_db: 0,
            vga_db: 20,
            lna_db: 16,
            amp_enable: false,
            antenna_enable: false,
            frequency_hz: 89_700_000, // aiming for KMFA Classical station FM89.5 + 200khz
            sample_rate_hz: 2_400_000,
            sample_rate_div: 1,
        })
        .expect("error with config}");
    let mut buf = vec![0u8; 262_144];

    let mut total_bytes_written = 0;
    let mut out_file = File::create("cap.cs8").expect("creating cap file");

    loop {
        let n = hackrf.read(&mut buf).expect("reading");

        let mean_abs = buf[..n]
            .iter()
            .map(|&b| (b as i8).unsigned_abs() as u64)
            .sum::<u64>()
            / n as u64;
        println!("mean |sample|: {mean_abs}");

        out_file.write_all(&buf[..n]).expect("writing to file");
        total_bytes_written += n;
        if total_bytes_written >= 2_400_000 * 2 * 20 {
            break;
        }
    }

    hackrf.stop().expect("stopping");
}

fn meter() -> anyhow::Result<()> {
    let config = Config {
        txvga_db: 0,
        vga_db: 8,
        lna_db: 8,
        amp_enable: false,
        antenna_enable: false,
        frequency_hz: 462_637_500,
        // frequency_hz: 89_500_000,
        sample_rate_hz: 2_400_000,
        sample_rate_div: 1,
    };
    let hackrf = HackRf::open_first()?;
    hackrf.start_rx(&config)?;

    let mut buf = vec![0u8; 262_144];
    let mut block = vec![0u8; BLOCK_SIZE];
    let mut filled = 0usize;
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let mut errors = 0;
    loop {
        let n = match hackrf.read(&mut buf) {
            Ok(n) => {
                errors = 0;
                n
            }
            Err(e) => {
                eprintln!("\nread error: {e:#} — continuing");
                if errors >= 5 {
                    eprintln!("restarting stream");
                    let _ = hackrf.stop(); // ignore stop error
                    hackrf.start_rx(&config)?;
                    errors = 0;
                }
                filled = 0; // discard block
                continue;
            }
        };
        let mut src = 0;
        while src < n {
            let take = (BLOCK_SIZE - filled).min(n - src);
            block[filled..filled + take].copy_from_slice(&buf[src..src + take]);
            filled += take;
            src += take;
            if filled == BLOCK_SIZE {
                draw_bar(&mut out, mean_abs_z(&block));
                filled = 0;
            }
        }
    }
}

fn mean_abs_z(block: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    for chunk in block.chunks_exact(2) {
        let [raw_i, raw_q] = chunk else {
            unreachable!()
        };
        let i = (*raw_i as i8) as f32 / 128.0;
        let q = (*raw_q as i8) as f32 / 128.0;
        sum += (i * i + q * q).sqrt();
    }
    sum / (block.len() / 2) as f32
}

fn draw_bar<W: Write>(out: &mut W, mean: f32) {
    let filled = ((mean / BAR_FULL_SCALE) * BAR_WIDTH as f32).round() as usize;
    let filled = filled.min(BAR_WIDTH);
    let bar: String = "█".repeat(filled) + &" ".repeat(BAR_WIDTH - filled);

    write!(out, "\r|{bar}| {mean:.4}").expect("write");
    out.flush().expect("flush");
}

fn demod() -> anyhow::Result<()> {
    use dsp::mixer::Mixer;
    use num_complex::Complex32;

    // read bytes to mem
    let bytes = std::fs::read("cap.cs8")?;
    eprintln!(
        "read {} bytes ({:.2} s @ 2.4 Msps)",
        bytes.len(),
        bytes.len() as f32 / 4_800_000.0
    );

    // convert i8 IQ -> Complex32
    let iq: Vec<Complex32> = bytes
        .chunks_exact(2)
        .map(|chunk| {
            let [raw_i, raw_q] = chunk else {
                unreachable!()
            };
            let i = (*raw_i as i8) as f32 / 128.0;
            let q = (*raw_q as i8) as f32 / 128.0;
            // sum += (i * i + q * q).sqrt();
            Complex32::new(i, q)
        })
        .collect();

    // shift station from -200kHz to DC
    let mut mixer = Mixer::new(200_000.0, 2_400_000.0);
    let mixed: Vec<Complex32> = iq.into_iter().map(|s| mixer.mix(s)).collect();

    // first FIR; low-pass @ 100kHz; 2,400 ksps -> 240 ksps
    let mut fir1 = Fir::new(63, 100_000.0, 2_400_000.0, 10);
    let mut decimated = Vec::with_capacity(mixed.len() / 10 + 1);
    fir1.process(&mixed, &mut decimated);

    // fm demod; complex -> audio (240 ksps)
    let mut fm = FmDemod::new();
    let mut audio_hi = Vec::with_capacity(decimated.len());
    fm.process(&decimated, &mut audio_hi);

    // audio FIR; low-pass @ 15kHz; 240 ksps -> 48 ksps
    let mut fir2 = RealFir::new(63, 15_000.0, 240_000.0, 5);
    let mut audio: Vec<f32> = Vec::with_capacity(audio_hi.len() / 5 + 1);
    fir2.process(&audio_hi, &mut audio);

    // write raw f32
    let mut out = std::fs::File::create("demod.f32")?;
    for s in &audio {
        out.write_all(&s.to_le_bytes())?;
    }
    eprintln!(
        "wrote {} audio samples ({:.2} s @ 48 ksps)",
        audio.len(),
        audio.len() as f32 / 48_000.0
    );

    Ok(())
}

fn play() -> anyhow::Result<()> {
    // ring buffer; bridge from radio to audio
    let (mut producer, consumer) = RingBuffer::<f32>::new(16284);

    // half fill to prevent jitter
    for _ in 0..8142 {
        let _ = producer.push(0.0);
    }

    // ctrl-c handling
    let stop = Arc::new(AtomicBool::new(false));
    let stop_handler = stop.clone();
    ctrlc::set_handler(move || {
        stop_handler.store(true, Ordering::Relaxed);
    })?;

    // start audio output
    let audio_out = audio::start(consumer)?;
    // configure hackrf
    let hackrf = HackRf::open_first()?;

    hackrf.start_rx(&Config {
        txvga_db: 0,
        vga_db: 20,
        lna_db: 16,
        amp_enable: false,
        antenna_enable: false,
        frequency_hz: KMFA + SHIFT_HZ,
        sample_rate_hz: RADIO_SAMPLE_RATE_HZ,
        sample_rate_div: 1,
    })?;

    // radio chain
    let mut mixer = Mixer::new(SHIFT_HZ as f32, RADIO_SAMPLE_RATE_HZ as f32);
    let mut fir1 = Fir::new(63, 100_000.0, 2_400_000.0, 10);
    let mut fm = FmDemod::new();
    let mut fir2 = RealFir::new(63, 15_000.0, 240_000.0, 5);
    let mut deemph = Deemph::new(75e-6, 48_000.0);

    let mut buf = vec![0u8; 262_144];
    let mut iq = Vec::with_capacity(buf.len() / 2);
    let mut mixed = Vec::with_capacity(buf.len() / 2);
    let mut decimated = Vec::new();
    let mut audio_hi = Vec::new();
    let mut audio = Vec::new();
    let mut deemphed = Vec::new();

    let mut drops: u64 = 0;
    let mut iters: u64 = 0;

    let mut total_pushed = 0u64;
    // let start = std::time::Instant::now();
    let mut total_bytes = 0u64;
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let n = hackrf.read(&mut buf)?;
        total_bytes += n as u64;

        iq.clear();
        mixed.clear();
        decimated.clear();
        audio_hi.clear();
        audio.clear();
        deemphed.clear();

        for chunk in buf[..n].chunks_exact(2) {
            let i = (chunk[0] as i8) as f32 / 128.0;
            let q = (chunk[1] as i8) as f32 / 128.0;
            iq.push(Complex32::new(i, q));
        }
        for &sample in &iq {
            mixed.push(mixer.mix(sample));
        }
        fir1.process(&mixed, &mut decimated);
        fm.process(&decimated, &mut audio_hi);
        fir2.process(&audio_hi, &mut audio);
        deemph.process(&audio, &mut deemphed);

        for &sample in &deemphed {
            if producer.push(sample).is_err() {
                drops += 1;
            } else {
                total_pushed += 1;
            }
        }

        iters += 1;
        if iters.is_multiple_of(20) {
            // let elapsed = start.elapsed().as_secs_f64();
            // let pops = audio_out.pops.load(std::sync::atomic::Ordering::Relaxed);
            // let underruns = audio_out
            //     .underruns
            //     .load(std::sync::atomic::Ordering::Relaxed);
            // eprintln!(
            //     "push={:.0}Hz pop={:.0}Hz iters={iters} drops={drops} under={underruns}",
            //     total_pushed as f64 / elapsed,
            //     pops as f64 / elapsed,
            // );
            //
            // in the periodic eprintln:
            // eprintln!(
            //     "hackrf rate: {:.3} Msps",
            //     total_bytes as f64 / 2.0 / elapsed / 1e6
            // );
        }
    }
    hackrf.stop()?;
    eprintln!("\nstopped cleanly. drops={drops}");
    Ok(())
}
