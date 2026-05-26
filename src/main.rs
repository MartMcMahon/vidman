use clap::{Parser, Subcommand};
use num_complex::Complex32;
use ratatui::crossterm::Command;
use rtrb::RingBuffer;
use seify_hackrfone::{Config, HackRf};
use std::{
    fs::File,
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};

use crate::dsp::{
    deemph::Deemph,
    demod::FmDemod,
    fft::Spectrum,
    fir::{Fir, RealFir},
    fleet::Fleet,
    mixer::Mixer,
    peaks::{Event, Params as PeakParams, PeakDetector},
    pipeline::Pipeline,
    spatial::SpatialMixer,
};
mod audio;
mod dsp;
mod gmrs;
mod hallway;
mod viz;
mod waterfall;

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
    #[command(alias = "wf")]
    Waterfall,
    /// Detect transmissions in the captured band and emit NDJSON events on stdout.
    Scan {
        /// Center frequency in Hz (suffix M = MHz, K = kHz)
        #[arg(long, default_value = "100M")]
        freq: String,
        /// Narrowband mode (smaller min-separation, suited to GMRS-style channels)
        #[arg(long)]
        narrow: bool,
    },
    /// Run K parallel FM demod chains tracking the K strongest stations, mono mix.
    Fleet {
        #[arg(long, default_value = "100M")]
        freq: String,
        #[arg(long, default_value_t = 4)]
        k: usize,
    },
    /// Fleet + spatial stereo pan. Camera auto-walks back and forth through the band.
    Spatial {
        #[arg(long, default_value = "100M")]
        freq: String,
        #[arg(long, default_value_t = 4)]
        k: usize,
        /// Camera walking speed in m/s
        #[arg(long, default_value_t = 2.0)]
        speed: f32,
    },
    /// 3D hallway scene: WASD to walk, A/D to turn, spectrum on walls, single-station audio.
    Hallway {
        #[arg(long, default_value = "89.5M")]
        freq: String,
    },
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
        Commands::Waterfall => waterfall::run(KMFA).expect("waterfall"),
        Commands::Scan { freq, narrow } => scan(parse_freq(&freq), narrow).expect("scan"),
        Commands::Fleet { freq, k } => fleet(parse_freq(&freq), k).expect("fleet"),
        Commands::Spatial { freq, k, speed } => {
            spatial(parse_freq(&freq), k, speed).expect("spatial")
        }
        Commands::Hallway { freq } => hallway::run(parse_freq(&freq)).expect("hallway"),
    }
}

fn parse_freq(s: &str) -> u64 {
    let s = s.trim();
    let (num, mult) = if let Some(rest) = s.strip_suffix(['M', 'm']) {
        (rest, 1_000_000.0)
    } else if let Some(rest) = s.strip_suffix(['K', 'k']) {
        (rest, 1_000.0)
    } else {
        (s, 1.0)
    };
    let n: f64 = num.parse().expect("freq is a number");
    (n * mult) as u64
}

fn scan(freq_hz: u64, narrow: bool) -> anyhow::Result<()> {
    const FFT_SIZE: usize = 1024;
    const AVG_FFTS: usize = 64;
    const ROWS_PER_SECOND: f32 =
        RADIO_SAMPLE_RATE_HZ as f32 / (FFT_SIZE as f32 * AVG_FFTS as f32);

    let stop = Arc::new(AtomicBool::new(false));
    let stop_handler = stop.clone();
    ctrlc::set_handler(move || {
        stop_handler.store(true, Ordering::Relaxed);
    })?;

    let hackrf = HackRf::open_first()?;
    let tuned_to = freq_hz + SHIFT_HZ;
    hackrf.start_rx(&Config {
        txvga_db: 0,
        vga_db: 16,
        lna_db: 16,
        amp_enable: false,
        antenna_enable: false,
        frequency_hz: tuned_to,
        sample_rate_hz: RADIO_SAMPLE_RATE_HZ,
        sample_rate_div: 1,
    })?;
    eprintln!(
        "scan: tuned to {:.3} MHz (center {:.3} MHz + {} kHz LO offset), {} rows/s",
        tuned_to as f64 / 1e6,
        freq_hz as f64 / 1e6,
        SHIFT_HZ / 1000,
        ROWS_PER_SECOND,
    );

    let mut detector = PeakDetector::new(PeakParams {
        sample_rate_hz: RADIO_SAMPLE_RATE_HZ as f32,
        fft_size: FFT_SIZE,
        open_threshold_db: 10.0,
        close_threshold_db: 6.0,
        hang_seconds: 0.5,
        rows_per_second: ROWS_PER_SECOND,
        min_separation_hz: if narrow { 12_500.0 } else { 150_000.0 },
        // skip the LO leakage zone plus a bit of slop; the tuned signal sits
        // SHIFT_HZ away from DC so it's not hidden by this exclusion
        dc_skip_hz: 50_000.0,
    });

    let mut spec = Spectrum::new(FFT_SIZE);
    let mut buf = vec![0u8; 262_144];
    let mut iq: Vec<Complex32> = Vec::with_capacity(buf.len() / 2);
    let mut events: Vec<Event> = Vec::new();
    let mut pushed = 0usize;
    let started = std::time::Instant::now();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    while !stop.load(Ordering::Relaxed) {
        let n = match hackrf.read(&mut buf) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("hackrf read error: {e:#}");
                continue;
            }
        };
        iq.clear();
        for chunk in buf[..n].chunks_exact(2) {
            let i = (chunk[0] as i8) as f32 / 128.0;
            let q = (chunk[1] as i8) as f32 / 128.0;
            iq.push(Complex32::new(i, q));
        }

        let mut s = 0;
        while s + FFT_SIZE <= iq.len() {
            spec.push(&iq[s..s + FFT_SIZE]);
            pushed += 1;
            s += FFT_SIZE;
            if pushed == AVG_FFTS {
                let row = spec.take();
                events.clear();
                detector.step(&row, tuned_to as f64, &mut events);
                let t_ms = started.elapsed().as_millis() as u64;
                for ev in &events {
                    match ev {
                        Event::Open(st) => {
                            writeln!(
                                out,
                                r#"{{"t_ms":{},"event":"open","id":{},"freq_hz":{:.0},"power_db":{:.2}}}"#,
                                t_ms, st.id, st.freq_hz, st.power_db
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
                pushed = 0;
            }
        }
    }
    hackrf.stop()?;
    eprintln!("\nscan stopped cleanly.");
    Ok(())
}

fn fleet(freq_hz: u64, k: usize) -> anyhow::Result<()> {
    const FFT_SIZE: usize = 1024;
    const AVG_FFTS: usize = 64;
    const ROWS_PER_SECOND: f32 =
        RADIO_SAMPLE_RATE_HZ as f32 / (FFT_SIZE as f32 * AVG_FFTS as f32);

    let stop = Arc::new(AtomicBool::new(false));
    let stop_handler = stop.clone();
    ctrlc::set_handler(move || {
        stop_handler.store(true, Ordering::Relaxed);
    })?;

    // audio sink: same ring/cpal setup as play() and waterfall
    let (mut producer, consumer) = RingBuffer::<f32>::new(16_384);
    for _ in 0..8_192 {
        let _ = producer.push(0.0);
    }
    let _audio_out = audio::start(consumer)?;

    let hackrf = HackRf::open_first()?;
    let tuned_to = freq_hz + SHIFT_HZ;
    hackrf.start_rx(&Config {
        txvga_db: 0,
        vga_db: 16,
        lna_db: 16,
        amp_enable: false,
        antenna_enable: false,
        frequency_hz: tuned_to,
        sample_rate_hz: RADIO_SAMPLE_RATE_HZ,
        sample_rate_div: 1,
    })?;
    eprintln!(
        "fleet: K={} tuned to {:.3} MHz (center {:.3} MHz + {} kHz LO offset)",
        k,
        tuned_to as f64 / 1e6,
        freq_hz as f64 / 1e6,
        SHIFT_HZ / 1000,
    );

    let mut detector = PeakDetector::new(PeakParams {
        sample_rate_hz: RADIO_SAMPLE_RATE_HZ as f32,
        fft_size: FFT_SIZE,
        open_threshold_db: 10.0,
        close_threshold_db: 6.0,
        hang_seconds: 0.5,
        rows_per_second: ROWS_PER_SECOND,
        min_separation_hz: 150_000.0,
        dc_skip_hz: 50_000.0,
    });
    let mut peak_events = Vec::new();
    let mut fleet = Fleet::new(k, tuned_to as f64);

    let mut spec = Spectrum::new(FFT_SIZE);
    let mut buf = vec![0u8; 262_144];
    let mut iq: Vec<Complex32> = Vec::with_capacity(buf.len() / 2);
    let mut scratch_c: Vec<Complex32> = Vec::new();
    let mut scratch_decim: Vec<Complex32> = Vec::new();
    let mut scratch_audio_hi: Vec<f32> = Vec::new();
    let mut scratch_audio: Vec<f32> = Vec::new();
    let mut mono_mix: Vec<f32> = Vec::new();
    let mut pushed = 0usize;
    let mut iters: u64 = 0;
    let mut drops: u64 = 0;

    let inv_k = 1.0 / k as f32;

    while !stop.load(Ordering::Relaxed) {
        let n = match hackrf.read(&mut buf) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("hackrf read error: {e:#}");
                continue;
            }
        };
        iq.clear();
        for chunk in buf[..n].chunks_exact(2) {
            let i = (chunk[0] as i8) as f32 / 128.0;
            let q = (chunk[1] as i8) as f32 / 128.0;
            iq.push(Complex32::new(i, q));
        }

        // process every active slot, then sample-wise sum into mono_mix
        let mut mix_len = 0usize;
        for slot in fleet.slots.iter_mut().filter(|s| s.freq_hz != 0.0) {
            slot.process(
                &iq,
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
            for (m, &s) in mono_mix.iter_mut().zip(&slot.out) {
                *m += s * inv_k;
            }
        }
        for &s in &mono_mix {
            if producer.push(s).is_err() {
                drops += 1;
            }
        }

        // FFT path: feed bins, detect peaks every AVG_FFTS, update fleet on each row
        let mut s_idx = 0;
        while s_idx + FFT_SIZE <= iq.len() {
            spec.push(&iq[s_idx..s_idx + FFT_SIZE]);
            pushed += 1;
            s_idx += FFT_SIZE;
            if pushed == AVG_FFTS {
                let row = spec.take();
                peak_events.clear();
                detector.step(&row, tuned_to as f64, &mut peak_events);
                let stations = detector.snapshot(tuned_to as f64);
                fleet.update(&stations);
                pushed = 0;
            }
        }

        iters += 1;
        if iters.is_multiple_of(20) {
            let mut parts: Vec<String> = Vec::with_capacity(fleet.slots.len());
            for (i, slot) in fleet.slots.iter().enumerate() {
                if slot.freq_hz == 0.0 {
                    parts.push(format!("slot{i}=idle"));
                } else {
                    parts.push(format!("slot{i}={:.3}MHz", slot.freq_hz / 1e6));
                }
            }
            eprintln!("{} drops={drops}", parts.join(" "));
        }
    }
    hackrf.stop()?;
    eprintln!("\nfleet stopped cleanly. drops={drops}");
    Ok(())
}

fn spatial(freq_hz: u64, k: usize, speed_mps: f32) -> anyhow::Result<()> {
    const FFT_SIZE: usize = 1024;
    const AVG_FFTS: usize = 64;
    const ROWS_PER_SECOND: f32 =
        RADIO_SAMPLE_RATE_HZ as f32 / (FFT_SIZE as f32 * AVG_FFTS as f32);

    let stop = Arc::new(AtomicBool::new(false));
    let stop_handler = stop.clone();
    ctrlc::set_handler(move || {
        stop_handler.store(true, Ordering::Relaxed);
    })?;

    // stereo ring + sink
    let (mut producer, consumer) = RingBuffer::<f32>::new(32_768);
    for _ in 0..16_384 {
        let _ = producer.push(0.0);
    }
    let _audio_out = audio::start_stereo(consumer)?;

    let hackrf = HackRf::open_first()?;
    let tuned_to = freq_hz + SHIFT_HZ;
    hackrf.start_rx(&Config {
        txvga_db: 0,
        vga_db: 16,
        lna_db: 16,
        amp_enable: false,
        antenna_enable: false,
        frequency_hz: tuned_to,
        sample_rate_hz: RADIO_SAMPLE_RATE_HZ,
        sample_rate_div: 1,
    })?;

    let mut spatial = SpatialMixer::new(tuned_to as f64);
    // band edges in hallway meters: ±(sample_rate/2) / hz_per_meter
    let edge_m = (RADIO_SAMPLE_RATE_HZ as f32 / 2.0) / spatial.hz_per_meter;
    let mut walk_dir: f32 = 1.0;
    spatial.camera_x = -edge_m;

    eprintln!(
        "spatial: K={} tuned to {:.3} MHz, hallway ±{:.1}m, walking {} m/s",
        k, tuned_to as f64 / 1e6, edge_m, speed_mps,
    );

    let mut detector = PeakDetector::new(PeakParams {
        sample_rate_hz: RADIO_SAMPLE_RATE_HZ as f32,
        fft_size: FFT_SIZE,
        open_threshold_db: 10.0,
        close_threshold_db: 6.0,
        hang_seconds: 0.5,
        rows_per_second: ROWS_PER_SECOND,
        min_separation_hz: 150_000.0,
        dc_skip_hz: 50_000.0,
    });
    let mut peak_events = Vec::new();
    let mut fleet = Fleet::new(k, tuned_to as f64);

    let mut spec = Spectrum::new(FFT_SIZE);
    let mut buf = vec![0u8; 262_144];
    let mut iq: Vec<Complex32> = Vec::with_capacity(buf.len() / 2);
    let mut scratch_c: Vec<Complex32> = Vec::new();
    let mut scratch_decim: Vec<Complex32> = Vec::new();
    let mut scratch_audio_hi: Vec<f32> = Vec::new();
    let mut scratch_audio: Vec<f32> = Vec::new();
    let mut stereo_mix: Vec<f32> = Vec::new();
    let mut pushed = 0usize;
    let mut iters: u64 = 0;
    let mut drops: u64 = 0;

    // block duration in audio seconds: ~131k IQ samples / 2.4M = ~54.6ms
    const BLOCK_SECONDS: f32 = (262_144 / 2) as f32 / RADIO_SAMPLE_RATE_HZ as f32;

    while !stop.load(Ordering::Relaxed) {
        let n = match hackrf.read(&mut buf) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("hackrf read error: {e:#}");
                continue;
            }
        };
        iq.clear();
        for chunk in buf[..n].chunks_exact(2) {
            let i = (chunk[0] as i8) as f32 / 128.0;
            let q = (chunk[1] as i8) as f32 / 128.0;
            iq.push(Complex32::new(i, q));
        }

        // advance the camera, ping-pong at edges
        spatial.camera_x += walk_dir * speed_mps * BLOCK_SECONDS;
        if spatial.camera_x > edge_m {
            spatial.camera_x = edge_m;
            walk_dir = -1.0;
        } else if spatial.camera_x < -edge_m {
            spatial.camera_x = -edge_m;
            walk_dir = 1.0;
        }

        // process active slots
        for slot in fleet.slots.iter_mut().filter(|s| s.freq_hz != 0.0) {
            slot.process(
                &iq,
                &mut scratch_c,
                &mut scratch_decim,
                &mut scratch_audio_hi,
                &mut scratch_audio,
            );
        }

        // collect (samples, freq) for the spatial mixer
        let streams = fleet
            .slots
            .iter()
            .filter(|s| s.freq_hz != 0.0)
            .map(|s| (s.out.as_slice(), s.freq_hz));
        spatial.mix(streams, &mut stereo_mix);

        for &s in &stereo_mix {
            if producer.push(s).is_err() {
                drops += 1;
            }
        }

        // FFT path: same cadence as fleet()
        let mut s_idx = 0;
        while s_idx + FFT_SIZE <= iq.len() {
            spec.push(&iq[s_idx..s_idx + FFT_SIZE]);
            pushed += 1;
            s_idx += FFT_SIZE;
            if pushed == AVG_FFTS {
                let row = spec.take();
                peak_events.clear();
                detector.step(&row, tuned_to as f64, &mut peak_events);
                let stations = detector.snapshot(tuned_to as f64);
                fleet.update(&stations);
                pushed = 0;
            }
        }

        iters += 1;
        if iters.is_multiple_of(20) {
            let parts: Vec<String> = fleet
                .slots
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    if s.freq_hz == 0.0 {
                        format!("slot{i}=idle")
                    } else {
                        let (lg, rg) = spatial.channel_gains(s.freq_hz);
                        format!(
                            "slot{i}={:.2}MHz L{:.2}R{:.2}",
                            s.freq_hz / 1e6,
                            lg,
                            rg
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
    hackrf.stop()?;
    eprintln!("\nspatial stopped cleanly. drops={drops}");
    Ok(())
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

    let ctrl_rx = stdin_reader_thread();
    let mut current_freq_hz: u64 = KMFA;

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
    // let mut mixer = Mixer::new(SHIFT_HZ as f32, RADIO_SAMPLE_RATE_HZ as f32);
    // let mut fir1 = Fir::new(63, 100_000.0, 2_400_000.0, 10);
    // let mut fm = FmDemod::new();
    // let mut fir2 = RealFir::new(63, 15_000.0, 240_000.0, 5);
    // let mut deemph = Deemph::new(75e-6, 48_000.0);

    let mut buf = vec![0u8; 262_144];
    let mut iq = Vec::with_capacity(buf.len() / 2);
    let mut mixed = Vec::with_capacity(buf.len() / 2);
    let mut decimated = Vec::new();
    let mut audio_hi = Vec::new();
    let mut audio: Vec<f32> = Vec::new();
    let mut deemphed = Vec::new();

    let mut drops: u64 = 0;
    let mut iters: u64 = 0;

    // let mut total_pushed = 0u64;
    // let start = std::time::Instant::now();
    // let mut total_bytes = 0u64;
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }

        // messages?
        let mut pipeline = Pipeline::fm();
        while let Ok(msg) = ctrl_rx.try_recv() {
            let new_freq = match msg {
                ControlMsg::SetMode(new_mode) => {
                    pipeline = match new_mode {
                        Mode::Fm => Pipeline::fm(),
                        Mode::Gmrs => Pipeline::gmrs(),
                    };
                    let mode = new_mode;
                    current_freq_hz = default_freq_for_mode(mode);
                    // retune(&hackrf, current_freq_hz)?;
                    eprintln!("switched to {}", mode_name(mode));
                    current_freq_hz
                }
                ControlMsg::RetuneAbs(f) => f,
                ControlMsg::RetuneRel(d) => (current_freq_hz as i64 + d).max(1) as u64,
            };
            match retune(&hackrf, new_freq) {
                Ok(()) => {
                    current_freq_hz = new_freq;
                    eprintln!("tuned to {:.3} Mhz", new_freq as f64 / 1e6);
                }
                Err(e) => eprintln!("retune error: {e}"),
            }
        }

        let n = hackrf.read(&mut buf)?;
        // total_bytes += n as u64;

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

        pipeline.process(
            &iq,
            &mut mixed,
            &mut decimated,
            &mut audio_hi,
            &mut deemphed,
        );
        // fir1.process(&mixed, &mut decimated);
        // fm.process(&decimated, &mut audio_hi);
        // fir2.process(&audio_hi, &mut audio);
        // deemph.process(&audio, &mut deemphed);

        for &sample in &deemphed {
            if producer.push(sample).is_err() {
                drops += 1;
            } /*else {
            total_pushed += 1;
            }*/
        }

        iters += 1;
        if iters.is_multiple_of(30) {
            // let elapsed = start.elapsed().as_secs_f64();
            // let pops = audio_out.pops.load(std::sync::atomic::Ordering::Relaxed);
            // let fill = producer.buffer().capacity() - producer.slots();
            // let underruns = audio_out
            //     .underruns
            //     .load(std::sync::atomic::Ordering::Relaxed);
            // eprintln!("iters={iters} drops={drops} under={underruns} fill={fill}",);
        }
    }
    hackrf.stop()?;
    eprintln!("\nstopped cleanly. drops={drops}");
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ControlMsg {
    RetuneAbs(u64),
    RetuneRel(i64),
    SetMode(Mode),
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Mode {
    Fm,
    Gmrs,
}

pub(crate) fn parse_control_msg(line: &str) -> Option<ControlMsg> {
    let s = line.trim();
    if s.is_empty() {
        return None;
    }
    if s == "+" {
        return Some(ControlMsg::RetuneRel(200_000));
    } else if s == "-" {
        return Some(ControlMsg::RetuneRel(-200_000));
    }

    if let Some(rest) = s.strip_prefix("mode ") {
        return match rest.trim() {
            "fm" => Some(ControlMsg::SetMode(Mode::Fm)),
            "gmrs" => Some(ControlMsg::SetMode(Mode::Gmrs)),
            _ => None,
        };
    }

    let (num_str, is_mhz_marker) = if let Some(stripped) = s.strip_suffix(['M', 'm']) {
        (stripped, true)
    } else {
        (s, false)
    };

    let n: f64 = num_str.parse().ok()?;
    let frequency_hz = if is_mhz_marker || n < 1000.0 {
        (n * 1_000_000.0) as u64
    } else {
        n as u64
    };
    Some(ControlMsg::RetuneAbs(frequency_hz))
}

fn stdin_reader_thread() -> mpsc::Receiver<ControlMsg> {
    let (tx, rx) = mpsc::channel::<ControlMsg>();
    spawn_stdin_reader(tx, "play");
    rx
}

pub(crate) fn spawn_stdin_reader(tx: mpsc::Sender<ControlMsg>, prompt: &'static str) {
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut buf = String::new();
        loop {
            buf.clear();
            print!("{prompt}> ");
            std::io::Write::flush(&mut std::io::stdout()).ok();
            if stdin.lock().read_line(&mut buf).unwrap_or(0) == 0 {
                break;
            }
            match parse_control_msg(&buf) {
                Some(msg) => {
                    if tx.send(msg).is_err() {
                        break;
                    }
                }
                None => {
                    if !buf.trim().is_empty() {
                        eprintln!("? unrecognized: {}", buf.trim());
                    }
                }
            }
        }
    });
}

pub(crate) fn retune(hackrf: &HackRf, target_rf_hz: u64) -> anyhow::Result<()> {
    let tuned_to = target_rf_hz + 200_000;
    hackrf.set_freq(tuned_to)?;
    Ok(())
}

pub(crate) fn default_freq_for_mode(mode: Mode) -> u64 {
    match mode {
        Mode::Fm => 89_500_000,
        Mode::Gmrs => gmrs::CHANNEL_1,
    }
}

fn step_for_mode(mode: Mode) -> i32 {
    match mode {
        Mode::Fm => 200_000,
        Mode::Gmrs => 12_500,
    }
}

pub(crate) fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Fm => "FM",
        Mode::Gmrs => "GMRS",
    }
}
