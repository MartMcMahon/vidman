use std::{
    fs::File,
    io::Read,
    time::{Duration, Instant},
};

use anyhow::Result;
use num_complex::Complex32;
use ratatui::{
    crossterm::event::{self, Event, KeyCode},
    text::Line,
    widgets::{Block, Borders, Paragraph},
};

use crate::dsp::fft::Spectrum;

const FFT_SIZE: usize = 1024;
const SAMPLE_RATE: f32 = 2_400_000.0;
const FRAME_HZ: f32 = 30.0;
const SAMPLES_PER_FRAME: usize = (SAMPLE_RATE / FRAME_HZ) as usize; // ~80_000
const DB_MIN: f32 = -80.0;
const DB_MAX: f32 = 0.0;

pub fn run() -> Result<()> {
    let mut file = File::open("cap.cs8")?;
    let mut spec = Spectrum::new(FFT_SIZE);
    let mut backend = ratatui::init();
    let frame_dur = Duration::from_secs_f32(1.0 / FRAME_HZ);

    let mut byte_buf = vec![0u8; FFT_SIZE * 2];
    let mut chunk = vec![Complex32::new(0.0, 0.0); FFT_SIZE];

    'outer: loop {
        let frame_start = Instant::now();

        // read samples for one frame and send to spectrum
        let mut samples_consumed = 0;
        while samples_consumed < SAMPLES_PER_FRAME {
            let n = file.read(&mut byte_buf)?;
            if n < byte_buf.len() {
                // EOF; loop playback
                file = File::open("cap.cs8")?;
                continue;
            }
            for i in 0..FFT_SIZE {
                let re = (byte_buf[2 * i] as i8) as f32 / 128.0;
                let im = (byte_buf[2 * i + 1] as i8) as f32 / 128.0;
                chunk[i] = Complex32::new(re, im);
            }
            spec.push(&chunk);
            samples_consumed += FFT_SIZE;
        }

        let bins = spec.take();
        backend.draw(|f| {
            let area = f.area();
            let line = render_spectrum(&bins, area.width as usize, area.height as usize - 2);
            let p = Paragraph::new(line)
                .block(Block::default().borders(Borders::ALL).title(" spectrum "));
            f.render_widget(p, area);
        })?;

        if event::poll(Duration::from_millis(0))? {
            if let Event::Key(k) = event::read()? {
                if k.code == KeyCode::Char('q') {
                    break 'outer;
                }
            }
        }

        let elapsed = frame_start.elapsed();
        if elapsed < frame_dur {
            std::thread::sleep(frame_dur - elapsed);
        }
    }

    ratatui::restore();
    Ok(())
}

fn render_spectrum(bins: &[f32], width: usize, height: usize) -> Vec<Line<'static>> {
    // Down-sample (or pick every Nth) bin to fit terminal width.
    let step = bins.len() as f32 / width as f32;
    let cols: Vec<f32> = (0..width)
        .map(|c| {
            let start = (c as f32 * step) as usize;
            let end = ((c + 1) as f32 * step) as usize;
            // Take max in this column's bin range so peaks survive downsampling.
            bins[start..end.min(bins.len())]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect();

    // Each column gets a height in fractional rows.
    let heights: Vec<f32> = cols
        .iter()
        .map(|&db| {
            let norm = ((db - DB_MIN) / (DB_MAX - DB_MIN)).clamp(0.0, 1.0);
            norm * height as f32
        })
        .collect();

    // Build the display row by row from top to bottom.
    let block_levels = ['█', '▇', '▆', '▅', '▄', '▃', '▂', '▁'];
    let mut lines = Vec::with_capacity(height);
    for row in 0..height {
        let row_from_bottom = height - 1 - row;
        let mut s = String::with_capacity(width);
        for &h in &heights {
            let h_floor = h.floor() as usize;
            if row_from_bottom < h_floor {
                s.push('█');
            } else if row_from_bottom == h_floor {
                let frac = h - h_floor as f32;
                let idx = ((1.0 - frac) * 7.0).round() as usize;
                s.push(block_levels[idx.min(7)]);
            } else {
                s.push(' ');
            }
        }
        lines.push(Line::raw(s));
    }
    lines
}
