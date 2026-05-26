use std::{io::BufRead, sync::mpsc};

use crate::gmrs;

#[derive(Debug, Clone, Copy)]
pub enum ControlMsg {
    RetuneAbs(u64),
    RetuneRel(i64),
    SetMode(Mode),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Fm,
    Gmrs,
}

pub fn default_freq_for_mode(mode: Mode) -> u64 {
    match mode {
        Mode::Fm => 89_500_000,
        Mode::Gmrs => gmrs::CHANNEL_1,
    }
}

pub fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Fm => "FM",
        Mode::Gmrs => "GMRS",
    }
}

pub fn parse_control_msg(line: &str) -> Option<ControlMsg> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed == "+" {
        return Some(ControlMsg::RetuneRel(200_000));
    }
    if trimmed == "-" {
        return Some(ControlMsg::RetuneRel(-200_000));
    }

    if let Some(rest) = trimmed.strip_prefix("mode ") {
        return match rest.trim() {
            "fm" => Some(ControlMsg::SetMode(Mode::Fm)),
            "gmrs" => Some(ControlMsg::SetMode(Mode::Gmrs)),
            _ => None,
        };
    }

    // Bare numeric input: treat as freq. `M`/`m` suffix or values < 1000 mean
    // MHz; anything larger is assumed already in Hz.
    let (num_str, mhz_marker) = if let Some(stripped) = trimmed.strip_suffix(['M', 'm']) {
        (stripped, true)
    } else {
        (trimmed, false)
    };
    let parsed: f64 = num_str.parse().ok()?;
    let frequency_hz = if mhz_marker || parsed < 1000.0 {
        (parsed * 1_000_000.0) as u64
    } else {
        parsed as u64
    };
    Some(ControlMsg::RetuneAbs(frequency_hz))
}

pub fn spawn_stdin_reader(tx: mpsc::Sender<ControlMsg>, prompt: &'static str) {
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

pub fn stdin_reader_channel(prompt: &'static str) -> mpsc::Receiver<ControlMsg> {
    let (tx, rx) = mpsc::channel::<ControlMsg>();
    spawn_stdin_reader(tx, prompt);
    rx
}
