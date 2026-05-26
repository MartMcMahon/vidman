use clap::{Parser, Subcommand};

use crate::radio::{KMFA, parse_freq};

mod audio;
mod commands;
mod control;
mod dsp;
mod gmrs;
mod hallway;
mod radio;
mod waterfall;

#[derive(Parser)]
#[command()]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Capture,
    Demod,
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
        Commands::Capture => commands::capture::run(),
        Commands::Demod => commands::demod::run().expect("demod"),
        Commands::Play => commands::play::run().expect("play"),
        Commands::Waterfall => waterfall::run(KMFA).expect("waterfall"),
        Commands::Scan { freq, narrow } => {
            commands::scan::run(parse_freq(&freq), narrow).expect("scan")
        }
        Commands::Fleet { freq, k } => commands::fleet::run(parse_freq(&freq), k).expect("fleet"),
        Commands::Spatial { freq, k, speed } => {
            commands::spatial::run(parse_freq(&freq), k, speed).expect("spatial")
        }
        Commands::Hallway { freq } => hallway::run(parse_freq(&freq)).expect("hallway"),
    }
}
