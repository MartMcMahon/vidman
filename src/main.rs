use std::{fs::File, io::Write, time::Instant};

use seify_hackrfone::{Config, HackRf};

fn main() {
    let hackrf = HackRf::open_first().expect("opening hackrf device");
    hackrf
        .start_rx(&Config {
            txvga_db: 0,
            vga_db: 20,
            lna_db: 16,
            amp_enable: false,
            antenna_enable: false,
            frequency_hz: 100_900_000,
            sample_rate_hz: 2_400_000,
            sample_rate_div: 1,
        })
        .expect("error with config}");
    let mut buf = vec![0u8; 262_144];

    let mut elapsed = 0;
    let duration = 1;
    let start_t = Instant::now();
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
        if total_bytes_written >= 2_400_000 * 2 * 3 {
            break;
        }
    }
    // hackrf.read(&mut buf).expect("error reading");
    // let n = buf.len()
    hackrf.stop().expect("stopping");
}

// Roughly (from the docs.rs surface):
// HackRf::open_first() → HackRf
// hackrf.start_rx(&Config { vga_db, lna_db, amp_enable, antenna_enable,
//                           frequency_hz, sample_rate_hz, sample_rate_div }) → ()
// hackrf.read(&mut buf) → bytes_read     // buf is &mut [u8]
// hackrf.stop_rx() → ()
//
// Buffer-based blocking read, not a callback model — simpler than you might be bracing for.
//
// M1's loop, in pseudocode (your code, not mine)
//
// open device
// configure: freq = args.freq           (tune directly at the station for M1 —
//            sample_rate_hz = 2_400_000  see note below)
//            gains from args
// start_rx
// open output file
// loop {
//     read into a buffer (say 256 KB — that's ~50 ms at 2.4 Msps)
//     write buffer verbatim to file
//     if elapsed >= duration: break
// }
// stop_rx
//
// A small but useful M1 choice
//
// The plan calls for tuning 200 kHz above the station everywhere. For M1 specifically, I'd actually suggest you tune directly at the station. Reason: when you open the
// resulting .cs8 in GQRX, you'll see the DC spike sitting right on top of the FM signal — that's a visceral, "oh that's what they were talking about" moment. Then M3 is
//  the lesson where we fix it. Tuning-offset starts in M3.
//
