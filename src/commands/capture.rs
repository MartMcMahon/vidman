use std::{fs::File, io::Write};

use seify_hackrfone::{Config, HackRf};

pub fn run() {
    let hackrf = HackRf::open_first().expect("opening hackrf device");
    hackrf
        .start_rx(&Config {
            txvga_db: 0,
            vga_db: 20,
            lna_db: 16,
            amp_enable: false,
            antenna_enable: false,
            frequency_hz: 89_700_000, // KMFA Classical FM89.5 + 200 kHz LO offset
            sample_rate_hz: 2_400_000,
            sample_rate_div: 1,
        })
        .expect("error with config");

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
