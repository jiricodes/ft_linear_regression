use std::collections::HashMap;
use std::fs;

use clap::{crate_authors, crate_name, crate_version};
use clap::{App, Arg};

#[derive(Debug)]
struct Trainer {
    data: HashMap<u64, u64>,
    labels: [String; 2],
}

impl Trainer {
    pub fn new(filename: &str) -> Self {
        let contents =
            fs::read_to_string(filename).expect(&format!("Reading \"{}\" file failed", filename));
        let mut data: HashMap<u64, u64> = HashMap::new();
        let mut labels: [String; 2] = [String::default(), String::default()];
        for (line_num, line) in contents.lines().enumerate() {
            if line.len() == 0 {
                continue;
            }
            let mut split_line = line.split(',');
            if line_num != 0 {
                let km: u64 = split_line
                    .next()
                    .unwrap_or_else(|| "")
                    .trim()
                    .parse()
                    .unwrap();
                let price: u64 = split_line
                    .next()
                    .unwrap_or_else(|| "")
                    .trim()
                    .parse()
                    .unwrap();
                data.insert(km, price);
            } else {
                labels[0] = split_line
                    .next()
                    .unwrap_or_else(|| "")
                    .trim()
                    .parse()
                    .unwrap();
                labels[1] = split_line
                    .next()
                    .unwrap_or_else(|| "")
                    .trim()
                    .parse()
                    .unwrap();
            }
        }
        Self { data, labels }
    }
}

fn main() {
    let matches = App::new(crate_name!())
        .author(crate_authors!("\n"))
        .version(crate_version!())
        .arg(
            Arg::with_name("datafile")
                .short("f")
                .long("file")
                .takes_value(true)
                .help("Input data file")
                .required(true),
        )
        .get_matches();
    let filename = matches.value_of("datafile").unwrap();
    println!("Input data location {}", filename);
    let trainer = Trainer::new(filename);
    dbg!(trainer);
}
