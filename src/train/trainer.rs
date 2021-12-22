//! ft_linear_regression training module
//!
//!
use std::collections::HashMap;
use std::fs;

use super::keyf64::KeyF64;

/// Main training struct
#[derive(Debug)]
pub struct Trainer {
    /// Key value pairs of data for linear regression
    data: HashMap<KeyF64, f64>,
    /// labels for key value pairs in `self.data`
    labels: [String; 2],
}

impl Trainer {
    /// Function that loads the `Trainer` struct from given file.
    ///
    /// There are certain assuptions about the file format... TBC
    ///
    /// Example:
    ///
    /// TBC
    ///
    pub fn load(filename: &str) -> Self {
        let contents =
            fs::read_to_string(filename).expect(&format!("Reading \"{}\" file failed", filename));
        let mut data: HashMap<KeyF64, f64> = HashMap::new();
        let mut labels: [String; 2] = [String::default(), String::default()];
        for (line_num, line) in contents.lines().enumerate() {
            if line.len() == 0 {
                continue;
            }
            let mut split_line = line.split(',');
            if line_num != 0 {
                let x: f64 = split_line
                    .next()
                    .unwrap_or_else(|| "")
                    .trim()
                    .parse()
                    .unwrap();
                let y: f64 = split_line
                    .next()
                    .unwrap_or_else(|| "")
                    .trim()
                    .parse()
                    .unwrap();
                data.insert(x.into(), y);
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
