//! ft_linear_regression training module
//!
//! As per [subject](../../../../resources/ft_linear_regression.en.pdf) the followig formulas are used:
//! - price estimation `estimatePrice(mileage) = theta0 + (theta1 * mileage)`
//! - training formulas
//!     - tmp_theta0 = learningRate * (1 / m) * Sum(i=0; m-1)(estimatePrice(mileage[i]) - price[i])`
//!     - tmp_theta1 = learningRate * (1 / m) * Sum(i=0; m-1)(estimatePrice(mileage[i]) - price[i]) * mileage[i]`
//!     - where m is length of the dataset
use std::collections::HashMap;
use std::fs;

use super::keyf64::KeyF64;

/// Main training struct
#[derive(Debug)]
pub struct Trainer {
    /// Context struct
    ctx: TrainerContext,
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
    pub fn load(filename: &str, ctx: Option<TrainerContext>) -> Self {
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
        Self {
            ctx: ctx.unwrap_or_default(),
            data,
            labels,
        }
    }
}

/// Context struct for trainer.
///
/// To-Do:
///     - change paths to `&str` to avoid pointless allocations
///     - paths could be options, if none -> stdout
#[derive(Debug)]
pub struct TrainerContext {
    /// Ratio of the dataset to be used for training. Rest is used for testing precision.
    ///
    /// The value is always adjusted to lie between 0 and 1.
    /// Default value is `0.7`, meaning 70% of the dataset is used to train
    /// and 30% is used to measure precision of the algorithm.
    /// The function responsible for the dataset splitting attempts to
    /// distribute values evently across given range.
    training_distribution: f32,
    /// Learning rate
    ///
    /// By default it is set to `0.1`.
    /// Learning rate *alpha* as generally used in ML / AI slang.
    learning_rate: f64,
    /// Theta for the linear regression equations
    ///
    /// By default it starts at 0.0, 0.0
    theta: (f64, f64),
    /// Path to file to save training results
    ///
    /// By default this is set to `data/weights`.
    outfile: String,
    /// Path to directory to store statistics for the training
    ///
    /// By default this is set to `stats/`.
    stats_dir: String,
}

impl TrainerContext {}

impl Default for TrainerContext {
    fn default() -> Self {
        Self {
            training_distribution: 0.7,
            learning_rate: 0.1,
            theta: (0.0, 0.0),
            outfile: String::from("data/weights"),
            stats_dir: String::from("stats/"),
        }
    }
}
