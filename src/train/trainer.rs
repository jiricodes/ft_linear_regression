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

    pub fn train(&mut self) {
        // Shoud segment dataset to training and testing sets here
        // PLACEHOLDER

        // Prep training set
        // The dataset should be standardized / normalized here
        let mut training_set: Vec<(f64, f64)> = Vec::new();
        let mut min_key = std::f64::MAX;
        let mut max_key = std::f64::MIN;

        // Do I need to do this?
        // Collect the data into vector
        for (key, value) in self.data.iter() {
            let k: f64 = key.into();
            training_set.push((k, *value));
            min_key = min_key.min(k);
            max_key = max_key.max(k);
        }
        dbg!(&training_set);
        dbg!(&min_key);
        dbg!(&max_key);
        // normalize keys
        for (key, _) in training_set.iter_mut() {
            *key = (*key - min_key) / (max_key - min_key);
        }
        dbg!(&training_set);
        // Get training set len and invert it, so we don't need to div in each loop
        // m_ratio == 1 / m as in formula.
        let m_ratio = 1.0 / training_set.len() as f64;

        let mut i: usize = 0;

        // Temporal difference init
        let mut temp_diff: (f64, f64) = (1.0, 1.0);

        // Main loop
        // Runs while iterations limit or precision is not reached
        while !self.ctx.is_done(i, temp_diff) {
            // Calculate sum here
            // sum(theta.0, theta.1)
            // TO be revisited
            let sum: (f64, f64) = (
                training_set.iter().fold(0.0, |acc, &val| {
                    acc + (self.ctx.theta.0 + self.ctx.theta.1 * val.0) - val.1
                }),
                training_set.iter().fold(0.0, |acc, &val| {
                    acc + ((self.ctx.theta.0 + self.ctx.theta.1 * val.0) - val.1) * val.0
                }),
            );

            // Update temporal difference
            temp_diff.0 = self.ctx.learning_rate * m_ratio * sum.0;
            temp_diff.1 = self.ctx.learning_rate * m_ratio * sum.1;
            // Update theta
            self.ctx.theta.0 -= temp_diff.0;
            self.ctx.theta.1 -= temp_diff.1;

            // increase iteration count
            i += 1;
            if i % 100 == 0 {
                println!("i {}: TD {:?}\n\t THETA {:?}", i, temp_diff, self.ctx.theta);
            }
        }

        println!(
            "Training finished after {} iterations.\nTemporal difference {:?}",
            i, temp_diff
        );

        // Scale theta1 back
        self.ctx.theta.1 = self.ctx.theta.1 / (max_key - min_key);
        dbg!(&self.ctx);
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
    /// Temporal Difference limit
    ///
    /// By default this is set to `0.001`.
    /// The linear regression will loop until reaching this limit.
    /// If iteration limit is set, the loop will prioritize that value over this limit.
    temp_diff_limit: f64,
    /// Iterations limit
    ///
    /// By default this is set to `Option::None`.
    /// If this value is set, then the linear regression loop will
    /// stop upon reaching this number and ignore `temp_diff_limit`.
    iterations: Option<usize>,
}

impl TrainerContext {
    fn is_done(&self, current_iter: usize, temp_diff: (f64, f64)) -> bool {
        match self.iterations.is_some() {
            true => current_iter >= self.iterations.unwrap(),
            false => {
                self.temp_diff_limit >= temp_diff.0.abs()
                    && self.temp_diff_limit >= temp_diff.1.abs()
            }
        }
    }
}

impl Default for TrainerContext {
    fn default() -> Self {
        Self {
            training_distribution: 0.7,
            learning_rate: 0.1,
            theta: (0.0, 0.0),
            outfile: String::from("data/weights"),
            stats_dir: String::from("stats/"),
            temp_diff_limit: 0.001,
            iterations: Option::None,
        }
    }
}
