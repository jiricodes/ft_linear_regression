//! # ft_linear_regression training module
//!
//! As per [subject](../../../../resources/ft_linear_regression.en.pdf) the followig formulas are used:
//! - price estimation `estimatePrice(mileage) = theta0 + (theta1 * mileage)`
//! - training formulas
//!     - `tmp_theta0 = learningRate * (1 / m) * Sum(i=0; m-1)(estimatePrice(mileage[i]) - price[i])`
//!     - `tmp_theta1 = learningRate * (1 / m) * Sum(i=0; m-1)(estimatePrice(mileage[i]) - price[i]) * mileage[i]`
//!     - where m is length of the dataset
use std::fs;
use std::fs::File;
use std::io::prelude::*;

use rand::distributions::Standard;
use rand::prelude::*;

use plotters::prelude::*;

use super::arguments::CmdArgs;

/// Main training struct
#[derive(Debug)]
pub struct Trainer {
	/// Context struct
	ctx: TrainerContext,
	/// Key value pairs of data for linear regression
	data: Vec<(f64, f64)>,
	/// labels for key value pairs in `self.data`
	labels: [String; 2],
	/// Test set
	test_set: Vec<(f64, f64)>,
	/// Training set
	train_set: Vec<(f64, f64)>,
}

impl Trainer {
	/// Function that loads the `Trainer` struct from given file.
	///
	/// Assumptions: Takes only two columns, first line contains labels of those columns,
	/// each line is parsed to f64
	///
	pub fn load(filename: &str, ctx: Option<TrainerContext>) -> Self {
		let contents =
			fs::read_to_string(filename).expect(&format!("Reading \"{}\" file failed", filename));
		let mut data: Vec<(f64, f64)> = Vec::new();
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
				data.push((x, y));
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
			test_set: Vec::new(),
			train_set: Vec::new(),
		}
	}

	/// Splits the input dataset into train and test sets. Returns (min_key. max_key)
	/// for normalisation.
	///
	/// The split is random, in preset ratio (`TrainerContext::training_distribution`).
	/// Seed can be saved for reproducibility.
	///
	fn split_dataset(&mut self) -> (f64, f64) {
		// Init rng pool
		let mut r = StdRng::seed_from_u64(self.ctx.get_seed());
		// Init min and max
		let mut min_key = std::f64::MAX;
		let mut max_key = std::f64::MIN;
		// Counter for test dataset
		let mut test_count: i32 =
			(self.data.len() as f32 * (1.0 - self.ctx.training_distribution)) as i32;
		for (key, value) in self.data.iter() {
			if r.sample::<bool, _>(Standard) && test_count > 0 {
				self.test_set.push((*key, *value));
				test_count -= 1;
			} else {
				self.train_set.push((*key, *value));
				min_key = min_key.min(*key);
				max_key = max_key.max(*key);
			}
		}
		(min_key, max_key)
	}

	pub fn train(&mut self) {
		// Shoud segment dataset to training and testing sets here
		// PLACEHOLDER
		let extremes = self.split_dataset();
		// normalize keys
		for (key, _) in self.train_set.iter_mut() {
			*key = (*key - extremes.0) / (extremes.1 - extremes.0);
		}
		// Get training set len and invert it, so we don't need to div in each loop
		// m_ratio == 1 / m as in formula.
		let m_ratio = 1.0 / self.train_set.len() as f64;

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
				self.train_set.iter().fold(0.0, |acc, &val| {
					acc + (self.ctx.theta.0 + self.ctx.theta.1 * val.0) - val.1
				}),
				self.train_set.iter().fold(0.0, |acc, &val| {
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
			// if i % 100 == 0 {
			// 	println!("i {}: TD {:?}\n\t THETA {:?}", i, temp_diff, self.ctx.theta);
			// }
		}

		println!(
			"Training finished after {} iterations.\nTemporal difference {:?}",
			i, temp_diff
		);

		// Scale theta1 back
		self.ctx.theta.1 = self.ctx.theta.1 / (extremes.1 - extremes.0);
	}

	pub fn test_accuracy(&self) {
		if self.test_set.len() == 0 {
			println!("No test set available");
			return;
		}
		// Accumulated error
		let mut acc: f64 = 0.0;
		// Loop over test set
		for (key, val) in self.test_set.iter() {
			// estimate value of a key
			let est = self.ctx.theta.0 + (self.ctx.theta.1 * key);
			acc += (val - est).abs() / est;
		}
		let avg_error = acc / self.test_set.len() as f64;
		println!("Average error ~{:.3}", avg_error);
	}

	fn get_bounding_box(&self, offset: f64) -> (f64, f64, f64, f64) {
		let mut x_min = std::f64::MAX;
		let mut x_max = std::f64::MIN;
		let mut y_min = std::f64::MAX;
		let mut y_max = std::f64::MIN;
		// Find min and max values
		for (x, y) in self.data.iter() {
			x_min = x_min.min(*x);
			x_max = x_max.max(*x);
			y_min = y_min.min(*y);
			y_max = y_max.max(*y);
		}
		// calculate x and y offsets
		let x_off = (x_max - x_min) * offset;
		let y_off = (y_max - y_min) * offset;

		(x_min - x_off, y_min - y_off, x_max + x_off, y_max + y_off)
	}
	/// Saves labels and theta values into a file.
	///
	/// If the path_overwrite is `Option::None`, then `TrainerContext::outfile` is used.
	pub fn save_output(&self, path_overwrite: Option<&str>) -> std::io::Result<()> {
		let mut file = match path_overwrite.is_some() {
			true => File::create(path_overwrite.unwrap())?,
			false => File::create(&self.ctx.outfile)?,
		};
		let out = format!(
			"{} {}\n{} {}\n",
			self.labels[0], self.labels[1], self.ctx.theta.0, self.ctx.theta.1
		);
		file.write_all(out.as_bytes())?;
		Ok(())
	}

	pub fn plot_result(&self) {
		let path = format!("{}/result.png", self.ctx.stats_dir);

		let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
		root.fill(&WHITE).expect("Failed to fill the graph");
		let root = root
			.titled("ft_linear_regression", ("sans-serif", 40))
			.unwrap();

		let bbox = self.get_bounding_box(0.1);
		let mut scatter_ctx = ChartBuilder::on(&root)
			.x_label_area_size(40)
			.y_label_area_size(80)
			.caption(
				&format!(
					"seed: {:?}; ratio {}",
					self.ctx.rng_seed, self.ctx.training_distribution
				),
				("sans-serif", 16),
			)
			.build_cartesian_2d(bbox.0..bbox.2, bbox.1..bbox.3)
			.unwrap();
		scatter_ctx.configure_mesh().draw().unwrap();
		scatter_ctx
			.draw_series(
				self.data
					.iter()
					.map(|(x, y)| Circle::new((*x, *y), 3, GREEN.filled())),
			)
			.unwrap();
		let x_axis = vec![bbox.0, bbox.2];
		scatter_ctx
			.draw_series(LineSeries::new(
				x_axis
					.iter()
					.map(|x| (*x, self.ctx.theta.0 + self.ctx.theta.1 * *x)),
				&RED,
			))
			.expect("Couldn't draw line");
		// To avoid the IO failure being ignored silently, we manually call the present function
		root.present().expect(&format!(
			"Unable to write result to file. Make sure directory {} exists",
			self.ctx.stats_dir
		));
		println!("Result has been saved to {}", path);
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
	/// Default value is `0.8`, meaning 4/5 of the dataset is used to train
	/// and 1/5 is used to test the model.
	/// The choice is scenariondependant, but commonly used are:
	/// - 0.8 -> 4/5 train & 1/5 test
	/// - 0.67 -> 2/3 train & 1/3 test
	/// - 0.5 -> 1/2 train & 1/2 test
	training_distribution: f32,
	/// Random seed, for reproducibility.
	/// By default set to None, which means random seed.
	rng_seed: Option<u64>,
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
	/// Helper function to decide whether the main linear regression loop should continue
	///
	/// The training is done if number of iteration is reached or if not specified
	/// the temporal difference limit is reached
	fn is_done(&self, current_iter: usize, temp_diff: (f64, f64)) -> bool {
		match self.iterations.is_some() {
			true => current_iter >= self.iterations.unwrap(),
			false => {
				self.temp_diff_limit >= temp_diff.0.abs()
					&& self.temp_diff_limit >= temp_diff.1.abs()
			}
		}
	}
	/// Random seed setter
	#[cfg(test)]
	pub fn set_seed(&mut self, seed: u64) {
		self.rng_seed = Some(seed);
	}
	/// Random seed getter
	///
	/// If the seed is None, new seed is randomly generated, assigned and returned
	fn get_seed(&mut self) -> u64 {
		if self.rng_seed.is_none() {
			self.rng_seed = Some(thread_rng().gen::<u64>());
		}
		self.rng_seed.unwrap()
	}
}

impl Default for TrainerContext {
	fn default() -> Self {
		Self {
			training_distribution: 0.8,
			rng_seed: Option::None,
			learning_rate: 0.1,
			theta: (0.0, 0.0),
			outfile: String::from("data/weights"),
			stats_dir: String::from("stats/"),
			temp_diff_limit: 0.001,
			iterations: Option::None,
		}
	}
}

impl<'a> From<&CmdArgs<'a>> for TrainerContext {
	fn from(cmdargs: &CmdArgs<'a>) -> Self {
		let seed = match cmdargs.matches.is_present("seed") {
			true => Some(
				cmdargs
					.matches
					.value_of("seed")
					.unwrap()
					.parse::<u64>()
					.expect("Couldn't parse seed"),
			),
			false => Option::None,
		};
		let iterations = match cmdargs.matches.is_present("iter") {
			true => Some(
				cmdargs
					.matches
					.value_of("iter")
					.unwrap()
					.parse::<usize>()
					.expect("Couldn't parse iterations"),
			),
			false => Option::None,
		};
		Self {
			training_distribution: cmdargs
				.matches
				.value_of("ratio")
				.unwrap_or("0.8")
				.parse::<f32>()
				.expect("Couldn't parse distribution ratio"),
			rng_seed: seed,
			learning_rate: cmdargs
				.matches
				.value_of("alpha")
				.unwrap_or("0.1")
				.parse::<f64>()
				.expect("Couldn't parse alpha"),
			theta: (0.0, 0.0),
			outfile: cmdargs
				.matches
				.value_of("outfile")
				.unwrap_or("data/weights")
				.to_string(),
			stats_dir: cmdargs
				.matches
				.value_of("stats")
				.unwrap_or("stats/")
				.to_string(),
			temp_diff_limit: cmdargs
				.matches
				.value_of("tdlimit")
				.unwrap_or("0.001")
				.parse::<f64>()
				.expect("Couldn't parse temporal difference limit"),
			iterations,
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;

	fn do_vecs_match<T: PartialEq>(a: &Vec<T>, b: &Vec<T>) -> bool {
		let matching = a.iter().zip(b.iter()).filter(|&(a, b)| a == b).count();
		matching == a.len() && matching == b.len()
	}

	#[test]
	fn rng_seed_get() {
		let mut ctx = TrainerContext::default();
		let x = ctx.get_seed();
		let y = ctx.get_seed();
		assert_eq!(x, y);
	}

	#[test]
	fn dataset_split_length() {
		let mut trainer = Trainer::load("data/subject_data.csv", Option::None);
		trainer.split_dataset();
		assert_eq!(trainer.train_set.len(), 20);
		assert_eq!(trainer.test_set.len(), 4);
	}

	#[test]
	fn dataset_split_per_seed() {
		let mut trainer_one = Trainer::load("data/subject_data.csv", Option::None);
		let seed = trainer_one.ctx.get_seed();
		let mut trainer_two = Trainer::load("data/subject_data.csv", Option::None);
		trainer_two.ctx.set_seed(seed);
		assert_eq!(seed, trainer_two.ctx.get_seed());
		let extemes_one = trainer_one.split_dataset();
		let extremes_two = trainer_two.split_dataset();
		assert_eq!(extemes_one, extremes_two);
		assert!(do_vecs_match(
			&trainer_one.train_set,
			&trainer_two.train_set
		));
		assert!(do_vecs_match(&trainer_one.test_set, &trainer_two.test_set));
	}
}
