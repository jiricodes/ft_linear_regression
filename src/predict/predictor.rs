//! Module responsible for predictions
//!
//! Handles model loading and predictions based on input.
use std::fs;
use std::io::Result;

#[derive(Debug)]
pub struct Predictor {
	labels: [String; 2],
	theta: (f64, f64),
}

impl Predictor {
	/// Loads model from a file, panics on error
	///
	/// Expects predefined format:
	/// ```text
	/// x_label y_label
	/// theta_0 theta_1
	/// ```
	/// where labels are string and theta is [`f64`](https://doc.rust-lang.org/std/primitive.f64.html) compatible
	///
	pub fn load(filename: &str) -> Result<Self> {
		let mut labels: [String; 2] = [String::new(), String::new()];
		let mut theta: (f64, f64) = (0.0, 0.0);
		let contents = fs::read_to_string(filename)?;
		let mut lines = contents.lines();
		// Take first line and split it. Assign labels.
		let mut split = lines.next().expect("Failed to read first line").split(' ');
		labels[0] = split.next().unwrap_or("").trim().parse().unwrap();
		labels[1] = split.next().unwrap_or("").trim().parse().unwrap();
		// Take second line and split it. Assign thetas.
		let mut split = lines.next().expect("Failed to read second line").split(' ');
		theta.0 = split
			.next()
			.unwrap_or("")
			.trim()
			.parse()
			.expect("Failed to parse theta0");
		theta.1 = split
			.next()
			.unwrap_or("")
			.trim()
			.parse()
			.expect("Failed to parse theta1");
		Ok(Self { labels, theta })
	}

	/// Labels getter
	pub fn get_labels(&self) -> &[String; 2] {
		&self.labels
	}

	/// Makes the prediction for given value
	///
	/// Uses formula:
	/// ```text
	/// estimate = theta.0 + (theta.1 * value)
	/// ```
	pub fn predict(&self, value: f64) {
		let estimate = self.theta.0 + (self.theta.1 * value);
		println!(
			"The estimate for {} [{}] is {:.3} [{}].",
			value, self.labels[0], estimate, self.labels[1]
		);
	}
}
