//! # Linear Regression Predictor
//! Uses a model trained by `train` crate to estimate values from a given key.
//! ## Usage
//! ```text
//! USAGE:
//! predict [OPTIONS] --modelfile <model>
//!
//! FLAGS:
//!     -h, --help       Prints help information
//!     -V, --version    Prints version information
//!
//! OPTIONS:
//!     -k, --key <key>            Key to use in value estimation, using trained linear regression model.
//!     -f, --modelfile <model>    Path to trained linear regression model
//! ```
use clap::{crate_authors, crate_name, crate_version, value_t};
use clap::{App, Arg};

mod predictor;
use predictor::Predictor;
use std::io::Result;
/// Handles user input
///
/// Helper function that handles user input.
fn ask_key(labels: &[String; 2]) -> f64 {
	use std::io::{stdin, stdout, Write};
	let mut s = String::new();
	let mut val: Option<f64> = Option::None;
	println!(
		"Please insert a key [{}] to estimate [{}]",
		labels[0], labels[1]
	);
	while val.is_none() {
		let _ = stdout().flush();
		stdin()
			.read_line(&mut s)
			.expect("Did not enter a valid string.");
		match s.trim().parse::<f64>() {
			Ok(v) => val = Some(v),
			Err(_) => {
				println!("Could not parse info f64 (float). Try again");
				s.clear()
			}
		}
	}
	val.unwrap()
}
/// Main
fn main() -> Result<()> {
	println!("\n\t## PREDICTOR ##\n");
	let matches = App::new(crate_name!())
		.author(crate_authors!("\n"))
		.version(crate_version!())
		.arg(
			Arg::with_name("key")
				.short("k")
				.long("key")
				.takes_value(true)
				.help("Key to use in value estimation, using trained linear regression model."),
		)
		.arg(
			Arg::with_name("model")
				.short("f")
				.long("modelfile")
				.takes_value(true)
				.help("Path to trained linear regression model")
				.required(true),
		)
		.get_matches();
	let modelfile = matches.value_of("model").unwrap();
	let predictor = Predictor::load(modelfile)?;

	let val: f64 =
		value_t!(matches, "key", f64).unwrap_or_else(|_| ask_key(predictor.get_labels()));
	predictor.predict(val);
	Ok(())
}
