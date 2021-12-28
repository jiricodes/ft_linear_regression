//! # Linear regression model trainer
//! This binary takes a dataset and creates a model for estimating values from it.
//! ## Example
//!
//! <a href="../../../resources/example.png">
//!     <img src="../../../resources/example.png"></img>
//! </a>
//!
//! ## Usage
//! ```text
//! USAGE:
//! train [OPTIONS] --file <datafile>
//!
//! FLAGS:
//!     -h, --help       Prints help information
//!     -V, --version    Prints version information
//!
//! OPTIONS:
//!     -a, --alpha <alpha>        α - Learning rate
//!     -f, --file <datafile>      Input data file
//!     -i, --iterations <iter>    Number of iterations to run, this will overwrite TD limit
//!     -o, --out <outfile>        Path to output file (model)
//!     -r, --ratio <ratio>        Distribution between test and train set ratio
//!     -s, --seed <seed>          Randomness seed for data splitting to train & test sets
//!         --stats <stats>        Path to a directory where plots and statistics should be saved
//!     -t, --tdlimit <tdlimit>    Temporal difference limit (amout of change per iteration). How accurate local minima is.
//!
//! ```
mod trainer;
use trainer::{Trainer, TrainerContext};
mod arguments;
use arguments::CmdArgs;
mod result;
use result::{Result, TrainError};

/// Main
fn main() -> Result<()> {
	println!("\n\t## TRAINER ##\n");
	let cmdargs = CmdArgs::new();
	let ctx = TrainerContext::from(&cmdargs);
	let filename = cmdargs.get_infile();
	println!("Input data location {}", filename);
	let mut trainer = Trainer::load(filename, Some(ctx));
	trainer.train();
	trainer.test_accuracy();
	trainer.save_output(Option::None)?;
	match trainer.plot_result() {
		Ok(_) => {}
		Err(e) => return Err(TrainError::Custom(format!("Plotter Error: {:?}", e))),
	}
	Ok(())
}
