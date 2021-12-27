//! Helper module that handles command line arguments
use clap::{crate_authors, crate_name, crate_version};
use clap::{App, Arg, ArgMatches};

pub struct CmdArgs<'a> {
	pub matches: ArgMatches<'a>,
}

impl<'a> CmdArgs<'a> {
	/// Default constructor
	pub fn new() -> Self {
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
		.arg(
			Arg::with_name("seed")
				.short("s")
				.long("seed")
				.takes_value(true)
				.help("Randomness seed for data splitting to train & test sets"),
		)
		.arg(
			Arg::with_name("ratio")
				.short("r")
				.long("ratio")
				.takes_value(true)
				.help("Distribution between test and train set ratio"),
		)
		.arg(
			Arg::with_name("outfile")
				.short("o")
				.long("out")
				.takes_value(true)
				.help("Path to output file (model)"),
		)
		.arg(
			Arg::with_name("alpha")
				.short("a")
				.long("alpha")
				.takes_value(true)
				.help("\u{03B1} - Learning rate ")
		)
		.arg(
			Arg::with_name("stats")
				.long("stats")
				.takes_value(true)
				.help("Path to a directory where plots and statistics should be saved")
		)
		.arg(
			Arg::with_name("iter")
				.short("i")
				.long("iterations")
				.takes_value(true)
				.help("Number of iterations to run, this will overwrite TD limit")
		)
		.arg(
			Arg::with_name("tdlimit")
				.short("t")
				.long("tdlimit")
				.takes_value(true)
				.help("Temporal difference limit (amout of change per iteration). How accurate local minima is.")
		)
		.get_matches();
		Self { matches }
	}

	/// Dataset inut file getter
	///
	/// Example:
	/// ```
	/// let args = CmdArgs::new();
	/// let dataset_path = args.get_infile();
	/// dbg!(dataset_path);
	/// ```
	pub fn get_infile(&self) -> &str {
		self.matches.value_of("datafile").unwrap()
	}
}
