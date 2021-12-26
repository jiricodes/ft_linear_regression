use clap::{crate_authors, crate_name, crate_version, value_t};
use clap::{App, Arg};

fn ask_key() -> f64 {
	use std::io::{stdin,stdout,Write};
	let mut s = String::new();
	let _ = stdout().flush();
	stdin().read_line(&mut s).expect("Did not enter a valid string.");
	let val = s.parse::<f64>().expect("String cannot be parsed to f64");
	val
}

fn main() {
    let matches = App::new(crate_name!())
        .author(crate_authors!("\n"))
        .version(crate_version!())
        .arg(
            Arg::with_name("key")
                .short("k")
                .long("key")
                .takes_value(true)
                .help("Key to use in value estimation, using trained linear regression model.")
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
	let val: f64 = value_t!(matches, "key", f64).unwrap_or(ask_key());
	dbg!(val);
}
