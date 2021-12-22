use clap::{crate_authors, crate_name, crate_version};
use clap::{App, Arg};

mod trainer;
use trainer::Trainer;
mod keyf64;

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
    let mut trainer = Trainer::load(filename, Option::None);
    dbg!(&trainer);
    trainer.train();
}
