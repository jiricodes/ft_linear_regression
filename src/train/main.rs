mod trainer;
use trainer::{Trainer, TrainerContext};
mod arguments;
use arguments::CmdArgs;

fn main() {
	let cmdargs = CmdArgs::new();
	let ctx = TrainerContext::from(&cmdargs);
	let filename = cmdargs.get_infile();
	println!("Input data location {}", filename);
	let mut trainer = Trainer::load(filename, Some(ctx));
	dbg!(&trainer);
	trainer.train();
	trainer.test_accuracy();
	match trainer.save_output(Option::None) {
		Ok(_) => print!("Model saved"),
		Err(e) => print!("IO Error: {}", e),
	}
}
