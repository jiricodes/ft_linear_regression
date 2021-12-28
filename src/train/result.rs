//! Utility mod for error handling
use std::fmt;
use std::io;

pub type Result<T> = std::result::Result<T, TrainError>;

#[derive(Debug)]
pub enum TrainError {
	Io(io::Error),
	Custom(String),
}

impl fmt::Display for TrainError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match *self {
			TrainError::Io(ref err) => err.fmt(f),
			TrainError::Custom(ref err) => write!(f, "Custom Error: {:?}", err),
		}
	}
}

impl From<io::Error> for TrainError {
	fn from(f: io::Error) -> Self {
		Self::Io(f)
	}
}
