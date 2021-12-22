all:

prereqs:
	sudo apt-get install -y libfontconfig libfontconfig1-dev

train:
	cargo run --bin train -- -f data/subject_data.csv

doc:
	cargo doc --no-deps --open