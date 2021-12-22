all:

prereqs:
	sudo apt-get install -y libfontconfig libfontconfig1-dev

doc:
	cargo doc --no-deps --open