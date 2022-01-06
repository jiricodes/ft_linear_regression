BIN_DIR=bin
STATS_DIR=stats
TARGETS=target/release/train target/release/predict

DATASET=data/subject_data.csv

MAKEOPTIONS=--no-print-directory

all:
	@echo "Running tests"
	@make $(MAKEOPTIONS) test
	@echo "Building"
	@make $(MAKEOPTIONS) build
	@echo "Training model for $(DATASET)"
	@make $(MAKEOPTIONS) train
	@make $(MAKEOPTIONS) predict
	@open stats/result.png

install:
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
	sudo apt-get install cmake libfreetype6-dev libfontconfig1-dev xclip

unistall:
	sudo apt-get purge cmake libfreetype6-dev libfontconfig1-dev xclip
	rustup self uninstall

dev-train:
	cargo run --bin train -- -f data/subject_data.csv

doc:
	cargo doc --no-deps --open

test:
	@cargo test

build:
	@cargo build --release
	@-make $(MAKEOPTIONS) fclean
	@mkdir -p $(BIN_DIR)
	@mv $(TARGETS) $(BIN_DIR)/

train:
	@mkdir -p $(STATS_DIR)
	./$(BIN_DIR)/train -f $(DATASET)

predict:
	./$(BIN_DIR)/predict -f data/weights

fclean:
	@rm -r $(BIN_DIR)