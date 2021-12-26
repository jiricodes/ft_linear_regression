BIN_DIR=bin
TARGETS=target/release/train target/release/predict

all:
	make test
	make build
	make train
	make predict

dev-train:
	cargo run --bin train -- -f data/subject_data.csv

doc:
	cargo doc --no-deps --open

test:
	cargo test

build:
	cargo build --release
	-make fclean
	mkdir -p $(BIN_DIR)
	mv $(TARGETS) $(BIN_DIR)/

train:
	./$(BIN_DIR)/train -f data/subject_data.csv

predict:
	./$(BIN_DIR)/predict -f data/model

fclean:
	rm -r $(BIN_DIR)