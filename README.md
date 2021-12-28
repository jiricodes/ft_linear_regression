# ft_linear_regression
This project is the first step into AI and Machine Learning under 42 curriculum at Hive Helsinki. Basic machine learning algorithm and program that predicts the price of a car by using a linear regression function with a gradient descent algorithm.

![](resources/example.png | width=1024)

## Requirements
Only installed rustup is required. It can be easily done with `make rust` from the root of this repo.

## Usage
The basic usage is to run `train` binary on your data. Then running `predict` binary supplying previously trained model.

### Simple
This is a showcase run, that uses provided dataset [`data/subject_data.csv`](data/subject_data.csv). The train program uses 4/5 of the dataset to train the model and remaining 1/5 to test its precision.

Simple `make` command will run following steps
- tests
- build
- train
- predict

You'll be promped to input a mileage that is then used to estimate a car's value using the pretrained model.

### Building
```
make build
```
Two bineries are created and located in `bin/` directory.

### Training
```
USAGE:
    train [OPTIONS] --file <datafile>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -a, --alpha <alpha>        α - Learning rate 
    -f, --file <datafile>      Input data file
    -i, --iterations <iter>    Number of iterations to run, this will overwrite TD limit
    -o, --out <outfile>        Path to output file (model)
    -r, --ratio <ratio>        Distribution between test and train set ratio
    -s, --seed <seed>          Randomness seed for data splitting to train & test sets
        --stats <stats>        Path to a directory where plots and statistics should be saved
    -t, --tdlimit <tdlimit>    Temporal difference limit (amout of change per iteration). How accurate local minima is.
```

### Predicting
```
USAGE:
    predict [OPTIONS] --modelfile <model>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -k, --key <key>            Key to use in value estimation, using trained linear regression model.
    -f, --modelfile <model>    Path to trained linear regression model
```

### Testing
```
make test
```
Runs `cargo test`

### Documentation
```
make doc
```
Compiles and opens documentation


## To-Do
- Error management - currently panics