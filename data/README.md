# Data

# cmd
`cat data/diamonds.csv | awk -F "," '{ print $2","$8 }' > data/diamonds_carat_price.csv`