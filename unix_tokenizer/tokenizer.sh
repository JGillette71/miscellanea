#!/bin/bash

# Citation:
# Jurafsky, D., & Martin, J. H. (2024). Speech and Language Processing: An Introduction 
# to Natural Language Processing, Computational Linguistics, and Speech Recognition 
# with Language Models (3rd ed.). Online manuscript released August 20, 2024. 
# https://web.stanford.edu/~jurafsky/slp3

# INSTRUCTIONS
# tr change every sequence of nonalphabetic characters to newline
# -c option negates regex pattern
# -s option squeezes consecutive changes into one change
# sort -n -r organize by count highest-to-lowest
# uniq -c to collapse duplicate tokens and count

if [ -f "data.txt" ]; then
    echo "data file found"
else
    echo "downloading default data source..."
    wget -O data.txt "https://www.gutenberg.org/ebooks/10681.txt.utf-8"
    DATA_FILE="data.txt"
fi

tr -sc "A-Za-z" "\n" < $DATA_FILE | tr "A-Z" "a-z" | sort | uniq -c | sort -n -r > output.txt

rm data.txt

echo "Tokenizer complete! See output.txt"