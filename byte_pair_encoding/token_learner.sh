#!/bin/bash

# WARNING: This inefficient BPE Algorithm is for learning purposes; set K < ~100

if (( ${BASH_VERSION%%.*} < 4 )); then
    echo "This script requires Bash version 4.0 or higher."
    exit 1
fi

# func byte_pair_encode(strings C, number of merges k) --> vocab V
# all uniq charactres in C --> starting vocabulary V
# for i = 1 to k do 
#    token_L, token_R most frequent pair of TOKENS in C
#    token_new = token_L, token_R
#    V append token_new 
#    C replace token_L, token_R with token_new
# retuen V

# Initialize files for vocabulary and merges
DATA_FILE="data.txt"
VOCAB_FILE="vocabulary.txt"
MERGE_FILE="merge_tokens.txt"

# Clear the files if already exists
> $VOCAB_FILE
> $MERGE_FILE

# get data to tokenize
if [ -f $DATA_FILE ]; then
    echo "data file found"
else
    echo "downloading default data source..."
    wget -O $DATA_FILE "https://www.gutenberg.org/ebooks/10681.txt"
fi

# transliterate any non-alphabetic into \n for line-by-line words
# sed substitute an underscore at the end of each word 
# sed substitute a space in between each character
# TODO bash substitution for better performance
CORPUS=$(tr -sc "A-Za-z" "\n" < $DATA_FILE | sed -e 's/\b\([A-Za-z]\+\)\b/\1_/g' -e 's/./& /g' -e 's/_ /_/g')

# print first 10 lines
echo "sample of preprocessed corpus: $CORPUS" | head -n 10

# cast corpus to array of TOKENS
TOKENS=($CORPUS)

# Initialize unique characters as the starting vocabulary
echo "$CORPUS" | grep -o . | sort | uniq >> "$VOCAB_FILE"

# declare target vocab size
k=100

# iterate through merges
for ((merge=1; merge<=k; merge++)); do
    
    # step 1: extract pairs and count frequencies
    # declare associative array 
    declare -A PAIR_COUNTS=()
    for ((i=0; i<${#TOKENS[@]}-1; i++)); do
        pair="${TOKENS[i]} ${TOKENS[i+1]}"
        PAIR_COUNTS["$pair"]=$((PAIR_COUNTS["$pair"] + 1))
    done

    # step 2 most frequent pair
    max_count=0
    MOST_FREQ_PAIR=""
    for pair in "${!PAIR_COUNTS[@]}"; do
        count=${PAIR_COUNTS["$pair"]}
        if (( count > max_count)); then
            max_count=$count
            MOST_FREQ_PAIR=$pair
        fi
    done

    # break if no more pairs
    if [ -z "$MOST_FREQ_PAIR" ]; then
        echo "no more pairs to merge"
        break
    fi

    # step 3 merge most frequent pair
    NEW_TOKEN=$(echo "$MOST_FREQ_PAIR" | tr -d " ")
    
    # Write the new token to both files
    echo "$NEW_TOKEN" >> "$VOCAB_FILE"
    echo "$NEW_TOKEN" >> "$MERGE_FILE"

    # step 4 update TOKENS array from corpus
    new_tokens=()
    i=0
    while [ $i -lt ${#TOKENS[@]} ]; do
        if [ $i -lt $((${#TOKENS[@]}-1)) ] && [ "${TOKENS[$i]} ${TOKENS[$i+1]}" = "$MOST_FREQ_PAIR" ]; then
            new_tokens+=("$NEW_TOKEN")
            i=$((i+2))
        else
            new_tokens+=("${TOKENS[$i]}")
            i=$((i+1))
        fi
    done
    TOKENS=("${new_tokens[@]}")
done

echo "Final vocabulary written to $VOCAB_FILE"
echo "Merged tokens written to $MERGE_FILE"