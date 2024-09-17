#!/bin/bash

# BPE Token Segmenter using saved merge rules from the learner

# Load the merge rules from the merge file
MERGE_FILE="merge_tokens.txt"

if [ ! -f "$MERGE_FILE" ]; then
    echo "Merge tokens file not found. Please run the BPE learner script first."
    exit 1
fi

# Read the merged tokens into an array
MERGES=($(cat "$MERGE_FILE"))

# Input text to tokenize (replace this with your input text or file)
TEXT="the quick brown fox jumps over the lazy dog."

# Preprocess text (breaking into characters and appending underscores at the end of each word)
# The preprocessing must match what was done in the learner script.
# For example, break words into characters and append underscores at the end of each word.
TOKENS=$(echo "$TEXT" | tr -sc "A-Za-z" "\n" | sed -e 's/\b\([A-Za-z]\+\)\b/\1_/g' -e 's/./& /g' -e 's/_ /_/g')

# Convert the tokens into an array
TOKENS=($TOKENS)

# Function to apply BPE merges
apply_merges() {
    local merge_token="$1"
    local new_tokens=()
    
    i=0
    while [ $i -lt ${#TOKENS[@]} ]; do
        # Try to merge the current token and the next one
        if [ $i -lt $((${#TOKENS[@]}-1)) ] && [ "${TOKENS[$i]}${TOKENS[$i+1]}" = "$merge_token" ]; then
            new_tokens+=("$merge_token")
            i=$((i+2)) # Skip the next token since it's merged
        else
            new_tokens+=("${TOKENS[$i]}")
            i=$((i+1))
        fi
    done
    
    # Update the TOKENS array with the new merged tokens
    TOKENS=("${new_tokens[@]}")
}

# outter loop through each merge token
for merge in "${MERGES[@]}"; do
    apply_merges "$merge"
done

# Output the tokenized text
echo "Tokenized text: ${TOKENS[@]}"