# Musings in Natural Language Processing

Loose collection of different exercises like word embedding, large language models, and retrieval augmented generation tools.

## Vector Store 

A simple *bag of words* model with a similarity search function. Testing done for the laughs with a 1811 publication called *Dictionary of the Vulgar Tongue* by author Francis Grose. The data was sources from [Project Gutenburg](https://www.gutenberg.org/ebooks/5402).

## Apache Tika Demo

Initial attempts at running Apache Tika in a Python Notebook environment to evaluate suitability for PDF text extraction.

## UNIX Tokenizer

Simple naive UNIX tokenizer that processes unstructured corpus into a vocabulary.

## Byte-Pair Encoding 

Bash scripts called `token_learner.sh` to train the tokenizer on a corpus and `tokenizer.sh` to tokenize original text. Bash scripting is used for practice, not performance. This token learner is very slow. I do not recommend `k>100`. 