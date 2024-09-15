from vector_store import VectorStore
from process_text import *
import numpy as np

# define functions

def tokenization(sentences: list) -> set:
    '''
    returns:
    param:
    '''
    vocab = set()
    for sentence in sentences:
        tokens = sentence.lower().split()
        vocab.update(tokens)
    return vocab

def index_tokens(vocab: set) -> dict:
    '''
    returns:
    param:
    '''
    token_index_dict = {word: i for i, word in enumerate(vocab)}
    return(token_index_dict)

def vectorization(sentences: list, vocab: set, token_index: dict) -> dict:
    '''
    returns:
    params:
    params:
    params:
    Note this BoW method does not account for order of tokens, omly frequency
    '''
    sentence_vectors = {}
    for sentence in sentences:
        tokens = sentence.lower().split()
        vector = np.zeros(len(vocab))
        for token in tokens:
            vector[token_index[token]] +=1 # for term freq in BoW
        sentence_vectors[sentence] = vector
    return sentence_vectors

def similarity_search(query_sentence: str, vocab: set, token_index: dict, num_results: int) -> list:
    '''
    return:
    param:
    param:
    param:
    '''
    query_vector = np.zeros(len(vocab))
    query_tokens = query_sentence.lower().split()
    for token in query_tokens:
        if token in token_index:
            query_vector[token_index[token]] +=1
    similar_sentences = vector_store.find_similar_vectors(query_vector, num_results)
    return similar_sentences

# begin processing 
print("initializing application, this may take a minute...")

# instantiate VectorStore
vector_store = VectorStore()

# get and pre-process data 
raw_text = extract_text_within_bounds('vulgar_toungue.txt', 
                                      start_line = 'ABBESS, or LADY ABBESS, A bawd, the mistress of a',
                                      end_line = '*** END OF THE PROJECT GUTENBERG EBOOK 1811 DICTIONARY OF THE VULGAR TONGUE ***'
                                      )

# remove everything but letters and numbers
pre_processed_text = remove_non_alphanumeric(raw_text)

# split up into sentence level
sentences = parse_text_into_sentences(pre_processed_text)

#tokenization 
vocab = tokenization(sentences)

# add indicies to each unique word in corpus
indexed_tokens = index_tokens(vocab)

# turn vocabulary in vectors 
sentence_vectors = vectorization(sentences, vocab, indexed_tokens)

# adding each sentence vector to the vectorStore
for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)

# Query Phase (can be repeated multiple times)

def query_loop():
    while True:
        search_pattern = input("Enter a string to query (or type 'exit' to quit): ")
        if search_pattern.lower() == 'exit':
            break
        results = similarity_search(
            query_sentence=search_pattern, 
            vocab=vocab, 
            token_index=indexed_tokens, 
            num_results=3
        )
        print("Results:", results)

# Run the query loop
query_loop()