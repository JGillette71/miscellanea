# naive pattern matching algorithm

def naive_pattern_match(search_term: str, corpus: str):
    """Naive pattern matching of substring."""

    if not search_term or not corpus:
        print("Empty search term or corpus")
        return None
    
    search_term_pointer = 0

    for pointer in range(len(corpus)):
        #print(f"Comparing corpus[{pointer}]='{corpus[pointer]}' with search_term[{search_term_pointer}]='{search_term[search_term_pointer]}'")
        
        if corpus[pointer] == search_term[search_term_pointer]:
            if search_term_pointer == len(search_term) - 1:
                found_at = pointer - search_term_pointer
                #print(f"Match found at indices: ({found_at}, {pointer})")
                return (found_at, pointer)
            else:
                search_term_pointer += 1
        else:
            search_term_pointer = 0  # Reset on mismatch

    print("No match found")
    return None  # Return None if no match is found

# TODO improved negative version 
def naive_search_alpha(search_term: str, corpus: str):
    """Naive pattern matching of all substring."""
    
    if not search_term or not corpus:
        print("Empty search term or corpus")
        return None

    results = []
    
    for corpus_pointer in range(len(corpus)):
        for search_term_pointer in range(len(search_term)):
            if search_term[search_term_pointer] != corpus[corpus_pointer]:
                continue
            elif search_term_pointer == len(search_term) - 1:
                results.append((corpus_pointer - (len(search_term) - 1), corpus_pointer))
    return results

# demo quadratic increase in execution time as corpus grows???

search_term = "caked"
corpus_a = "extracakeycakegotcakedonthecake"
corpus_b = "amaextracaextraxextracakedxtraxaextracakeycakegotcakedonthecak"

# Now use timeit separately for timing once you're sure the function works
import timeit

def time_udf(search_term: str, corpus: str):
    timer = timeit.Timer(lambda: naive_pattern_match(search_term, corpus))
    full_time = timer.timeit(number=1000)  # Run 1000 times
    return full_time

def time_udf_a(search_term: str, corpus: str):
    timer = timeit.Timer(lambda: naive_search_alpha(search_term, corpus))
    full_time = timer.timeit(number=1000)  # Run 1000 times
    return full_time

# print len diff
print(f'len of corpus_a {len(corpus_a)}')
print(f'len of corpus_b {len(corpus_b)}')

# Measure execution time separately
time_a = time_udf(search_term, corpus_a)
print(f"Average execution time for corpus_a: {time_a:.6f} seconds")

# Measure execution time separately
time_b = time_udf(search_term, corpus_b)
print(f"Average execution time for corpus_b: {time_b:.6f} seconds")

# Measure execution time separately
time_c = time_udf_a(search_term, corpus_a)
print(f"Average execution time for corpus_a: {time_c:.6f} seconds")

# Measure execution time separately
time_d = time_udf_a(search_term, corpus_b)
print(f"Average execution time for corpus_b: {time_d:.6f} seconds")