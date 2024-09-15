import re

def extract_text_within_bounds(filename, start_line, end_line):
    '''
    retruns text content within a start and end ending index
    param: filename: string txt file name
    param: start_line: line of text to begin extract
    param: end_line: line of text to stop extract
    ### ADD default and handling of multiple finds for start or end index
    '''
    with open(filename, 'r') as file:
        content = file.read()
    
    # Split the content at the start and end lines
    start_index = content.find(start_line)
    end_index = content.find(end_line)
    
    if start_index == -1 or end_index == -1:
        raise ValueError("Start or end line not found in the file.")
    
    # Extract the relevant section
    relevant_content = content[start_index:end_index + len(end_line)]
    
    return relevant_content

def parse_text_into_sentences(text):
    # Split text into sentences based on double newlines
    sentences = [sentence.strip() for sentence in text.split('\n\n')]
    return sentences

def remove_non_alphanumeric(text):
    # Use regular expression to remove all non-alphanumeric characters
    import re
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return clean_text