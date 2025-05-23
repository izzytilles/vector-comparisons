from pypdf import PdfReader
import re
from collections import Counter
import nltk

def extract_pdf(file_path):
    """
    Extracts text from given PDF and returns in it as a string
    
    Args:
        file_path (str): path to PDF file
    Returns:
        text (str): the text extracted from the PDF
    """
    reader = PdfReader(file_path)
    page = reader.pages[0]
    text = page.extract_text()
    if text is None:
        raise ValueError("No text found in the PDF file.")
    return text


def calculate_density(text):
    """
    Calculates the density of the given text
    Density is defuned as the ratio of unique words to total words
    
    Args:
        text (str): text to analyze
    Returns: 
        (int): the density of the text
    Notes:
        The goal is to be able to quantify how 'meaningful' a chunk of text is
        Adapted from https://github.com/xbeat/Machine-Learning/blob/main/Optimizing%20RAG%20with%20Document%20Chunking%20Techniques%20Using%20Python.md
    """
    words = re.findall(r'\w+', text.lower())
    word_freq = Counter(words)
    unique_words = len(word_freq)
    total_words = len(words)
    return unique_words / total_words

def dynamic_density_chunking(text, min_chunk_size=0, max_chunk_size=7000):
    """
    Splits the given text into chunks based on density so it can be used for RAG

    Args:
        text (str): text to analyze
        min_chunk_size (int): the minimum size (in words) of a chunk
        max_chunk_size (int): the maximum size (in words) of a chunk
    Returns:
        chunks (list): a list of the entire text, split into chunks  
    Note:
        Adapted from https://github.com/xbeat/Machine-Learning/blob/main/Optimizing%20RAG%20with%20Document%20Chunking%20Techniques%20Using%20Python.md
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        density = calculate_density(current_chunk + sentence)
        target_size = min_chunk_size + (1 - density) * (max_chunk_size - min_chunk_size)
        
        if len(current_chunk) + len(sentence) <= target_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks




    

