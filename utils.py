from pypdf import PdfReader
import re
from collections import Counter
import nltk
import openai
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm
from env import AZURE_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, POSTGRES_SSL, POSTGRES_HOST, POSTGRES_USERNAME, POSTGRES_PASSWORD, POSTGRES_DATABASE

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

def connect_to_openai():
    """
    Connects to OpenAI API using Azure OpenAI service
    """
    openai.api_type = "azure"
    openai.api_key = AZURE_OPENAI_KEY
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = "2023-05-15"

def connect_to_db():
    """
    Connects to a PostgreSQL database

    Returns:
        conn (psycopg connection): conenction to the db
        cursor (psycopg cursor): cursor to the db
    Note:
        Adapted from https://wiki.postgresql.org/wiki/Psycopg2_Tutorial
    """
    conn = psycopg2.connect("dbname='POSTGRES_DATABASE' user='POSTGRES_USERNAME' host='POSTGRES_HOST' password='POSTGRE_PASSWORD'")
    cursor = conn.cursor()
    return conn, cursor

def create_embedding(text):
    """
    From the given text, produces an embedding using OpenAI's API
    Args:
        text (str): text to embed
    Returns:
        embedding (list of floats): the embedding of the text
    """
    response = openai.Embedding.create(
        input=text,
        engine=AZURE_DEPLOYMENT_NAME
    )
    return response['data'][0]['embedding']

def embedding_and_inserting_flow(text):
    """
    General flow for chunking, embedding, and inserting text into the database

    Args:
        text (str): text to embed and insert into the database
    """
    conn, cursor = connect_to_db()
    cursor.execute(
        """CREATE TABLE text_chunks (
            id PRIMARY KEY,
            content TEXT,
            embedding VECTOR(1536)
        );
        """
    )
    conn.commit()
    records = []
    chunks = dynamic_density_chunking(text)
    for chunk in chunks:
        embedding = create_embedding(chunk)
        records.append((chunk, embedding))

    execute_values(
        cursor,
        "INSERT INTO article_chunks (content, embedding) VALUES %s",
        records
    )
    conn.commit()
    cursor.close()
    conn.close()




    

