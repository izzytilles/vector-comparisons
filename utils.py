from pypdf import PdfReader

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

def create_chunks(text, max_length=7000, overlap_pct=0.1):
    """
    Splits given text into chunks of max length or less
    For use with embedders
    
    Args: 
        text (str): the text to be split
        max_length (int): the maximum length allowed by the embedder - defaults to 7000 for OpenAI
        overlap_pct (float): percentage of overlap between chunks - defaults to 10%
    Returns:
        chunks (list): a list containing all of the contents of the given text, in sizes approved by the embedder
    Notes:
        The maximum length is set to 7000 words, but for more precise results this should be MUCH less, closer to 200-600
        """
    words = text.split()
    chunks = []
    curr_chunk = []
    curr_length = 0

    for word in words:
        word_length = len(word) + 1
        if curr_length + word_length > max_length:
            chunks.append(' '.join(curr_chunk))
            curr_chunk = []
            curr_length = 0
        curr_chunk.append(word)
        curr_length += word_length

    if curr_chunk:
        chunks.append(' '.join(curr_chunk))

    return chunks


    

