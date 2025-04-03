def chunk_text(text, max_chunk_size=512, overlap=50):
    # Split text into overlapping chunks for processing
    words = text.split()
    chunks = []
    
    if len(words) <= max_chunk_size:
        return [text]
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i + max_chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        i += max_chunk_size - overlap
    
    return chunks