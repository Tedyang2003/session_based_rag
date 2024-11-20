import numpy as np 

def pool_embeddings(token_embeddings, shape):
           
    input_tensor =  np.array(token_embeddings)
    batch_size, seq_len, embedding_dim = shape

    # Reshape the 1D array into the original 3D shape 
    token_embeddings = token_embeddings.reshape(batch_size, seq_len, embedding_dim)

    # Apply mean pooling across the sequence length (axis=1)
    pooled_embeddings = np.mean(token_embeddings, axis=1) 
    
    return pooled_embeddings.tolist()[0]


def chunk_with_metadata(
    pages: list[dict[str, str]], chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Chunk text from multiple pages with metadata tracking.

    pages: A list of dictionaries containing page content and metadata.
    chunk_size: The size of each chunk (number of tokens/words).
    overlap: The number of tokens to overlap between chunks. (For better coherence)
    
    :return: A list of chunks with metadata.
    """
    chunks = []
    current_chunk = []
    current_pages = []

    #Iterates through all pages in the book and splits them to token length

    for page in pages:
        words = page["content"].split()
        page_number = page["page"]
            
        while words:
            # Calculate remaining space available for current chunk
            remaining_space = chunk_size - len(current_chunk)

            # Fill the chunk for any remaining space
            if remaining_space > 0:
                current_chunk += words[:remaining_space]

                words = words[remaining_space:]
                current_pages.append(page_number)

            else:
                # Save the current chunk with metadata
                chunks.append({
                    "chunk": " ".join(current_chunk),
                    "pages": list(set(current_pages))  # Remove duplicate pages
                })
                # Slide the overlap into the next chunk
                current_chunk = current_chunk[-overlap:]
                current_pages = current_pages[-1:]

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append({
            "chunk": " ".join(current_chunk),
            "pages": list(set(current_pages)),
        })

    return chunks
