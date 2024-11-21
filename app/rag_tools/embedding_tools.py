import requests
import numpy as np 
from utils.data_processing import pool_embeddings


class TritonEmbedder:
    '''
    Custom Embedder class used to leverage LangChains functionality

    '''

    def __init__(self,  api_url: str): 
        
        self.api_url = api_url
    
    
    # Embeddings Request
    def embed_query(self, text: str) -> list[float]:

        payload = {
            "inputs": [
                {
                    "name": "TEXT",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": [text]
                },
            ]
        }

        response = requests.post(self.api_url, json=payload)

        if response.status_code != 200: 
            raise Exception(f"Error from external API: {response.text}")
        
        # Emebeddings Extraction
        embeddings = response.json().get('outputs')[0]
        embeddings_data =  np.array(embeddings['data'])
        embeddings_shape = embeddings['shape']
        
            
        # Embeddings Pooling
        pooled_embeddings = pool_embeddings(embeddings_data, embeddings_shape)

        return pooled_embeddings
    
    # Embeddings Request for Multiple Chunks (Can be Optimized Further)
    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        
        
        chunk_embeddings = []
        for chunk in chunks:
            pages = chunk['pages']
            content = chunk['chunk']

            chunk_embedding = self.embed_query(content)

            chunk_embeddings.append({
                'pages': pages,
                'embeddings': chunk_embedding,
                'chunk': content
            })

        return chunk_embeddings