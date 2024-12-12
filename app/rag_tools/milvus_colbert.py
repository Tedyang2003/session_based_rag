from pdf2image import convert_from_path
from pymilvus import DataType
import concurrent.futures
import numpy as np
import os
from flask import jsonify
import uuid

# Milvus Retriever specialised for Colpali data storage and retrieval
class MilvusColbertRetriever: 
    '''
    Custom Milvus DB Class for interactions to DB 

    '''
    def __init__(self, milvus_client, collection_name, dim=128):
    
        self.collection_name = collection_name
        self.client = milvus_client

        # If collection exists, load collection
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)

        self.dim = dim

    
    # Convert Pages to Images for Colpali Model based on file type
    def rasterize(self, path):
        
        file_directory, _, file_extention = self.get_file_parts(path)

        if file_extention == '.pdf':
            images = convert_from_path(pdf_path=path)
            
            page_directory = f'{file_directory}/pages'

            if not os.path.exists(page_directory):
                os.makedirs(page_directory)

            for i, image in enumerate(images):
                image.save(f"{page_directory}/page_{i + 1}.png", "PNG")

        else: 
            return jsonify({'error': 'This file type is not supported'}), 400

    
    # Get file parts from path
    def get_file_parts(self, file_path):
        """
        Get the directory, file name without extension, and file extension separately.
        Convert the extension to lowercase for case-insensitive comparison.
        """
        directory = os.path.dirname(file_path)
        file_name_with_ext = os.path.basename(file_path)
        file_name, ext = os.path.splitext(file_name_with_ext)
        unique_id = self.generate_random_id()

        return directory, file_name+unique_id, ext.lower()


    # Create a new collection in Milvus for storing embeddings using fixed schema
    def create_collection(self):

        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

        # DB Schema
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema.add_field(field_name="seq_id", datatype=DataType.INT16)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="page_id", datatype=DataType.INT64)
        schema.add_field(field_name="page", datatype=DataType.VARCHAR, max_length=65535)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )


    # Creates index to allow ranking and searching of vectors
    def create_index(self):
        
        # Release collection before dropping to ensure consistency
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="vector"
        )

        # Prepare the index to be used, metric and which field name
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="HNSW",
            metric_type="IP",  # Change if needed
            params={
                "M": 16,
                "efConstruction": 500,
            },  # adjust these parameters as needed
        )

        # Create the index
        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )


    # Create a scalar index for the "page_id" field to enable fast lookups by page ID.
    def create_scalar_index(self):
        self.client.release_collection(collection_name=self.collection_name)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="page_id",
            index_name="int32_index",
            index_type="INVERTED",  # or any other index type you want
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )


    #Insert new vector data into the collection for retrieval during search
    def insert(self, data):

        # Insert ColBERT embeddings and metadata for a page into the collection.
        colbert_vecs = [vec for vec in data["colbert_vecs"]]
        seq_length = len(colbert_vecs)
        page_ids = [data["page_id"] for i in range(seq_length)]
        seq_ids = list(range(seq_length))
        pages = [""] * seq_length
        pages[0] = data["filepath"]
        doc_ids = [data["doc_id"] for i in range(seq_length)]

        # Insert the data as multiple vectors (one for each sequence) along with the corresponding metadata.
        list_of_vectors = []

        for i in range(seq_length):
            list_of_vectors.append(
                {
                    "vector": colbert_vecs[i],
                    "seq_id": seq_ids[i],
                    "page_id": page_ids[i],
                    "page": pages[i],
                    "doc_id": doc_ids[i]
                }
            )

        self.client.insert(self.collection_name, list_of_vectors)
        
        return "success"



    # Vector search for top-k most similar searches releated to data vector
    def search(self, data, topk):

        search_params = {
            "metric_type": "IP", 
            "params": {}
        }

        # Returns top 50 most relevant sequences using dot product metric
        results = self.client.search(
            self.collection_name,
            data,
            limit=int(50),
            output_fields=["vector", "seq_id", "page_id", 'doc_id'],
            search_params=search_params,
        )

        # Get the relevant page ids
        page_ids = set()

        for r_id in range(len(results)):
            for r in range(len(results[r_id])):
                page_ids.add((results[r_id][r]["entity"]["page_id"], results[r_id][r]["entity"]["doc_id"]))

        scores = []

        # For each page, it will extract all sequences and then stack them and get an overall score for a page  
        def rerank_single_page(page_id, doc_id, data, collection_name):
            
            page_colbert_vecs = self.retrieve_by_page_id(collection_name=collection_name, page_id=page_id, doc_id=doc_id)

            page_vecs = np.vstack(
                [page_colbert_vecs[i]["vector"] for i in range(len(page_colbert_vecs))]
            )

            score = self.compare_vector_chunks(data, page_vecs, mode='dot')
            relevant_page = page_colbert_vecs[0]['page']

            return (score, page_id, doc_id, relevant_page)

        # Multi threading page reranking to identify scores for each page 
        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:

            futures = {}

            for page in page_ids:
                index = executor.submit(
                    rerank_single_page, page[0], page[1], data, self.collection_name
                )

                futures[index] = page

            for future in concurrent.futures.as_completed(futures):
                score, page_id, doc_id, relevant_page = future.result()
                scores.append((score, page_id, doc_id, relevant_page))
        
        # Get Highest scores
        scores.sort(key = lambda x: x[0], reverse=True)

        # Return most relevant pages 
        if len(scores) >=topk:
            return scores[:topk]
        else: 
            return scores


    # Verify the relevance between 2 pages
    def is_page_relevant(self, collection_name, page_id1, page_id2, doc_id, thresh=1):

        
        page_1 = self.retrieve_by_page_id(collection_name=collection_name, page_id=page_id1, doc_id=doc_id)
        page_2 = self.retrieve_by_page_id(collection_name=collection_name, page_id=page_id2, doc_id=doc_id)

        if not page_1 or not page_2:
            return False 

        page_1_matrix = np.vstack(
            [page_1[i]["vector"] for i in range(len(page_1))]
        )

        page_2_matrix = np.vstack(
            [page_2[i]["vector"] for i in range(len(page_2))]
        )

        relevance = self.compare_vector_chunks(page_1_matrix, page_2_matrix, mode='cosine')

        if relevance > thresh:
            return (1, page_id2, doc_id, page_2[0]['page'])
    
        return False

    # Calculate the cosine similarity between 2 vector stacks (matrices) 
    def compare_vector_chunks(self, matrix_1, matrix_2, mode='dot'): 
        if mode == 'cosine': 
            # Normalize the vectors 
            data_normalized = self.normalize(matrix_1) 
            page_vecs_normalized = self.normalize(matrix_2) 
            
            # Calculate cosine similarity 
            cosine_similarities = np.dot(data_normalized, page_vecs_normalized.T) 
            
            # Using max similarity and summing 
            max_cosine_similarity_scores = cosine_similarities.max(axis=1).sum() 
            
            return max_cosine_similarity_scores 
        
        elif mode == 'dot': 
            # Calculate dot product 
            dot_product_scores = np.dot(matrix_1, matrix_2.T).max(axis=1).sum() 
            
            return dot_product_scores 
        
        else: 
            
            raise ValueError("Mode should be either 'cosine' or 'dot'")
        
    def normalize(self, vectors):
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors/norm
    
    
    def retrieve_by_page_id(self, collection_name, page_id, doc_id):
        page_colbert_vecs = self.client.query(
            collection_name=collection_name,
            filter=f"page_id == {page_id} && doc_id == '{doc_id}'",
            output_fields=["seq_id", "vector", "page"],
            limit=1000,
        )

        #Check if there even is a subsequent page
        if len(page_colbert_vecs[0]) == 0:
            return []

        return page_colbert_vecs
    
    @staticmethod
    def generate_random_id(): 
        return str(uuid.uuid4())