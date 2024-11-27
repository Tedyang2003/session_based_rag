from pymilvus import MilvusClient, DataType

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

    
    # Convert PDF Pages to Images for Colpali Model
    def rasterize(self, pdf_path):
        
        images = convert_from_path(pdf_path=pdf_path)
        
        for i, image in enumerate(images):
            image.save(f"{self.collection_name}/pages/page_{i + 1}.png", "PNG")



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
        schema.add_field(field_name="doc_id", datatype=DataType.INT64)
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )


    # Creates an index to allow ranking and searching of vectors
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



    def create_scalar_index(self):
        # Create a scalar index for the "doc_id" field to enable fast lookups by document ID.
        self.client.release_collection(collection_name=self.collection_name)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="doc_id",
            index_name="int32_index",
            index_type="INVERTED",  # or any other index type you want
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )


    #Insert new vector data into the collection
    def insert(self, data):

        # Insert ColBERT embeddings and metadata for a document into the collection.
        colbert_vecs = [vec for vec in data["colbert_vecs"]]
        seq_length = len(colbert_vecs)
        doc_ids = [data["doc_id"] for i in range(seq_length)]
        seq_ids = list(range(seq_length))
        docs = [""] * seq_length
        docs[0] = data["filepath"]

        # Insert the data as multiple vectors (one for each sequence) along with the corresponding metadata.
        self.client.insert(
            self.collection_name,
            [
                {
                    "vector": colbert_vecs[i],
                    "seq_id": seq_ids[i],
                    "doc_id": doc_ids[i],
                    "doc": docs[i],
                }
                for i in range(seq_length)
            ],
        )