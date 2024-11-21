from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
import os
print(os.getcwd())

class MilvusDB: 
    '''
    Custom Milvus DB Class for interactions to DB 
    '''
    def __init__(self, host: str, port: str):
        self.host = host
        self.port = port

        # self.client = MilvusClient(uri=f"{host}:{port}")
        self.client = MilvusClient("milvus_demo.db")

        self.schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), 
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384),  #TO DO: Make this dynamic 
            FieldSchema(name="pages", dtype=DataType.VARCHAR,  max_length=200),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR,  max_length=3000)
        ])

        self.index_params = self.client.prepare_index_params()

        self.index_params.add_index(
            field_name="embeddings", 
            index_type="HNSW",
            metric_type="IP",
            params={}
        )



    def create_collection(self, collection_name: str, metric_type: str):
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
    
        self.client.create_collection(
            collection_name=collection_name,
            schema= self.schema,
            index_params=self.index_params
        )

    def insert_chunks(self, collection_name: str, data: list[float]):

        res = self.client.insert(collection_name=collection_name, data = data)
        return res


    def search_chunks(self, collection_name: str, query_vector: list[float],  limit: int):

        res = self.client.search(
            collection_name=collection_name,  # target collection
            data=[query_vector],  # query vectors
            limit=limit,  # number of returned entities
            search_params={"metric_type": "IP", "params": {}},
            output_fields=['chunk', 'pages']
        )

        return res
