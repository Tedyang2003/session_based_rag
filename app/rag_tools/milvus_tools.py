from pymilvus import MilvusClient


class MilvusDB: 
    '''
    Custom Milvus DB Class for interactions to DB 
    '''
    def __init__(self, host: str, port: str):
        self.host = host
        self.port = port

        # self.client = MilvusClient(uri=f"{host}:{port}")
        self.client = MilvusClient("milvus_demo.db")


    def create_collection(self, collection_name: str, vector_dimension: int, metric_type: str):
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
    
        self.client.create_collection(
            collection_name=collection_name,
            dimension=vector_dimension,
            metric_type=metric_type
        )

    def insert_chunks(self, collection_name: str, data: list[float]):

        res = self.client.insert(collection_name=collection_name, data = data)
        return res


    def search_chunks(self, collection_name: str, query_vector: list[float],  limit: int, output_fields: list):

        res = client.search(
            collection_name=collection_name,  # target collection
            data=query_vector,  # query vectors
            limit=limit,  # number of returned entities
            output_fields=output_fields,  # specifies fields to be returned
        )

        return res
