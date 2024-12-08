from flask import Blueprint, request, jsonify
from rag_tools.colpali_handler import ColpaliHandler
from rag_tools.milvus_colbert import MilvusColbertRetriever
from rag_tools.vlm_tools import VlmHandler
from pymilvus import MilvusClient
from PIL import Image
import os
import logging
import base64


# Initialize Logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),  # Log to a file
                        logging.StreamHandler()          # Log to the console
                    ])
logger = logging.getLogger(__name__)

logger.info("Application Initialization Started")
logger.info(f"Using {os.getcwd()}")


# ENV variables
vlm_url = "http://ollama-route-ollama-daryl.apps.nebula.sl/api/chat"
device_name = "auto"
model_path = "vidore/colpali-v1.2"

document_storage_prefix = "/workspaces/rag_prototype/storage"
milvus_uri = f"{document_storage_prefix}/milvus.db"

topk= 1
supported_extensions = {'.pdf'}


# Ensure the directory exists and save the PDF file to the specified path for document use later
if not os.path.exists(document_storage_prefix):
    os.makedirs(document_storage_prefix)

# Intialize Components
logger.info("Initializing Colpali Engine and Downloading Model")
colpali = ColpaliHandler(device_name=device_name, model_path=model_path)

logger.info("Initializing Milvus Retriever")
client = MilvusClient(milvus_uri)
vlm =  VlmHandler(api_url=vlm_url)

logger.info("Initializing RAG Routes")
rag_bp = Blueprint('rag', __name__)

logger.info("Server is running...")



# Single Query API to handle user questions
@rag_bp.route('/query', methods=['POST'])
def query():
    
    # Retrive request collection name and query 
    request_json = request.get_json()
    collection_name = request_json['collection_name']
    query = request_json['query']
    logger.info(f"Querying {collection_name}")
    

    # Query Embedding Generation using colpali for vector similarity comparison
    logger.info(f"Generating Embeddings for {collection_name}")
    query_embedding = colpali.process_query(query)
    query_embedding = query_embedding[0].float().numpy()


    # Initalize Colbert Retriever for data base similarity search
    logger.info(f"Searching {collection_name} for document page")
    retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
    results = retriever.search(query_embedding, topk=topk)

    
    # Request VLM to answer and attatch recommended pages to reply 
    logger.info(f"VLM is reading  {collection_name}")
    images_in_ref = [result[2] for result in results]
    message = vlm.generate(query=query, image_paths=images_in_ref)
    message['recommended_pages'] = images_in_ref
    
    logger.info(f"Returining results for {collection_name}")

    return jsonify(message), 200



# Document Uplaod API to parse whole documents for vector conversion
@rag_bp.route('/upload', methods=['POST'])
def upload():

    # Retrive pdf document and collection name
    request_json = request.get_json()
    collection_name = request_json['collection_name']
    file_list = request_json['files']
    
    if not file_list:
        return jsonify({'error': 'No file provided'}), 400

    # Initialize a collection with vector index using Colbert Retriever for DB search later on
    retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
    retriever.create_collection()
    retriever.create_index()

    # Save each file as their original
    for file in file_list:
        base64_data = file['bytes']
        file_name = file['name']

        _, file_label, file_extention = retriever.get_file_parts(file_name)

        # Define the file path where you want to save the uploaded PDF
        directory = f"{document_storage_prefix}/{collection_name}/{file_label}"

        # Ensure the directory exists and save the PDF file to the specified path for document use later
        if not os.path.exists(directory):
            os.makedirs(directory)


        logger.info(f"Processing and Verifying {file_name} for {collection_name}")
    
        if not (file_extention in supported_extensions):
            return jsonify({'error': f'Error Processing {file_name} as its extension is not supported'}), 400

        logger.info(f"Saving file for {collection_name}")

        try:

            file_data = base64.b64decode(base64_data)    

            # Save the decoded data to a file
            with open(f'{directory}/{file_name}', 'wb') as file:
                file.write(file_data)

        except Exception as e:
            return jsonify({'error': str(e)}), 500


        # Convert pdf pages to individual images for Colpali
        logger.info(f"Rasterizing document for {collection_name}")
        retriever.rasterize(path=f'{directory}/{file_name}')


        # Convert to embeddings for vector store
        logger.info(f"Generating Embeddings for {collection_name}")
        images = [Image.open(f"{directory}/pages/" + name) for name in os.listdir(f"{directory}/pages")]
        image_embeddings = colpali.process_images(images)


        # Insert the embeddings to Milvus with meta data 
        logger.info(f"Storing embeddings to {collection_name}")
        filepaths = [f"{directory}/pages/" + name for name in os.listdir(f"{directory}/pages/")]

        for i in range(len(filepaths)):
            data = {
                "colbert_vecs": image_embeddings[i].float().numpy(),
                "doc_id": i,
                "filepath": filepaths[i],
            }
            retriever.insert(data)


    return jsonify({'status': 'success'}), 200


