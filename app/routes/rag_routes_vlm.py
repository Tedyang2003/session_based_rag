from flask import Blueprint, request, jsonify
from rag_tools.colpali_handler import ColpaliHandler
from rag_tools.milvus_colbert import MilvusColbertRetriever
from rag_tools.vlm_tools import VlmHandler
from pymilvus import MilvusClient
from PIL import Image
import os
import logging
import base64
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


# ENV variables
VLM_URL = os.getenv('VLM_URL')
DEVICE_NAME = os.getenv('DEVICE_NAME')
MODEL_PATH = os.getenv('MODEL_PATH')
DOCUMENT_STORAGE_PREFIX = os.getenv('DOCUMENT_STORAGE_PREFIX')
MILVUS_DB = os.getenv('MILVUS_DB')
TOPK = int(os.getenv('TOPK'))
SUPPORTED_EXTENSIONS = os.getenv('SUPPORTED_EXTENSIONS').split(',')
APP_LOG = os.getenv('APP_LOG')

# Initialize Logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{APP_LOG}/app.log"),  # Log to a file
                        logging.StreamHandler()          # Log to the console
                    ])
logger = logging.getLogger(__name__)

logger.info(f"VLM_URL: {VLM_URL}")
logger.info(f"DEVICE_NAME: {DEVICE_NAME}")
logger.info(f"MODEL_PATH: {MODEL_PATH}")
logger.info(f"DOCUMENT_STORAGE_PREFIX: {DOCUMENT_STORAGE_PREFIX}")
logger.info(f"MILVUS_DB: {MILVUS_DB}")
logger.info(f"TOPK: {TOPK}")
logger.info(f"SUPPORTED_EXTENSIONS: {SUPPORTED_EXTENSIONS}")
logger.info(f"APP_LOG: {APP_LOG}")

logger.info("Application Initialization Started")
logger.info(f"Using {os.getcwd()}")


# Ensure the directory exists and save the PDF file to the specified path for document use later
if not os.path.exists(DOCUMENT_STORAGE_PREFIX):
    os.makedirs(DOCUMENT_STORAGE_PREFIX)

# Intialize Components
logger.info("Initializing Colpali Engine and Downloading Model")
colpali = ColpaliHandler(device_name=DEVICE_NAME, model_path=MODEL_PATH)

logger.info("Initializing Milvus Retriever")

# Change to stand alone later on
client = MilvusClient(f'{DOCUMENT_STORAGE_PREFIX}/{MILVUS_DB}')
vlm =  VlmHandler(api_url=VLM_URL)

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
    logging.info(f"Query Embedding {query_embedding}")

    # Initalize Colbert Retriever for data base similarity search
    logger.info(f"Searching {collection_name} for document page")
    retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
    
    results = retriever.search(query_embedding, topk=TOPK)
    images_in_ref = [result[3] for result in results]

    ref_pages = [str(result[1]+1) for result in results]

    # Relevance check of adjacent pages to original recommendations
    # images_in_ref = []
    # first_result = results[0]
    # index = 0

    # while index < len(results):
    #     result = results[index]

    #     if result[3] not in images_in_ref:
    #         images_in_ref.append(result[3])

    #     next_page_id = result[1] + 1
        
    #     # Check if next_page_id is already in results
    #     if not any(res[1] == next_page_id for res in results):
    #         next_page = retriever.is_page_relevant(
    #             collection_name=collection_name,
    #             page_id1=first_result[1],
    #             page_id2=next_page_id,
    #             doc_id=result[2],
    #             thresh=1
    #         )

    #         if next_page:
    #             results.append(next_page)
            
    #     index += 1

    # logger.info(f"Relevant pages are: {results}")


    # Request VLM to answer and attatch recommended pages to reply 
    logger.info(f"VLM is reading relevant pages{collection_name}")

    message = vlm.generate(query=query, image_paths=images_in_ref)
    message['text_output'] = f"{message['content']} \n\nRecommended Reference: Page {', '.join(ref_pages)}"
    del message['content']
    
    logger.info(f"Returining results for {collection_name}")

    return jsonify(message), 200



# Document Uplaod API to parse whole documents for vector conversion
@rag_bp.route('/upload', methods=['POST'])
def upload():

    # Retrive pdf document and collection name
    request_json = request.get_json()
    collection_name = request_json['collection_name']
    file = request_json['file']
    
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Initialize a collection with vector index using Colbert Retriever for DB search later on
    retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
    retriever.create_collection()
    retriever.create_index()

    # Save file as their original
    base64_data = file['bytes']
    file_name = file['name']

    _, file_label, file_extention = retriever.get_file_parts(file_name)

    # Define the file path where you want to save the uploaded PDF
    directory = f"{DOCUMENT_STORAGE_PREFIX}/{collection_name}/{file_label}"

    # Ensure the directory exists and save the PDF file to the specified path for document use later
    if not os.path.exists(directory):
        os.makedirs(directory)


    logger.info(f"Processing and Verifying {file_name} for {collection_name}")

    if not (file_extention in SUPPORTED_EXTENSIONS):
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
            "page_id": i,
            'doc_id': file_label,
            "filepath": filepaths[i]
        }
        retriever.insert(data)


    return jsonify({'status': 'success'}), 200


