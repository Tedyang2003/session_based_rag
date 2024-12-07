from flask import Blueprint, request, jsonify
from rag_tools.colpali_handler import ColpaliHandler
from rag_tools.milvus_colbert import MilvusColbertRetriever
from rag_tools.vlm_tools import VlmHandler
from pymilvus import MilvusClient
from PIL import Image
import os
import logging


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
milvus_uri = "/data/document_storage/milvus.db"
document_storage_prefix = "/data/document_storage"
topk= 1


# Intialize Components
logger.info("Initializing Colpali Engine and Downloading Model")
colpali = ColpaliHandler(device_name=device_name, model_path=model_path)

logger.info("Initializing Milvus Retriever")
client = MilvusClient(uri=milvus_uri)
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
    logger.info(f"VLM is Reading for {collection_name}")
    images_in_ref = [result[2] for result in results]
    message = vlm.generate(query=query, image_paths=images_in_ref)
    message['recommended_pages'] = images_in_ref
    
    logger.info(f"Returining results for {collection_name}")

    return jsonify(message), 200



# Document Uplaod API to parse whole documents for vector conversion
@rag_bp.route('/upload', methods=['POST'])
def upload():

    # Retrive pdf document and collection name
    collection_name = request.form.get('collection_name')
    file = request.files.get('file')
    logger.info(f"Processing and Verifyingfile for {collection_name}")

    if not file:
        return jsonify({'error': 'No PDF file provided'}), 400
    
    if file.mimetype != 'application/pdf':
        return jsonify({'error': 'Uploaded file is not a PDF'}), 400


    # Define the file path where you want to save the uploaded PDF
    file_path = f"{document_storage_prefix}/{collection_name}"

    # Ensure the directory exists and save the PDF file to the specified path for document use later
    logger.info(f"Saving file for {collection_name}")
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    try:
        file.save(f'{file_path}/original.pdf')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


    # Initialize a collection with vector index using Colbert Retriever for DB search later on
    retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
    retriever.create_collection()
    retriever.create_index()


    # Convert pdf pages to individual images for Colpali
    logger.info(f"Rasterizing document for {collection_name}")
    retriever.rasterize(pdf_path=f'{file_path}/original.pdf')


    # Convert to embeddings for vector store
    logger.info(f"Generating Embeddings for {collection_name}")
    images = [Image.open(f"{file_path}/pages/" + name) for name in os.listdir(f"{file_path}/pages")]
    image_embeddings = colpali.process_images(images)

    # Insert the embeddings to Milvus with meta data 
    logger.info(f"Storing embeddings to {collection_name}")
    filepaths = [f"{file_path}/pages/" + name for name in os.listdir(f"{file_path}/pages/")]

    for i in range(len(filepaths)):
        data = {
            "colbert_vecs": image_embeddings[i].float().numpy(),
            "doc_id": i,
            "filepath": filepaths[i],
        }
        retriever.insert(data)


    return jsonify({'status': 'success'}), 200


