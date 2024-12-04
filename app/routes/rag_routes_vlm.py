from flask import Blueprint, request, jsonify
from rag_tools.colpali_handler import ColpaliHandler
from rag_tools.milvus_colbert import MilvusColbertRetriever
from rag_tools.vlm_tools import VlmHandler
from pymilvus import MilvusClient
from PIL import Image
import os
import base64
import logging



logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),  # Log to a file
                        logging.StreamHandler()          # Log to the console
                    ])
logger = logging.getLogger(__name__)

logger.info("Application Initialization Started")
logger.info(f"Using {os.getcwd()}")

rag_bp = Blueprint('rag', __name__)

# emb_url = "http://triton-direct-s3-route-triton-inference-services.apps.nebula.sl/v2/models/all-minilm-l6-v2/infer"
vlm_url = "http://triton-route-triton-inference-services.apps.nebula.sl/v2/models/meta-llama-3-8b-instruct-awq/generate"

logger.info("Initializing Colpali Engine and Downloading Model")
colpali = ColpaliHandler('auto')

logger.info("Initializing Milvus Retriever")
client = MilvusClient(uri="milvus.db")
vlm =  VlmHandler(api_url=vlm_url)


logger.info("Server is running...")
# Meant for single user query
@rag_bp.route('/query', methods=['POST'])
def query():
    
    request_json = request.get_json()
    collection_name = request_json['collection_name']
    query = request_json['query']

    retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
    
    # Change to Single Query Processing
    query_embedding = colpali.process_query(query)

    query_embedding = query_embedding[0].float().numpy()
    result = retriever.search(query_embedding, topk=2)

    print(result)
    return jsonify({'result': 'success'}), 200



# Meant for document upload.
@rag_bp.route('/upload', methods=['POST'])
def upload():

    collection_name = request.form.get('collection_name')
    file = request.files.get('file')

    if not file:
        return jsonify({'error': 'No PDF file provided'}), 400
    
    if file.mimetype != 'application/pdf':
        return jsonify({'error': 'Uploaded file is not a PDF'}), 400

    # Define the file path where you want to save the uploaded PDF
    file_path = f"document_storage/{collection_name}"


    # Ensure the directory exists
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Save the PDF file to the specified path
    try:
        file.save(f'{file_path}/original.pdf')
        # return jsonify({'message': 'PDF file successfully uploaded and saved'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Initialize a collection with vector index
    retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
    retriever.create_collection()
    retriever.create_index()

    # Convert pdf pages to individual images
    retriever.rasterize(pdf_path=f'{file_path}/original.pdf')

    images = [Image.open(f"./document_storage/{collection_name}/pages/" + name) for name in os.listdir(f"./document_storage/{collection_name}/pages")]

    # Convert to embeddings
    image_embeddings = colpali.process_images(images)

    # Insert the embeddings to Milvus
    filepaths = [f"./document_storage/{collection_name}/pages/" + name for name in os.listdir(f"./document_storage/{collection_name}/pages/")]

    for i in range(len(filepaths)):
        data = {
            "colbert_vecs": image_embeddings[i].float().numpy(),
            "doc_id": i,
            "filepath": filepaths[i],
        }
        retriever.insert(data)


    return jsonify({'status': 'success'}), 200


