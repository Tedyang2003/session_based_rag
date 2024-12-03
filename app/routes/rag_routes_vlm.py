from flask import Blueprint, request, jsonify
from rag_tools.colpali_handler import ColpaliHandler
from rag_tools.milvus_colbert import MilvusColbertRetriever
from rag_tools.vlm_tools import VlmHandler
from pymilvus import MilvusClient
from PIL import Image
import os
import base64


rag_bp = Blueprint('rag', __name__)
# emb_url = "http://triton-direct-s3-route-triton-inference-services.apps.nebula.sl/v2/models/all-minilm-l6-v2/infer"
vlm_url = "http://triton-route-triton-inference-services.apps.nebula.sl/v2/models/meta-llama-3-8b-instruct-awq/generate"

colpali = ColpaliHandler(api_url=emb_url)
milvus = MilvusColbertRetriever(host='a', port='b')
client = MilvusClient(uri="milvus.db")
vlm =  VlmHandler(api_url=vlm_url)



# Meant for single user query
@rag_bp.route('/query', methods=['POST'])
def query():
    
    request_json = request.get_json()
    collection_name = request_json['collection_name']
    query = request_json['query']

    retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
    
    # Change to Single Query Processing
    query_embedding = colpali.process_query(query)

    query_embedding = query_embedding.float().numpy()
    result = retriever.search(query_embedding, topk=1)


    return jsonify({'result': result}), 200



# Meant for document upload.
@rag_bp.route('/upload', methods=['POST'])
def upload():

    collection_name = request.form.get('collection_name')
    base64_file = request.files.get('file')

    if not base64_file:
        return jsonify({'error': 'No PDF file provided'}), 400
    
    if base64_file.mimetype != 'application/pdf':
        return jsonify({'error': 'Uploaded file is not a PDF'}), 400

    # Write the base64 data to a PDF file
    pdf_data = base64.b64decode(base64_file)
    file_path = f"{collection_name}/original.pdf"
    with open(file_path, "wb") as pdf_file:
        pdf_file.write(pdf_data)

    # Initialize a collection with vector index
    retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
    retriever.create_collection()
    retriever.create_index()

    # Convert pdf pages to individual images
    retriever.rasterize(pdf_path=file_path)

    images = [Image.open(f"./{collection_name}/pages/" + name) for name in os.listdir("./pages")]

    # Convert to embeddings
    image_embeddings = colpali.process_images(images)

    # Insert the embeddings to Milvus
    filepaths = ["./pages/" + name for name in os.listdir("./pages")]
    for i in range(len(filepaths)):
        data = {
            "colbert_vecs": image_embeddings[i].float().numpy(),
            "doc_id": i,
            "filepath": filepaths[i],
        }
        milvus.insert(data)


    return jsonify({'status': 'success'}), 200


