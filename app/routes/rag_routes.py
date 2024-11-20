from flask import Blueprint, request, jsonify
from rag_tools.embedding_tools import TritonEmbedder
from rag_tools.llm_tools import TritonLLM
from utils.data_extraction import read_pdf
from utils.data_processing import chunk_with_metadata
from rag_tools.milvus_tools import MilvusDB

rag_bp = Blueprint('rag', __name__)
emb_url = "http://triton-direct-s3-route-triton-inference-services.apps.nebula.sl/v2/models/all-minilm-l6-v2/infer"
llm_url = "http://triton-route-triton-inference-services.apps.nebula.sl/v2/models/meta-llama-3-8b-instruct-awq/generate"

embedder = TritonEmbedder(api_url=emb_url)
llm = TritonLLM(api_url=llm_url)
# milvus = MilvusDB()


@rag_bp.route('/query', methods=['POST'])
def query():

        

    return jsonify({"status":"success"}), 200


@rag_bp.route('/upload', methods=['POST'])
def upload():
    
    collection_name = request.form.get('collection_name')
    base64_file = request.files.get('file')

    if not base64_file:
        return jsonify({'error': 'No PDF file provided'}), 400
    
    if base64_file.mimetype != 'application/pdf':
        return jsonify({'error': 'Uploaded file is not a PDF'}), 400

    content_list = read_pdf(base64_file)
    chunked_data = chunk_with_metadata(content_list)
    
    chunk_embeddings, vector_dimension = embedder.embed_chunks(chunked_data[:3])
    
    # milvus.create_collection(collection_name, vector_dimension, metric_type="IP")
    
    # response = milvus.insert_chunks(collection_name, chunked_embeddings, )

    return jsonify({'name': chunk_embeddings, 'vd': vector_dimension}), 200


