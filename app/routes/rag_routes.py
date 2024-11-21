from flask import Blueprint, request, jsonify
from rag_tools.embedding_tools import TritonEmbedder
from rag_tools.llm_tools import TritonLLM
from utils.data_extraction import read_pdf
from utils.data_processing import chunk_with_metadata, format_embeddings
from rag_tools.milvus_tools import MilvusDB

rag_bp = Blueprint('rag', __name__)
emb_url = "http://triton-direct-s3-route-triton-inference-services.apps.nebula.sl/v2/models/all-minilm-l6-v2/infer"
llm_url = "http://triton-route-triton-inference-services.apps.nebula.sl/v2/models/meta-llama-3-8b-instruct-awq/generate"
template = """
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>
<context>
You are an AI assistant that provides detailed information for general user queries in an organization, with some context support. 
Throughout the discussion ensure that your responses:
1. Provide relevant and appropriate information that users can understand easily.
2. Maintain a supportive and empathetic tone, avoiding and language that is percieved as discriminatory or insensitve.
3. Format responses as detailed paragraphs with clear action steps.
4. Provide formatted coding responses where appropriate.
5. You will be given context first, use it only as a piece of reference.
</context>

<format>
1. Ensure that you use proper markdown formats for your answers.
2. Ensure proper color coding of code syntax and provide comments. 
4. Do not specify the fact that you referred to any contexts.
</format>
<|eot_id|>

{chat_history}

<|start_header_id|>user<|end_header_id>
This is the context:
{context}

This is the question:
{question}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id>
"""



embedder = TritonEmbedder(api_url=emb_url)
llm = TritonLLM(api_url=llm_url, template=template)
milvus = MilvusDB(host='a', port='b')


@rag_bp.route('/query', methods=['POST'])
def query():
    
    request_json = request.get_json()
    collection_name = request_json['collection_name']
    query = request_json['query']
    chat_history = request_json['chat_history']

    embedding = embedder.embed_query(query)

    # Change this to non hardcoded and use key args
    results = milvus.search_chunks(collection_name, embedding, limit=3)
    
    # TO DO: Functionalize this
    retrieved_lines_with_distances = [(res["entity"]["chunk"], res["distance"]) for res in results[0]]
    context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])

    response = llm.generate(chat_history = chat_history, context=context, question=query)

    return jsonify({query: response, 'context': context}), 200

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
    
    chunk_embeddings = embedder.embed_chunks(chunked_data)
    
    formatted_embeddings = format_embeddings(chunk_embeddings)

    milvus.create_collection(collection_name, metric_type="IP")
    
    response = milvus.insert_chunks(collection_name, formatted_embeddings)

    return jsonify({'status': 'success'}), 200


