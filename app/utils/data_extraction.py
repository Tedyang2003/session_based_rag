from PyPDF2 import PdfReader

def read_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text_with_pages = []
        
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            text_with_pages.append({
                'page': page_number,
                'content': text
            })

        return text_with_pages

    except Exception as e:
        return jsonify({'error': 'Failed to read PDF', 'details': str(e)}), 400
