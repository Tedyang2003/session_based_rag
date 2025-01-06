import requests
import base64

# VLLM Hand
class VlmHandler():
    '''
    Custom VLM Handler to send data to vlm for inference

    '''
    def __init__(self, api_url: str):
        
        self.api_url = api_url

    # Generate Data from Model API using base64 image and user query
    def generate(self, query, image_paths) -> str:

        base64_images = list(map(self.image_base64, image_paths))

        payload = {
            "model": "llama3.2-vision:11b",
            "messages": [
                {
                    "role": "system",
                    "content": """
                        You are a highly detailed and smart assistant.
                        Provide comprehensive, responses that are clear, logical, and well-supported by evidence from the given image, do not mention that they are image. 
                        if the image relates to the question, include the page number in your answer.
                        if the image does not relate to the question, dont mention any documents at all and answer from your own knowledge.
                        If no image is provided, just say that your upload is still processing.
                    """
                },
                {
                    "role": "user",
                    "content": 'Question: '+ query + "be more elaborate",
                    "images": base64_images
                }
            ],
            "stream": False
        }

        response = requests.post(self.api_url, json=payload)

        if response.status_code != 200: 
            raise Exception(f"Error from external API: {response.text}")

        data = response.json()
        message = data.get('message')

        return message


    # Open the image file in binary mode and convert to base64
    def image_base64(self, path):
        with open(path, 'rb') as image_file:
            # Read the image's contents
            image_data = image_file.read()
            # Encode the bytes using Base64
            base64_encoded = base64.b64encode(image_data)
            # Convert the bytes to a string
            base64_string = base64_encoded.decode('utf-8')
        
        return base64_string
