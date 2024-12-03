from langchain.prompts import ChatPromptTemplate
import requests



class TritonLLM():
    '''
    Custom LLM class used to leverage LangChains functionality

    '''
    def __init__(self, api_url: str, template: str):
        
        self.api_url = api_url

    def generate(self, query, image) -> str:


        payload = {
            "model": "llama3.2-vision",
            "messages": [
                {
                "role": "user",
                "content": "what is in this image?",
                "images": ["<base64-encoded image data>"]
                }
            ]
        }

        response = requests.post(self.api_url, json=payload)

        if response.status_code != 200: 
            raise Exception(f"Error from external API: {response.text}")

        data = response.json()
        text_output = data.get('text_output')

        return text_output 
        self.template = template

    def format_template(self, **kwargs) -> str:
        """Format the template by replacing placeholders with dynamic values from kwargs."""
        prompt = ChatPromptTemplate.from_template(self.template)
        return prompt.format(**kwargs)

    
