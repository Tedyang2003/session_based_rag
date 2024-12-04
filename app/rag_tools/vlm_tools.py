import requests

class VlmHandler():
    '''
    Custom LLM class used to leverage LangChains functionality

    '''
    def __init__(self, api_url: str):
        
        self.api_url = api_url

    def generate(self, chat_history, query, image) -> str:

        payload = {
            "model": "llama3.2-vision",
            "messages": [
                *chat_history,
                {
                "role": "user",
                "content": query,
                "images": [image]
                }
            ]
        }

        response = requests.post(self.api_url, json=payload)

        if response.status_code != 200: 
            raise Exception(f"Error from external API: {response.text}")

        data = response.json()
        text_output = data.get('response')

        return text_output 

    
