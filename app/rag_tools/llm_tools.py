from langchain.prompts import ChatPromptTemplate
import requests



class TritonLLM():
    '''
    Custom LLM class used to leverage LangChains functionality

    '''
    def __init__(self, api_url: str, template: str):
        
        self.api_url = api_url
        self.template = template

    def generate(self, **kwargs) -> str:

        prompt_text = self.format_template(**kwargs)
        
        headers = {"Content-Type": "application/json"}       
        payload = {
            "text_input": prompt_text,
            "parameters": {"stream": False, "temperature": 0, "max_tokens": 1000},
            "exclude_input_in_output": True
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

    
