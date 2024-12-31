import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

# for gemma-7b
import requests
API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b"
headers = {"Authorization": "Bearer YOUR_API_KEY"}


# for local LLM
import json


def gemma_query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


class RAG:
    def __init__(self, api_key):
        self.curr_docs = 0
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=60,
            length_function=len,
            is_separator_regex=False,
        )
        self.api_key = api_key
        self.client = InferenceClient(
            "mistralai/Mistral-7B-Instruct-v0.2",
            token=api_key,
        )
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="test-0")

    def split_text(self, doc):
        texts = self.text_splitter.create_documents([doc])
        return texts

    def add_doc(self, doc):
        texts = self.split_text(doc)
        for text in texts:
            self.collection.add(documents=[str(text)[14:-1]], ids=[str(self.curr_docs)])
            self.curr_docs += 1

    def retrieve_doc(self, query):
        results = self.collection.query(
            query_texts=[query],
            n_results=3,
        )
        return results

    def get_contexts(self, query):
        results = self.retrieve_doc(query)
        return results['documents'][0]

    def get_answer(self, query, length):
        contexts = self.get_contexts(query)
        contexts_concatenated = '.'.join(filter(None, contexts))

        prompt = f'''
        Context: {contexts_concatenated}.
        Question: Answer in {length} based on the context. {query}?
        '''

        response = self.client.text_generation(
            prompt,
            max_new_tokens=500,
            return_full_text=False
        )
        
        return response
    
    def get_answer_locally(self, query, length):
    '''
    	Using ollama to run Mistral 7B quantized model on Local Machine
    '''
        contexts = self.get_contexts(query)
        contexts_concatenated = '.'.join(filter(None, contexts))
        
        prompt = f'''
        Context: {contexts_concatenated}.
        Question: Answer in {length} based on the context. {query}?
        '''
        url = "http://localhost:11434/api/generate"
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "model": "mistral",
            "prompt": prompt,
            "stream": False 
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            try:
                response_data = response.json()
                if "response" in response_data:
                    actual_response = response_data["response"]
                    # print(actual_response)
                    return {"response":actual_response}
                else:
                    # print("Error: 'response' key not found in JSON data")
                    return {"response":"response' key not found in JSON data"}
                
            except json.JSONDecodeError:
                return {"response": "Error: Failed to decode JSON response"}
        else:
            return {"response":"Error: response.status_code, response.text"}
        
    
