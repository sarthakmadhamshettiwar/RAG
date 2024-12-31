# Fastapi server to run Vanilla RAG on local machine
from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pydantic import BaseModel
from RAG import RAG
API_KEY = 'YOUR_HF_API_KEY'
api_key = API_KEY
rag = RAG(api_key=api_key)
app = FastAPI()
docs = []

class Text(BaseModel):
    text: str

@app.get('/')
def homePage():
    return "Welcome to RAG Microservice"

# add new piece of text in chromaDB
@app.post("/add-text/")
async def addText(text: Text):
    rag.add_doc(text.text)
    docs.append(text.text)
    return {"message": 'Document added into DB'}


# get all the texts
@app.get("/get-texts")
def getText():
    return {'texts': docs}


# get the answer to query
@app.get("/get-answer")
async def getAnswer(query: Text):
    query_text = query.text
    # print(query_text)
    answer = rag.get_answer(query_text, 'short')
    return {"answer": answer}

@app.get("/get-answer-locally")
async def getAnswerLocally(query: Text):
    query_text = query.text
    # print(query_text)
    answer = rag.get_answer_locally(query_text, 'short')
    return {"answer": answer}


# add-pdf
@app.post("/add-pdf")
async def add_pdf(pdf_path: Text):
'''Upload the pdf using Post request at this route by giving local-url of textbook as a'''
    path = pdf_path.text
    
    try:
        # Load the PDF using PyMuPDF
        loaded_pdf = PyMuPDFLoader(path) # Use fitz.open for PyMuPDF
        
        all_data = loaded_pdf.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        result_texts = []
        for data in all_data:
            texts = text_splitter.create_documents([data.page_content])
            for t in texts:
                result_texts.append(t.page_content)
                # print(t.page_content)
                # print("----------------------------------------------------------------")
        
        for texts in result_texts:
            rag.add_doc(texts)
        return {"operation": "completed", "result_texts": result_texts}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    
