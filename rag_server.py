from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from urllib.parse import quote
import requests
import os
from openai import OpenAI
from dotenv import load_dotenv
app = FastAPI()
load_dotenv()
# OpenAI client configuration
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"

client = OpenAI(api_key="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")

class Query(BaseModel):
    query: str
    max_tokens: int = 1000
    num_docs: int = 5

class AnswerResponse(BaseModel):
    answer: str

def query_retrieval_service(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Query the local retrieval service for relevant documents."""
    try:
        encoded_query = quote(query)
        response = requests.get(f"http://0.0.0.0:4004/v1/retrieve?query={encoded_query}&k={k}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying retrieval service: {str(e)}")

def format_context(documents: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into context string."""
    formatted_docs = []
    for i, doc in enumerate(documents, 1):
        formatted_docs.append(f"Document {i}:\n{doc.get('text', '')}")
    return "\n\n".join(formatted_docs)

def generate_answer_openai(query: str, retrieved_docs: List[Dict[str, Any]], max_tokens: int = 1000) -> str:
    """Generate an answer using OpenAI model."""
    try:
        context = format_context(retrieved_docs)
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a precise and factual research assistant. Answer questions based solely on the provided context.
                    Important Instructions:
                    - Base your answer ONLY on the provided context documents
                    - Cite every statement using [Document X, Page Y] format
                    - Use quotes when directly quoting text
                    - If information isn't available in the context, state: "This information is not available in the provided documents."
                    - If you find conflicting information, point it out"""
                },
                {
                    "role": "user",
                    "content": f"""Context:\n{context}\n\nQuestion: {query}\n\nProvide a well-structured answer with clear citations:"""
                }
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=max_tokens,
            model=model_name
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer with OpenAI: {str(e)}")

@app.post("/generate", response_model=AnswerResponse)
async def generate(query_request: Query):
    """
    Generate an answer for a given query using retrieved documents and OpenAI LLM.
    Returns only the final answer from the LLM.
    """
    # Retrieve relevant documents
    retrieved_docs = query_retrieval_service(query_request.query, query_request.num_docs)
    
    # Generate answer using OpenAI
    answer = generate_answer_openai(
        query_request.query, 
        retrieved_docs, 
        query_request.max_tokens
    )
    
    return AnswerResponse(answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4005)
