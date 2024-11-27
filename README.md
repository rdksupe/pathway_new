## Prerequisites 

```
pip install -r requirements.txt

Add OpenAI key in the .env file.
ollama run nomic-embed-text:latest
```
This will pull and load the embedding and the chat completion models from Ollama and expose the endpoints at http://0.0.0.0:11434

If using some other base_url for your ollama installation you will need to change it in LiteLLMChat and Embedder instances in the pathway.py file.

## Running 
```
python pathway.py
```
Please note that the entire code has been tested and written in a Python venv of version 3.11.10

Running the code will start two servers:
   - Question Answering Server on `http://0.0.0.0:4005`
   - Document Store Server on `http://0.0.0.0:4004`

And will also start live indexing and embedding generation on the pdf files contained in the folder specified by the data_dir variable which by default has been set as  './knowledge_base'

## Exposed Endpoints

### Document Store Server (port 4004)

1. `/v1/retrieve`: Retrieves documents based on a query
#### Parameters :
query - "query for retrieval of documents"

k - number of top k documents to be returned
Sampple curl request 
```
curl "http://localhost:4004/v1/retrieve?query=annual%20turnover%20of%20dell&k=1"
```
2. `/v1/statistics`: Provides statistics about the document store
3. `/v1/inputs`: Lists all documents in the store

### Question Answering Server (port 4005)
4. `/generate`: Answers a given query using the RAG-based system
#### Parameters :
prompt - "query for retrieval of documents"

Sample python code to make a simple get request

```
curl -X POST "http://localhost:4005/generate" \
     -H "Content-Type: application/json" \
     -d '{"query": "what are the net product sales of AMZN in 2022 ?"}'

```