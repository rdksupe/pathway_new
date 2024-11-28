import os
import logging
import pathway as pw
from typing import List, Dict, Any
from pathway.udfs import DiskCache, ExponentialBackoffRetryStrategy
from pathway.xpacks.llm import embedders, llms, parsers, prompts
from pathway.xpacks.llm.question_answering import AdaptiveRAGQuestionAnswerer
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.servers import DocumentStoreServer
from pathway.stdlib.indexing import UsearchKnnFactory,TantivyBM25Factory,HybridIndexFactory
from langchain.text_splitter import RecursiveCharacterTextSplitter
import threading
from  dotenv import load_dotenv

load_dotenv()
# Configure text splitting parameters for document chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)

class DocumentProcessor:
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        # Configure environment and logging
        os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract/tessdata/"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        self.host = host
        self.port = port
        self.app = None
        self.vector_store = None
        
        # Initialize LLM components with Ollama backend
        self.chat = llms.LiteLLMChat(
            model="openai/gpt-4o",
            retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6),
            cache_strategy=DiskCache(),
            temperature=0.0,
        )
        self.parser = parsers.ParseUnstructured()
        self.embedder = embedders.LiteLLMEmbedder(
            model='voyageai/voyage-3',
            cache_strategy=DiskCache()
        )

    def initialize_vector_store(self, path):
        """Initialize document store with provided file path"""
        source = pw.io.fs.read(path=path, with_metadata=True, format="binary")
        folder = pw.io.gdrive.read(
            object_id="1_ga91J5sZ_YcQdcNWcQXtrBHqvODpEa5",
            mode="streaming",
            object_size_limit=None,
            service_user_credentials_file="team-30-441514-d9a9da2d500c.json",
            with_metadata=True,
            file_name_pattern=None
        )

        usearch = UsearchKnnFactory(embedder=self.embedder)
        #knn = LshKnnFactory(embedder=self.embedder)
        bm25 = TantivyBM25Factory(ram_budget=524288000, in_memory_index=True)
        factories = [usearch,bm25]
        retriever_factory=HybridIndexFactory(factories,k=60)
        
        self.vector_store = DocumentStore.from_langchain_components(
            retriever_factory=retriever_factory,
            docs=[source,folder],
            parser=self.parser,
            splitter=text_splitter,
        )

    def setup_question_answerer(self):
        """Configure RAG-based question answering system"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
            
        self.app = AdaptiveRAGQuestionAnswerer(
            llm=self.chat,
            indexer=self.vector_store,
            long_prompt_template=prompts.prompt_citing_qa,
        )
        self.app.build_server(host=self.host, port=self.port)

    def start_server(self):
        """Launch question answering server in background thread"""
        if not self.app:
            raise ValueError("Question answerer not configured")
            
        server_thread = threading.Thread(
            target=self.app.run_server,
            name="BaseRAGQuestionAnswerer"
        )   
        server_thread.daemon = True
        server_thread.start()
        logging.info(f"Server started on http://{self.host}:{self.port}")
        return server_thread

    def setup_document_server(self):
        """Configure document store server"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
            
        self.app1 = DocumentStoreServer(
            host=self.host,
            port=4004,
            document_store=self.vector_store
        )
        
    def start_document_server(self):
        """Launch document server in background thread"""
        server_thread = threading.Thread(
            target=self.app1.run,
            name="BaseDocument"
        )   
        server_thread.daemon = True
        server_thread.start()
        return server_thread

def main():
    # Initialize data directory for document storage
    data_dir = "./kb_1001"
    os.makedirs(data_dir, exist_ok=True)
    
    # Set up and start servers
    processor = DocumentProcessor()
    processor.initialize_vector_store(data_dir)
    processor.setup_question_answerer()
    processor.setup_document_server()
    processor.start_document_server()
    
    try:
        server_thread = processor.start_server()
        server_thread.join()
    except KeyboardInterrupt:
        logging.info("Shutting down server...")

if __name__ == "__main__":
    main()
