import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from llama_index import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    Document,
    ServiceContext,
    set_global_service_context
)
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI

# Load environment variables
load_dotenv()

app = FastAPI(title="Document Analysis Agent API")

# Initialize OpenAI and LlamaIndex
llm = OpenAI(temperature=0, model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context)

# Global storage for documents and agents
documents = {}
query_engines = {}

class QueryRequest(BaseModel):
    query: str
    doc_ids: List[str]

class AnalysisResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        # Save uploaded file temporarily
        content = await file.read()
        doc_id = f"doc_{len(documents)}"
        temp_path = f"temp_{doc_id}.pdf"
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process document with LlamaIndex
        documents[doc_id] = SimpleDirectoryReader(input_files=[temp_path]).load_data()
        
        # Create query engine for the document
        index = GPTVectorStoreIndex.from_documents(documents[doc_id])
        query_engines[doc_id] = index.as_query_engine()
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return {"doc_id": doc_id, "message": "Document processed successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=AnalysisResponse)
async def query_documents(request: QueryRequest):
    """Query multiple documents using the agent."""
    try:
        # Create tools for each requested document
        tools = []
        for doc_id in request.doc_ids:
            if doc_id not in query_engines:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            
            tool = QueryEngineTool(
                query_engine=query_engines[doc_id],
                metadata=ToolMetadata(
                    name=f"document_{doc_id}",
                    description=f"Query document {doc_id} for information"
                )
            )
            tools.append(tool)
        
        # Create agent with tools
        agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
        
        # Execute query
        response = agent.chat(request.query)
        
        return AnalysisResponse(
            response=str(response),
            sources=[f"document_{doc_id}" for doc_id in request.doc_ids]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all processed documents."""
    return {"documents": list(documents.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
