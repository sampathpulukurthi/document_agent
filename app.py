import gradio as gr
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import PyPDF2

# Load environment variables
load_dotenv()

class DocumentAgent:
    def __init__(self):
        self.documents = {}
        self.vector_stores = {}
        self.chat_history = []
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.embeddings = OpenAIEmbeddings()
        
    def process_document(self, file_paths: List[str]) -> tuple:
        """Process uploaded documents and store them."""
        try:
            processed_docs = []
            for file_path in file_paths:
                # Read PDF content
                pdf_reader = PyPDF2.PdfReader(file_path)
                text_content = ""
                page_contents = []
                
                # Extract text with page numbers
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    text_content += page_text
                    page_contents.append({
                        'text': page_text,
                        'page': page_num
                    })
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_text(text_content)
                
                # Create document ID and store
                doc_id = f"doc_{len(self.documents)}"
                
                # Create vector store with metadata
                chunk_with_metadata = []
                for chunk in chunks:
                    # Find which page this chunk belongs to
                    for page_content in page_contents:
                        if chunk in page_content['text']:
                            chunk_with_metadata.append({
                                'text': chunk,
                                'metadata': {
                                    'page': page_content['page'],
                                    'doc_id': doc_id
                                }
                            })
                            break
                
                vector_store = FAISS.from_texts(
                    texts=[c['text'] for c in chunk_with_metadata],
                    embedding=self.embeddings,
                    metadatas=[c['metadata'] for c in chunk_with_metadata]
                )
                self.vector_stores[doc_id] = vector_store
                self.documents[doc_id] = chunks
                processed_docs.append(doc_id)
            
            return f"✅ Processed {len(processed_docs)} documents successfully!", list(self.documents.keys())
        except Exception as e:
            return f"❌ Error processing document: {str(e)}", list(self.documents.keys())

    def query_documents(self, question: str, selected_docs: List[str]) -> str:
        """Query the selected documents using conversational retrieval."""
        try:
            if not selected_docs:
                return "Please select at least one document to query."
            
            # Combine vector stores for selected documents
            if len(selected_docs) == 1:
                vector_store = self.vector_stores[selected_docs[0]]
            else:
                # Merge relevant vector stores
                vector_store = self.vector_stores[selected_docs[0]]
                for doc_id in selected_docs[1:]:
                    vector_store.merge_from(self.vector_stores[doc_id])
            
            # Create retrieval chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                verbose=True
            )
            
            # Get response
            result = qa_chain({"question": question, "chat_history": self.chat_history})
            
            # Update chat history
            self.chat_history.append((question, result["answer"]))
            
            # Format answer with source references
            answer = result["answer"]
            source_docs = result["source_documents"]
            
            # Add source references
            references = "\n\nSources:\n"
            seen_refs = set()
            for doc in source_docs:
                # Get document ID and page number from metadata
                doc_id = doc.metadata.get('doc_id', 'Unknown')
                page_num = doc.metadata.get('page', 'Unknown')
                ref = f"{doc_id} (Page {page_num})"
                
                if ref not in seen_refs:
                    references += f"- {ref}\n"
                    seen_refs.add(ref)
            
            return f"{answer}\n{references}"
        except Exception as e:
            return f"Error processing query: {str(e)}"

# Initialize the agent
doc_agent = DocumentAgent()

# Define the Gradio interface
with gr.Blocks(title="Document Analysis Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📚 Document Analysis Agent
    Upload PDF documents and ask questions about their content. The AI agent will analyze the documents
    and provide detailed answers based on the content.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload PDF Documents",
                file_types=[".pdf"],
                file_count="multiple",
                type="filepath"
            )
            upload_button = gr.Button("Process Document", variant="primary")
            status_text = gr.Textbox(
                label="Status",
                placeholder="Upload status will appear here...",
                interactive=False
            )
            
            doc_list = gr.Dropdown(
                label="Available Documents",
                multiselect=True,
                choices=[],
                value=None,
                interactive=True
            )
        
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about the uploaded documents...",
                lines=3
            )
            query_button = gr.Button("Ask Question", variant="primary")
            answer_output = gr.Markdown(
                label="Answer",
                value="Answers will appear here..."
            )
    
    # Define interactions
    def process_upload(files):
        if not files:
            return "Please upload at least one file.", gr.Dropdown(choices=[])
        try:
            file_paths = [f.name for f in files]
            status, docs = doc_agent.process_document(file_paths)
            return status, gr.Dropdown(choices=docs, value=docs[0] if docs else None)
        except Exception as e:
            return f"Error uploading files: {str(e)}", gr.Dropdown(choices=[])
    
    def update_answer(question, selected_docs):
        if not question.strip():
            return "Please enter a question."
        if not selected_docs or len(selected_docs) == 0:
            return "Please select at least one document to query."
        try:
            return doc_agent.query_documents(question, selected_docs)
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    # Set up event handlers
    upload_button.click(
        process_upload,
        inputs=[file_upload],
        outputs=[status_text, doc_list]
    )
    
    query_button.click(
        update_answer,
        inputs=[question_input, doc_list],
        outputs=answer_output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7868,
        show_api=False
    )
