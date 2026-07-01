import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# RAG Modules
from load_pdf import load_pdf
from clean_text import clean_text
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

load_dotenv()
app = FastAPI(title="Advanced RAG API Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline reference dictionary
RAG_PIPELINE = {}

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    print("=== STARTING RAG INGESTION ===")
    
    # 1. Load and clean full PDF
    docs = load_pdf("Enterprise_AI_Knowledge_Base_200_Pages.pdf")
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        
    # 2. Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # 3. Build dense FAISS index
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = FAISS.from_documents(chunks, embedding)
    
    # 4. Initialize retrieval & rerank components
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    mq_retriever = MultiQueryRetriever.from_llm(retriever=vector_db.as_retriever(search_kwargs={"k": 5}), llm=llm)
    rerank_model = CrossEncoder("BAAI/bge-reranker-base")
    
    # 5. Set up contextual compression pipeline
    embedding_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.4)
    extractor = LLMChainExtractor.from_llm(llm)
    compressor_pipeline = DocumentCompressorPipeline(transformers=[embedding_filter, extractor])
    
    # Cache references
    RAG_PIPELINE["mq_retriever"] = mq_retriever
    RAG_PIPELINE["rerank_model"] = rerank_model
    RAG_PIPELINE["embedding"] = embedding
    RAG_PIPELINE["compressor_pipeline"] = compressor_pipeline
    RAG_PIPELINE["llm"] = llm
    
    print("=== RAG PIPELINE READY ===")

@app.get("/")
def get_index():
    return FileResponse("../index.html")

@app.post("/api/query")
def query_rag(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Step 1: MultiQuery retrieval
        retrieved_docs = RAG_PIPELINE["mq_retriever"].invoke(question)

        # Step 2: BGE Reranking
        pairs = [(question, doc.page_content) for doc in retrieved_docs]
        scores = RAG_PIPELINE["rerank_model"].predict(pairs)
        ranked = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)

        all_retrieved = []
        for score, doc in ranked:
            all_retrieved.append({
                "content": doc.page_content,
                "score": float(score),
                "title": doc.metadata.get("source", "PDF Document"),
                "pages": f"Page {doc.metadata.get('page', 'unknown')}"
            })

        # Step 3: Context Compression (top 3 chunks)
        top_k = [doc for score, doc in ranked[:3]]
        temp_faiss = FAISS.from_documents(top_k, RAG_PIPELINE["embedding"])
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=temp_faiss.as_retriever(search_kwargs={"k": 3}),
            base_compressor=RAG_PIPELINE["compressor_pipeline"]
        )
        compressed_docs = compression_retriever.invoke(question)
        context_str = "\n\n".join([doc.page_content for doc in compressed_docs])

        # Step 4: Final LLM Generation
        prompt = f"""
Use ONLY the retrieved context below to answer the question.
If the answer is unavailable in the context, say: Information not found.
Do not guess or use external knowledge.

Context:
{context_str if context_str else "No context available."}

Question:
{question}
"""
        answer = RAG_PIPELINE["llm"].invoke(prompt).content

        # Step 5: Fact Check Guardrail
        fact_checker_prompt = f"""
Context:
{context_str}

Answer:
{answer}

Check if answer is fully supported by context. Reply only SUPPORTED or NOT_SUPPORTED.
"""
        verification = RAG_PIPELINE["llm"].invoke(fact_checker_prompt).content.strip()

        return {
            "answer": answer,
            "verification": verification,
            "compressed_context": context_str,
            "retrieved_documents": all_retrieved[:5]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
