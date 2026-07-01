import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# RAG Ingestion & Cleaning imports
from load_pdf import load_pdf
from clean_text import clean_text

# LangChain and Vector store
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Qdrant Vector store
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# MultiQuery and Compressor
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor
)

# BGE Reranker
from sentence_transformers import CrossEncoder

# Load environment variables
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment!")
    sys.exit(1)

app = FastAPI(title="Advanced RAG API Server")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify React URL (e.g. http://localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse

@app.get("/")
def get_index():
    return FileResponse("../index.html")

# Global variables to store the initialized index and retrievers
RAG_DATA = {
    "agentic_docs": [],
    "embedding": None,
    "vector_faiss": None,
    "hybrid_retriever": None,
    "vector_qdrant": None,
    "qdrant_client": None,
    "llm_mq": None,
    "mq_retriever": None,
    "rerank_model": None,
    "llm_generation": None,
    "prompt_template": None,
    "compressor_pipeline": None
}

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    print("=" * 60)
    print("INITIALIZING ADVANCED RAG PIPELINE (INGESTION)...")
    print("=" * 60)

    # 1. Load and clean PDF
    pdf_path = "Enterprise_AI_Knowledge_Base_200_Pages.pdf"
    print(f"Loading PDF: {pdf_path} (takes around 60 seconds)...")
    docs = load_pdf(pdf_path)
    print(f"Loaded {len(docs)} pages.")

    print("Cleaning document text...")
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    print("Cleaned text successfully.")

    # 2. Agentic Chunking
    print("Running parallel Agentic Chunking...")
    llm_chunker = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Group pages into sections (e.g. 15 pages per group)
    group_size = 15
    page_groups = [docs[i:i + group_size] for i in range(0, len(docs), group_size)]
    
    agentic_docs = []

    def process_group(index, group):
        group_text = "\n\n".join([f"--- Page {g.metadata.get('page', 'unknown')} ---\n{g.page_content}" for g in group])
        group_text = group_text[:6000]

        prompt = f"""
Read the following document group (Pages {index * group_size + 1} to {min((index + 1) * group_size, len(docs))}):
{group_text}

Split this document into logical sections.
For each section, extract and return:
Title: <Section Title>
Summary: <Brief summary of this section>
Content: <Main body/paragraphs of this section>
"""
        try:
            response = llm_chunker.invoke(prompt)
            response_text = response.content
            
            sections = response_text.split("Title:")
            parsed_chunks = []
            for sec in sections:
                if not sec.strip():
                    continue
                lines = sec.split("Summary:")
                title = lines[0].strip()
                summary = ""
                content = ""
                if len(lines) > 1:
                    lines2 = lines[1].split("Content:")
                    summary = lines2[0].strip()
                    if len(lines2) > 1:
                        content = lines2[1].strip()
                
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={
                            "title": title,
                            "summary": summary,
                            "source_pages": f"{index * group_size + 1}-{min((index + 1) * group_size, len(docs))}"
                        }
                    )
                    parsed_chunks.append(doc)
            
            if not parsed_chunks:
                parsed_chunks.append(Document(
                    page_content=response_text,
                    metadata={"title": f"Section {index + 1}", "summary": "Automatically chunked section", "source_pages": f"{index * group_size + 1}"}
                ))
            return parsed_chunks
        except Exception as e:
            print(f"Warning: Chunking failed for group {index}: {e}")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
            return text_splitter.split_documents(group)

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda item: process_group(item[0], item[1]), enumerate(page_groups)))
    
    for res_list in results:
        agentic_docs.extend(res_list)

    print(f"Created {len(agentic_docs)} chunks from agentic chunking.")

    # 3. Dense FAISS + Sparse BM25
    print("Building Hybrid Retrievers (FAISS + BM25)...")
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_faiss = FAISS.from_documents(agentic_docs, embedding)
    dense_retriever = vector_faiss.as_retriever(search_kwargs={"k": 5})

    bm25_retriever = BM25Retriever.from_documents(agentic_docs)
    bm25_retriever.k = 5

    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    # 4. Qdrant DB
    print("Storing chunks in Qdrant (In-Memory)...")
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="rag_collection",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    vector_qdrant = QdrantVectorStore(
        client=client,
        collection_name="rag_collection",
        embedding=embedding
    )
    vector_qdrant.add_documents(agentic_docs)

    # 5. Reranker & LLM Generation tools
    print("Loading models and structuring prompts...")
    llm_mq = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_qdrant.as_retriever(search_kwargs={"k": 5}),
        llm=llm_mq
    )

    rerank_model = CrossEncoder("BAAI/bge-reranker-base")
    llm_generation = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Document compressor pipeline
    embedding_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.4)
    extractor = LLMChainExtractor.from_llm(llm_mq)
    compressor_pipeline = DocumentCompressorPipeline(transformers=[embedding_filter, extractor])

    prompt_template = ChatPromptTemplate.from_template(
        """
You are a professional AI Assistant.

Instructions:
Use ONLY the retrieved context below to answer the question.
If the answer is unavailable in the context, say: Information not found.
Do not guess or use external knowledge.
The answer should be: Accurate, Short, and Professional.

Context:
{context}

Question:
{question}
"""
    )

    # Store references globally
    RAG_DATA["agentic_docs"] = agentic_docs
    RAG_DATA["embedding"] = embedding
    RAG_DATA["vector_faiss"] = vector_faiss
    RAG_DATA["hybrid_retriever"] = hybrid_retriever
    RAG_DATA["vector_qdrant"] = vector_qdrant
    RAG_DATA["qdrant_client"] = client
    RAG_DATA["llm_mq"] = llm_mq
    RAG_DATA["mq_retriever"] = mq_retriever
    RAG_DATA["rerank_model"] = rerank_model
    RAG_DATA["llm_generation"] = llm_generation
    RAG_DATA["prompt_template"] = prompt_template
    RAG_DATA["compressor_pipeline"] = compressor_pipeline

    print("=" * 60)
    print("ADVANCED RAG PIPELINE READY FOR QUERIES")
    print("=" * 60)


@app.post("/api/query")
def query_rag(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not RAG_DATA["hybrid_retriever"]:
        raise HTTPException(status_code=503, detail="RAG system is still initializing. Please wait.")

    try:
        # Step 7: MultiQuery retrieval
        retrieved_docs = RAG_DATA["mq_retriever"].invoke(question)

        # Step 8: BGE Reranking
        pairs = [(question, doc.page_content) for doc in retrieved_docs]
        scores = RAG_DATA["rerank_model"].predict(pairs)
        
        ranked_docs = sorted(
            zip(scores, retrieved_docs),
            key=lambda x: x[0],
            reverse=True
        )

        # Prepare source chunks list before compression to return to frontend
        all_retrieved = []
        for score, doc in ranked_docs:
            all_retrieved.append({
                "content": doc.page_content,
                "score": float(score),
                "title": doc.metadata.get("title", "Section"),
                "summary": doc.metadata.get("summary", ""),
                "pages": doc.metadata.get("source_pages", "")
            })

        # Select top 3 reranked documents for compression
        top_k_reranked = [doc for score, doc in ranked_docs[:3]]

        # Step 9: Context Compression
        temp_faiss = FAISS.from_documents(top_k_reranked, RAG_DATA["embedding"])
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=temp_faiss.as_retriever(search_kwargs={"k": 3}),
            base_compressor=RAG_DATA["compressor_pipeline"]
        )
        compressed_docs = compression_retriever.invoke(question)
        context_str = "\n\n".join([doc.page_content for doc in compressed_docs])

        # Step 11: LLM Generation
        formatted_prompt = RAG_DATA["prompt_template"].format_messages(
            context=context_str if context_str else "No context available.",
            question=question
        )
        response = RAG_DATA["llm_generation"].invoke(formatted_prompt)
        answer = response.content

        # Step 12: Hallucination Check
        fact_checker_prompt = f"""
You are a strict fact checker.

Context:
{context_str}

Answer:
{answer}

Check whether the generated answer is completely and fully supported by the provided context.
Reply ONLY with:
SUPPORTED
or
NOT_SUPPORTED
"""
        fact_check_response = RAG_DATA["llm_generation"].invoke(fact_checker_prompt)
        verification = fact_check_response.content.strip()

        return {
            "answer": answer,
            "verification": verification,
            "compressed_context": context_str,
            "retrieved_documents": all_retrieved[:5]  # Top 5 retrieved matches
        }

    except Exception as e:
        print(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
