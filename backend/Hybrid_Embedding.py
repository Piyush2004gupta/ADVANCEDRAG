from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from load_pdf import load_pdf
from clean_text import clean_text
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. Load and clean original PDF (Full PDF)
docs = load_pdf("Enterprise_AI_Knowledge_Base_200_Pages.pdf")
for doc in docs:
    doc.page_content = clean_text(doc.page_content)

# 2. Chunk documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

# 3. Dense FAISS + Sparse BM25 Retrievers
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
dense_retriever = FAISS.from_documents(documents, embedding).as_retriever(search_kwargs={"k": 2})

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 2

# 4. Ensemble Hybrid Retriever
hybrid_retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.5, 0.5])
results = hybrid_retriever.invoke("Explain semantic search and hybrid search in production systems")

for doc in results[:3]:
    print(f"Content: {doc.page_content[:300]}...\n" + "-" * 40)