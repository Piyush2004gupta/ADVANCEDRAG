from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor
)
from load_pdf import load_pdf
from clean_text import clean_text
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. Load and clean original PDF (Full PDF)
docs = load_pdf("Enterprise_AI_Knowledge_Base_200_Pages.pdf")
for doc in docs:
    doc.page_content = clean_text(doc.page_content)

# 2. Chunk documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

# 3. Create FAISS Vector Store
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = FAISS.from_documents(documents, embedding)

# 4. Set up contextual compression pipeline
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embedding_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.4)
extractor = LLMChainExtractor.from_llm(llm)
pipeline = DocumentCompressorPipeline(transformers=[embedding_filter, extractor])

compression_retriever = ContextualCompressionRetriever(
    base_retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
    base_compressor=pipeline
)

# 5. Retrieve compressed context
compressed_docs = compression_retriever.invoke("What role does metadata play in RAG?")

for doc in compressed_docs:
    print(f"Content: {doc.page_content[:300]}...")
    print(f"Metadata: {doc.metadata}")
    print("-" * 50)