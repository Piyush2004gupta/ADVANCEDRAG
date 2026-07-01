from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
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
chunks = text_splitter.split_documents(docs)

# 3. Store chunks in Qdrant (In-Memory)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="rag_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vector_db = QdrantVectorStore(client=client, collection_name="rag_collection", embedding=embedding)
vector_db.add_documents(chunks)
print("Stored chunks in Qdrant DB successfully.")

# 4. Search verification
results = vector_db.similarity_search("customer support solutions", k=1)
for doc in results:
    print(f"Retrieved: {doc.page_content[:300]}...")