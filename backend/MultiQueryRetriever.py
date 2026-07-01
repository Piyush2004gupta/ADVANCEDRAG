from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from load_pdf import load_pdf
from clean_text import clean_text
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Load and clean original PDF documents (first 20 pages)
docs = load_pdf("Enterprise_AI_Knowledge_Base_200_Pages.pdf")
cleaned_docs = []
for doc in docs[:20]:
    doc.page_content = clean_text(doc.page_content)
    cleaned_docs.append(doc)

# Split original documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(cleaned_docs)

# Create dense embeddings and FAISS vector store
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = FAISS.from_documents(documents, embedding)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_db.as_retriever(),
    llm=llm
)

docs = retriever.invoke(
    "RAG Ingestion and chunking architecture"
)

for doc in docs:
    print(doc.page_content[:300] + "...")
    print("-" * 30)