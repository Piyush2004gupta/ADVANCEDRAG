from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


if __name__ == "__main__":
    docs = load_pdf("Enterprise_AI_Knowledge_Base_200_Pages.pdf")

    print(f"Total Pages: {len(docs)}")
    print(docs[0].page_content[:300])