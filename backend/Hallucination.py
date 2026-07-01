from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from load_pdf import load_pdf
from clean_text import clean_text

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Load original PDF and extract a real paragraph as context
docs = load_pdf("Enterprise_AI_Knowledge_Base_200_Pages.pdf")
context = clean_text(docs[0].page_content)[:1000]

supported_answer = "The organization builds enterprise AI solutions for customer support, healthcare, finance, education, legal analytics, and automation."
hallucinated_answer = "The organization was founded in Paris in 2018 and has 500 employees."

def verify(answer):
    prompt = f"Context:\n{context}\n\nAnswer:\n{answer}\n\nCheck if the answer is supported by context. Reply only SUPPORTED or NOT_SUPPORTED."
    return llm.invoke(prompt).content.strip()

print(f"Supported Answer: {verify(supported_answer)}")
print(f"Hallucinated Answer: {verify(hallucinated_answer)}")