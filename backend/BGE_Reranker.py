from sentence_transformers import CrossEncoder
from load_pdf import load_pdf
from clean_text import clean_text

# 1. Load BGE Re-ranker
model = CrossEncoder("BAAI/bge-reranker-base")

# 2. Load original PDF (Full PDF)
docs = load_pdf("Enterprise_AI_Knowledge_Base_200_Pages.pdf")
documents = [clean_text(doc.page_content) for doc in docs]

query = "What industries does the organization build enterprise AI solutions for?"

# 3. Score and Rerank
pairs = [(query, doc) for doc in documents]
scores = model.predict(pairs)

ranked_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

# Print top K (K = 3)
k = 3
for score, doc in ranked_docs[:k]:
    print(f"Score: {score:.4f} | {doc[:200]}...")