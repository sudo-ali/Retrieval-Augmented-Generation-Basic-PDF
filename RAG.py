from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import os


PDF_PATH = "/Users/ali/Desktop/RAG/notes.pdf"
CHROMA_PATH = "./vector_store"
COLLECTION_NAME = "pdf_notes"
GROQ_MODEL = "llama-3.1-8b-instant"


class RAG():
    def __init__(self, pdf_path, chroma_path, collection_name):
        self.pdf_path = pdf_path
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.text = ''
        self.chunks = []

# 1. Extract text from PDF
    def load_pdf_text(self):
        reader = PdfReader(self.pdf_path)
        self.text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                self.text += page_text + "\n"

        return self.text


    # 2. Split text into chunks
    def chunk_text(self, chunk_size=500, overlap=100):
        words = self.text.split()
        self.chunks = []

        step = chunk_size - overlap

        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            self.chunks.append(chunk)

        return self.chunks


    # 3. Create vector store
    def build_vector_store(self):
        client = chromadb.PersistentClient(self.chroma_path)
        collection = client.get_or_create_collection(self.collection_name)

        model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        embeddings = model.encode(self.chunks, normalize_embeddings=True).tolist()

        ids = [f"chunk_{i}" for i in range(len(self.chunks))]

        collection.add(
            ids=ids,
            documents=self.chunks,
            embeddings=embeddings,
            metadatas=[{"source": self.pdf_path, "chunk": i} for i in range(len(self.chunks))]
        )

        print(f"Stored {len(self.chunks)} chunks.")

        return collection, model


    # 4. Search relevant chunks
    def retrieve(self, query, top_k=3):
        client = chromadb.PersistentClient(self.chroma_path)
        collection = client.get_or_create_collection(self.collection_name)
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results["documents"][0]


    # 5. Ask LLM using retrieved context
    def ask_gemma(self, query, collection, model):
        context = self.retrieve(query, collection, model)

        prompt = f"""
    Use the context below to answer the question.
    If the context does not contain the answer, say you do not know.

    Context:
    {context}

    Question:
    {query}
    """

        r = requests.post(
            f"{http://localhost:11434/api}/chat",
            json={
                "model": "gemma4:e2b",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    def ask_groq(self, query):
        context = self.retrieve(query)
        groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        prompt = f"""
    You are a helpful RAG assistant.

    Answer the user's question using ONLY the context below.
    If the answer is not in the context, say: "I don't know based on the PDF."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=700
        )

        return response.choices[0].message.content


# rag = RAG(PDF_PATH, CHROMA_PATH, COLLECTION_NAME)

# rag.load_pdf_text()
# rag.chunk_text()

# collection, model = rag.build_vector_store()

# while True:
#     query = input("\nAsk a question about the PDF: ")

#     if query.lower() in ["exit", "quit"]:
#         break

#     relevant_chunks = rag.retrieve(query, collection, model)
#     answer = rag.ask_llm(query, relevant_chunks)

#     print("\nAnswer:")
    # print(answer)