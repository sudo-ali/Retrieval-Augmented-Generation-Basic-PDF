from RAG import RAG

rag = RAG('/Users/ali/Desktop/RAG/notes.pdf', './vector_store', 'pdf_notes')
def build_rag():
    rag.load_pdf_text()
    rag.chunk_text()
    rag.build_vector_store()
while True:
    # rag.load_pdf_text()
    # rag.chunk_text()
    # collection, model = rag.build_vector_store()
    question = input("Ask Groq: ")
    if question == "exit":
        break
    
    answer = rag.ask_groq(question)
    print(answer)

