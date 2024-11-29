from transformers import AutoTokenizer, AutoModel
import torch
import faiss


def main():
    # Load pre-trained model and tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Function to embed texts using a pre-trained model
    def embed_texts(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():  # Disables gradient calculation to save memory
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Get sentence-level embedding by averaging token embeddings
        return embeddings

    # Example documents
    documents = [
        "The cat sat on the mat.",
        "The dog barked at the moon.",
        "The sun is shining bright."
    ]

    # Generate embeddings for the documents
    doc_embeddings = embed_texts(documents).numpy()

    # Example query
    query = "cat on mat"

    # Generate an embedding for the query
    query_embedding = embed_texts([query]).numpy()

    # Use FAISS to perform the nearest neighbor search
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])  # L2 distance index
    index.add(doc_embeddings)  # Add the document embeddings to the index

    # Search for the top 3 most similar documents
    _, top_indices = index.search(query_embedding, k=3)  # Find top 3 most similar documents
    top_results = [documents[idx] for idx in top_indices[0]]

    print("Top 3 documents based on dense retrieval:", top_results)


if __name__ == "__main__":
    main()
