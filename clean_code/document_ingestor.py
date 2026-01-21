import hashlib
from sentence_transformers import SentenceTransformer
from typing import Tuple
from document_processor import DocumentProcessor

class DocumentIngester:
    """Ingest documents into vector database."""

    def __init__(self, embedder: SentenceTransformer, collection):
        self.embedder = embedder
        self.collection = collection

    @staticmethod
    def get_file_hash(file) -> str:
        """Generate file hash."""
        file.seek(0)
        h = hashlib.md5(file.read()).hexdigest()
        file.seek(0)
        return h

    def is_ingested(self, file_hash: str) -> bool:
        """Check if file already ingested."""
        try:
            results = self.collection.get(where={"file_hash": file_hash}, limit=1)
            return len(results["ids"]) > 0
        except:
            return False

    def ingest(self, file) -> Tuple[int, str]:
        """Ingest a file into the database."""
        file_hash = self.get_file_hash(file)

        if self.is_ingested(file_hash):
            return 0, "exists"

        chunks = DocumentProcessor.process_document(file)
        if not chunks:
            return 0, "no content"

        # Prepare for embedding
        texts_to_embed = []
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_hash}_{i}"

            # Format for E5 model
            embed_text = f"passage: {chunk['child_text']}"
            texts_to_embed.append(embed_text)

            ids.append(chunk_id)
            documents.append(chunk["child_text"])
            metadatas.append(
                {
                    **chunk["metadata"],
                    "file_hash": file_hash,
                    "parent_text": chunk["parent_text"],
                }
            )

        # Generate embeddings
        embeddings = self.embedder.encode(
            texts_to_embed, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks), "success"