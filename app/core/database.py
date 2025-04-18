import json
import sqlite3
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DatabaseManager:
    """Manages storage and retrieval of image generations with embedding-based similarity search.

    This class combines SQLite for persistent storage with FAISS for efficient similarity
    search of text embeddings. It uses SentenceTransformer to generate embeddings for text prompts.

    Attributes:
        encoder (SentenceTransformer): Model for generating text embeddings
        index (faiss.IndexFlatL2): FAISS index for similarity search
        db_path (str): Path to SQLite database file
    """

    def __init__(
        self, db_path: str = "memory.db", encoder_model: str = "all-MiniLM-L6-v2"
    ) -> None:
        """Initialize the DatabaseManager with specified database and encoder model.

        Args:
            db_path (str, optional): Path to SQLite database file. Defaults to "memory.db".
            encoder_model (str, optional): Name of the sentence transformer model.
                Defaults to "all-MiniLM-L6-v2".
        """
        # Initialize FAISS index
        self.encoder = SentenceTransformer(encoder_model)
        self.index = faiss.IndexFlatL2(384)  # 384 is the embedding dimension for MiniLM

        # Setup SQLite
        self.db_path = db_path
        self.setup_database()

        # Load existing embeddings
        self.load_existing_embeddings()

    def setup_database(self) -> None:
        """Initialize SQLite database with required schema.

        Creates a 'generations' table if it doesn't exist with columns for storing
        generation details including prompts, embeddings, and metadata.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    original_prompt TEXT,
                    expanded_prompt TEXT,
                    embedding BLOB,
                    metadata JSON
                )
            """)

    def load_existing_embeddings(self) -> None:
        """Load existing embeddings from database into FAISS index.

        Retrieves all stored embeddings from the SQLite database and adds them to
        the FAISS index for similarity search functionality.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, original_prompt, embedding FROM generations"
            )
            for row in cursor:
                if row[2]:  # if embedding exists
                    embedding = np.frombuffer(row[2], dtype=np.float32)
                    self.index.add(embedding.reshape(1, -1))

    def save_generation(
        self,
        original_prompt: str,
        expanded_prompt: str,
        metadata: Dict = None,
    ) -> int:
        """Save a new generation to the database and update FAISS index.

        Args:
            original_prompt (str): The original user input prompt
            expanded_prompt (str): The expanded/processed prompt used for generation
            metadata (Dict, optional): Additional metadata to store. Defaults to None.

        Returns:
            int: ID of the newly inserted generation record
        """
        embedding = self.encoder.encode([original_prompt])[0]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO generations 
                (original_prompt, expanded_prompt, embedding, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (
                    original_prompt,
                    expanded_prompt,
                    embedding.tobytes(),
                    json.dumps(metadata or {}),
                ),
            )

            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1))

            return cursor.lastrowid

    def find_similar_generations(self, prompt: str, k: int = 5) -> List[Dict]:
        """Search for similar text generations based on prompt embedding similarity.

        This function finds the k most similar previous generations by:
        1. Converting the input prompt to an embedding vector
        2. Using FAISS similarity search to find nearest neighbor embeddings
        3. Retrieving the full generation records from SQLite database

        Args:
            prompt (str): The input text prompt to find similar generations for
            k (int, optional): Number of similar generations to return. Defaults to 5.

        Returns:
            List[Dict]: List of k most similar generation records, each containing:
                - prompt: Original prompt text
                - completion: Generated completion text
                - metadata: Any additional metadata stored with the generation
                etc.

        Note:
            The similarity search uses cosine similarity between prompt embeddings.
            FAISS indices are 0-based but SQLite rowids are 1-based, hence the +1 adjustment.
        """
        embedding = self.encoder.encode([prompt])

        # Search FAISS index
        _distances, indices = self.index.search(embedding, k)

        # Get full records from SQLite
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            results = []
            for idx in indices[0]:
                row = conn.execute(
                    "SELECT * FROM generations WHERE rowid = ?",
                    (int(idx) + 1,),  # FAISS indices are 0-based
                ).fetchone()
                if row:
                    results.append(dict(row))
            return results
