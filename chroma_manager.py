"""
Gestionnaire pour les opérations ChromaDB.
"""

# Workaround pour SQLite version < 3.35
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Désactiver la télémétrie ChromaDB (incompatible avec Python 3.8)
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import chromadb
from chromadb.utils import embedding_functions
from config import BATCH_SIZE, EMBEDDING_MODEL_NAME


class ChromaDBManager:
    """Classe pour gérer les opérations sur ChromaDB."""

    def __init__(self, db_path, collection_name, description=""):
        """
        Initialise le gestionnaire ChromaDB.

        Args:
            db_path: Chemin vers la base de données ChromaDB
            collection_name: Nom de la collection
            description: Description de la collection
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.description = description
        self.client = None
        self.collection = None

        # Modèle d'embedding : all-MiniLM-L6-v2 (Sentence Transformers)
        # Dimensions : 384, optimal pour recherche sémantique multilingue
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )

    def connect(self):
        """Crée la connexion au client ChromaDB."""
        self.client = chromadb.PersistentClient(path=self.db_path)
        print(f"✓ Connecté à ChromaDB: {self.db_path}")

    def create_collection(self, reset=True):
        """
        Crée une nouvelle collection.

        Args:
            reset: Si True, supprime la collection existante
        """
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"   → Collection existante '{self.collection_name}' supprimée")
            except:
                pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": self.description},
            embedding_function=self.embedding_function
        )
        print(f"✓ Collection '{self.collection_name}' créée")
        print(f"✓ Modèle d'embedding: all-MiniLM-L6-v2 (384 dimensions)")

    def insert_documents(self, texts, metadatas, ids, batch_size=BATCH_SIZE):
        """
        Insère les documents dans la collection par lots.

        Args:
            texts: Liste des textes
            metadatas: Liste des métadonnées
            ids: Liste des IDs
            batch_size: Taille des lots
        """
        total_docs = len(texts)
        print(f"\nInsertion de {total_docs} documents...")

        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            self.collection.add(
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            print(f"   → Batch {i//batch_size + 1}: {end_idx - i} documents insérés")

        print(f"✓ {total_docs} documents insérés avec succès")

    def get_stats(self):
        """
        Retourne les statistiques de la collection.

        Returns:
            Dictionnaire avec les statistiques
        """
        if not self.collection:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )

        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "db_path": self.db_path
        }

    def query(self, query_text, n_results=3):
        """
        Effectue une requête sur la collection.

        Args:
            query_text: Texte de la requête
            n_results: Nombre de résultats à retourner

        Returns:
            Résultats de la requête
        """
        if not self.collection:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def close(self):
        """Ferme la connexion (cleanup si nécessaire)."""
        self.client = None
        self.collection = None
