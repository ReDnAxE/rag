#!/usr/bin/env python3
"""
Exemple d'utilisation de la base ChromaDB avec Ollama pour du RAG.
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
import requests
import json
from config import CHROMA_DB_PATH, COLLECTION_NAME


class RAGSystem:
    """Système RAG simple utilisant ChromaDB et Ollama."""

    def __init__(self, chroma_path, collection_name, ollama_model="llama3.2"):
        """
        Initialise le système RAG.

        Args:
            chroma_path: Chemin vers la base ChromaDB
            collection_name: Nom de la collection
            ollama_model: Nom du modèle Ollama à utiliser
        """
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"

        # Modèle d'embedding : all-MiniLM-L6-v2 (Sentence Transformers)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Connecter à ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        print(f"✓ Connecté à ChromaDB")
        print(f"✓ Collection: {collection_name}")
        print(f"✓ Modèle d'embedding: all-MiniLM-L6-v2 (384 dimensions)")
        print(f"✓ Modèle Ollama: {ollama_model}\n")

    def search_documents(self, query, n_results=3):
        """
        Recherche les documents pertinents dans ChromaDB.

        Args:
            query: Requête de l'utilisateur
            n_results: Nombre de résultats à retourner

        Returns:
            Liste des documents pertinents
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        documents = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            documents.append({
                'text': doc,
                'source': metadata['source']
            })

        return documents

    def generate_response(self, query, context_docs):
        """
        Génère une réponse avec Ollama en utilisant le contexte.

        Args:
            query: Question de l'utilisateur
            context_docs: Documents de contexte

        Returns:
            Réponse générée
        """
        # Construire le contexte
        context = "\n\n".join([
            f"[Source: {doc['source']}]\n{doc['text']}"
            for doc in context_docs
        ])

        # Construire le prompt
        prompt = f"""Contexte pertinent:
{context}

Question: {query}

Instructions: Réponds à la question en te basant sur le contexte fourni ci-dessus.
Si la réponse n'est pas dans le contexte, dis-le clairement.
Cite les sources quand tu utilises des informations du contexte."""

        # Appeler Ollama
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180  # 3 minutes pour les modèles lourds
            )

            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Erreur Ollama: {response.status_code}"

        except requests.exceptions.ConnectionError:
            return "Erreur: Impossible de se connecter à Ollama. Assurez-vous qu'Ollama est lancé."
        except Exception as e:
            return f"Erreur: {e}"

    def ask(self, question):
        """
        Pose une question au système RAG.

        Args:
            question: Question de l'utilisateur

        Returns:
            Réponse générée avec le contexte
        """
        print(f"Question: {question}\n")
        print("Recherche des documents pertinents...\n")

        # Rechercher les documents
        docs = self.search_documents(question)

        print(f"✓ {len(docs)} documents trouvés:\n")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc['source']}")
            print(f"     {doc['text'][:100]}...\n")

        print("Génération de la réponse avec Ollama...\n")

        # Générer la réponse
        response = self.generate_response(question, docs)

        print("=" * 60)
        print("RÉPONSE:")
        print("=" * 60)
        print(response)
        print("=" * 60 + "\n")

        return response


def main():
    """Fonction principale de démonstration."""
    print("=" * 60)
    print("Système RAG avec ChromaDB et Ollama")
    print("=" * 60 + "\n")

    # Initialiser le système RAG
    try:
        rag = RAGSystem(
            chroma_path=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            ollama_model="rag-assistant"  # Modèle personnalisé créé avec le Modelfile
        )
    except Exception as e:
        print(f"✗ Erreur lors de l'initialisation: {e}")
        print("\nAssurez-vous d'avoir:")
        print("1. Exécuté main.py pour créer la base ChromaDB")
        print("2. Installé Ollama")
        print("3. Créé le modèle : ollama create rag-assistant -f Modelfile")
        return

    # Questions de démonstration
    questions = [
        "Qu'est-ce que le RAG et comment ça fonctionne?",
        "Quels sont les avantages de Python?",
        "Comment utiliser Ollama avec un Modelfile?",
    ]

    for question in questions:
        rag.ask(question)
        print("\n" + "-" * 60 + "\n")

    # Mode interactif
    print("\nMode interactif (tapez 'exit' pour quitter)")
    print("-" * 60 + "\n")

    while True:
        try:
            user_question = input("Votre question: ").strip()

            if user_question.lower() in ['exit', 'quit', 'q']:
                print("\nAu revoir!")
                break

            if not user_question:
                continue

            rag.ask(user_question)
            print()

        except KeyboardInterrupt:
            print("\n\nAu revoir!")
            break
        except Exception as e:
            print(f"\n✗ Erreur: {e}\n")


if __name__ == "__main__":
    main()
