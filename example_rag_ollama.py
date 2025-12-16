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
from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME


class RAGSystem:
    """Système RAG simple utilisant ChromaDB et Ollama."""

    def __init__(self, chroma_path, collection_name, ollama_model):
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
            model_name=EMBEDDING_MODEL_NAME
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

    def _build_context(self, context_docs):
        """
        Construit le contexte à partir des documents.

        Args:
            context_docs: Documents de contexte

        Returns:
            Contexte formaté
        """
        return "\n\n".join([
            f"[Source: {doc['source']}]\n{doc['text']}"
            for doc in context_docs
        ])

    def _build_prompt(self, query, context):
        """
        Construit le prompt pour Ollama.

        Args:
            query: Question de l'utilisateur
            context: Contexte formaté

        Returns:
            Prompt complet
        """
        return f"""Contexte pertinent:
{context}

Question: {query}

Instructions: Réponds à la question en te basant sur le contexte fourni ci-dessus.
Si la réponse n'est pas dans le contexte, dis-le clairement.
Cite les sources quand tu utilises des informations du contexte."""

    def _stream_response(self, response):
        """
        Lit et affiche le stream de réponse d'Ollama.

        Args:
            response: Objet Response de requests avec streaming

        Returns:
            Réponse complète
        """
        full_response = ""
        print("=" * 60)
        print("RÉPONSE:")
        print("=" * 60)

        # Lire le stream ligne par ligne
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        token = chunk['response']
                        print(token, end='', flush=True)
                        full_response += token
                except json.JSONDecodeError:
                    continue

        print("\n" + "=" * 60 + "\n")
        return full_response

    def generate_response(self, query, context_docs):
        """
        Génère une réponse avec Ollama en utilisant le contexte.

        Args:
            query: Question de l'utilisateur
            context_docs: Documents de contexte

        Returns:
            Réponse générée
        """
        # Construire le contexte et le prompt
        context = self._build_context(context_docs)
        prompt = self._build_prompt(query, context)

        # Appeler Ollama avec streaming
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": True
                },
                timeout=180,  # 3 minutes pour les modèles lourds
                stream=True  # Important pour recevoir le stream
            )

            if response.status_code == 200:
                return self._stream_response(response)
            else:
                return f"Erreur Ollama: {response.status_code}"

        except requests.exceptions.ConnectionError:
            return "Erreur: Impossible de se connecter à Ollama. Assurez-vous qu'Ollama est lancé."
        except Exception as e:
            return f"Erreur: {e}"

    def _print_question(self, question):
        """Affiche la question."""
        print(f"Question: {question}\n")
        print("Recherche des documents pertinents...\n")

    def _print_documents(self, docs):
        """Affiche les documents trouvés."""
        print(f"✓ {len(docs)} documents trouvés:\n")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc['source']}")
            print(f"     {doc['text'][:100]}...\n")

    def ask(self, question):
        """
        Pose une question au système RAG.

        Args:
            question: Question de l'utilisateur

        Returns:
            Réponse générée avec le contexte
        """
        self._print_question(question)

        # Rechercher les documents
        docs = self.search_documents(question)
        self._print_documents(docs)

        print("Génération de la réponse avec Ollama...\n")

        # Générer la réponse (affiche automatiquement le stream)
        return self.generate_response(question, docs)


def _print_header():
    """Affiche l'en-tête du programme."""
    print("=" * 60)
    print("Système RAG avec ChromaDB et Ollama")
    print("=" * 60 + "\n")


def _initialize_rag_system():
    """
    Initialise le système RAG.

    Returns:
        Instance de RAGSystem ou None en cas d'erreur
    """
    try:
        return RAGSystem(
            chroma_path=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            ollama_model=LLM_MODEL_NAME  # Modèle personnalisé créé avec le Modelfile
        )
    except Exception as e:
        print(f"✗ Erreur lors de l'initialisation: {e}")
        print("\nAssurez-vous d'avoir:")
        print("1. Exécuté main.py pour créer la base ChromaDB")
        print("2. Installé Ollama")
        print("3. Créé le modèle : ollama create rag-assistant -f Modelfile")
        return None


def _run_demo_questions(rag):
    """
    Exécute les questions de démonstration.

    Args:
        rag: Instance du système RAG
    """
    questions = [
        "Qu'est-ce que le RAG et comment ça fonctionne?",
        "Quels sont les avantages de Python?",
        "Comment utiliser Ollama avec un Modelfile?",
    ]

    for question in questions:
        rag.ask(question)
        print("\n" + "-" * 60 + "\n")


def _run_interactive_mode(rag):
    """
    Lance le mode interactif.

    Args:
        rag: Instance du système RAG
    """
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


def main():
    """Fonction principale de démonstration."""
    _print_header()

    # Initialiser le système RAG
    rag = _initialize_rag_system()
    if not rag:
        return

    # Questions de démonstration
    _run_demo_questions(rag)

    # Mode interactif
    _run_interactive_mode(rag)


if __name__ == "__main__":
    main()
