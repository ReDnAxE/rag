#!/usr/bin/env python3
"""
Script principal pour créer une base de données ChromaDB à partir de documents texte.
Cette base pourra ensuite être utilisée avec Ollama pour du RAG.
"""

import os
import sys
from document_loader import load_documents, get_documents_summary
from text_utils import prepare_chunks_for_db
from chroma_manager import ChromaDBManager
from config import (
    DOCUMENTS_DIR,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    COLLECTION_DESCRIPTION
)


def print_header(title):
    """Affiche un en-tête formaté."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def create_database():
    """Fonction principale pour créer la base de données."""
    print_header("Création de la base de données ChromaDB")

    # 1. Charger les documents
    print(f"\n1. Chargement des documents depuis '{DOCUMENTS_DIR}'...")
    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        print("\n✗ Aucun document trouvé. Arrêt du script.")
        return False

    summary = get_documents_summary(documents)
    print(f"\n   → {summary['count']} document(s) chargé(s)")
    print(f"   → {summary['total_chars']:,} caractères au total")

    # 2. Découper les documents en chunks
    print("\n2. Découpage des documents en chunks...")
    all_texts, all_metadatas, all_ids = prepare_chunks_for_db(documents)
    print(f"\n   → Total: {len(all_texts)} chunks à insérer")

    # 3. Créer la base ChromaDB
    print(f"\n3. Création de la base ChromaDB...")
    db_manager = ChromaDBManager(
        db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        description=COLLECTION_DESCRIPTION
    )

    try:
        db_manager.connect()
        db_manager.create_collection(reset=True)

        # 4. Insérer les documents
        print("\n4. Insertion des documents dans ChromaDB...")
        db_manager.insert_documents(all_texts, all_metadatas, all_ids)

        # 5. Vérification
        print("\n5. Vérification...")
        stats = db_manager.get_stats()
        print(f"   → Collection: {stats['collection_name']}")
        print(f"   → Nombre de chunks: {stats['total_documents']}")

        print_header("✓ Base de données créée avec succès!")
        print(f"\nChemin de la base: {os.path.abspath(CHROMA_DB_PATH)}")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Chunks stockés: {stats['total_documents']}")

        return True

    except Exception as e:
        print(f"\n✗ Erreur lors de la création de la base: {e}")
        return False
    finally:
        db_manager.close()


def test_query():
    """Fonction pour tester une requête sur la base."""
    print_header("Test de requête sur la base ChromaDB")

    query = "Qu'est-ce que le RAG?"
    print(f"\nRequête: '{query}'")
    print("\nRésultats les plus pertinents:\n")

    db_manager = ChromaDBManager(
        db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    try:
        db_manager.connect()
        results = db_manager.query(query, n_results=3)

        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"{i+1}. Source: {metadata['source']}")
            print(f"   Distance: {distance:.4f}")
            print(f"   Extrait: {doc[:200]}...")
            print()

    except Exception as e:
        print(f"✗ Erreur lors de la requête: {e}")
    finally:
        db_manager.close()


def print_usage_info():
    """Affiche les informations d'utilisation avec Ollama."""
    print("\n" + "=" * 60)
    print("Utilisation avec Ollama")
    print("=" * 60)
    print("\nPour utiliser cette base avec Ollama dans un système RAG:")
    print("\n1. Dans votre code Python, chargez la base:")
    print("   import chromadb")
    print(f"   client = chromadb.PersistentClient(path='{CHROMA_DB_PATH}')")
    print(f"   collection = client.get_collection('{COLLECTION_NAME}')")
    print("\n2. Recherchez les documents pertinents:")
    print("   results = collection.query(query_texts=['votre question'], n_results=3)")
    print("\n3. Utilisez les résultats comme contexte pour Ollama:")
    print("   context = '\\n'.join(results['documents'][0])")
    print("   prompt = f'Contexte: {context}\\n\\nQuestion: {votre_question}'")
    print("\n4. Envoyez le prompt à Ollama via son API")
    print()


def main():
    """Point d'entrée principal."""
    try:
        # Créer la base de données
        success = create_database()

        if not success:
            sys.exit(1)

        # Afficher les informations d'utilisation
        print_usage_info()

        # Proposer un test
        print("Voulez-vous tester une requête? (O/n): ", end="", flush=True)
        try:
            response = input().strip().lower()
            if response != 'n':
                test_query()
        except (EOFError, KeyboardInterrupt):
            print("\nTest ignoré")

    except KeyboardInterrupt:
        print("\n\n✗ Opération annulée par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Erreur inattendue: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
