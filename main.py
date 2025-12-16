#!/usr/bin/env python3
"""
Script principal pour créer une base de données ChromaDB à partir de documents texte.
Cette base pourra ensuite être utilisée avec Ollama pour du RAG.
"""

import os
import sys
from document_loader import load_documents, get_documents_summary
from chunk_strategies import prepare_chunks_for_db
from chroma_manager import ChromaDBManager
from config import (
    DOCUMENTS_DIR,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    COLLECTION_DESCRIPTION,
    CHUNK_STRATEGY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_METHOD
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
    print(f"\n2. Découpage des documents en chunks (stratégie: {CHUNK_STRATEGY})...")
    all_texts, all_metadatas, all_ids = prepare_chunks_for_db(documents, CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_METHOD)
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


def main():
    """Point d'entrée principal."""
    try:
        # Créer la base de données
        success = create_database()

        if not success:
            sys.exit(1)



    except KeyboardInterrupt:
        print("\n\n✗ Opération annulée par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Erreur inattendue: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
