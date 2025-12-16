"""
Module unifié pour le chunking avec support de multiples méthodes.
"""


def chunk_text(text, strategy, chunk_size, chunk_overlap, chunk_method):
    """
    Divise un texte en chunks selon la méthode spécifiée.

    Args:
        text: Le texte à diviser
        strategy: Stratégie de chunking ("llamaindex")
        chunk_size: Taille des chunks
                   Note: Pour "token", ceci sera converti en nombre de tokens
        overlap: Chevauchement
        chunk_method: méthode de chunk spécifique à la stratégie

    Returns:
        Liste de chunks de texte
    """

    # Appeler la méthode appropriée
    if strategy == "llamaindex":
        try:
            from .chunk_llamaindex import chunk_text_llamaindex
            return chunk_text_llamaindex(text, chunk_method, chunk_size, chunk_overlap)
        except ImportError:
            print("⚠️  LlamaIndex non disponible. Installation requise :")
            print("   pip install llama-index-core")
            exit(1)

    else:
        raise ValueError(f"Méthode de chunking inconnue : {strategy}. "
                        f"Options : 'llamaindex'")


def prepare_chunks_for_db(documents, strategy, chunk_size, chunk_overlap, chunk_method):
    """
    Prépare les documents en chunks pour insertion dans ChromaDB.

    Args:
        documents: Liste de tuples (nom_fichier, contenu)
        strategy: Méthode de chunking

    Returns:
        Tuple (texts, metadatas, ids) prêts pour ChromaDB
    """

    all_texts = []
    all_metadatas = []
    all_ids = []

    chunk_id = 0
    for filename, content in documents:
        chunks = chunk_text(content, strategy, chunk_size, chunk_overlap, chunk_method)
        print(f"   → {filename}: {len(chunks)} chunks (méthode: {strategy})")

        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Ne pas ajouter de chunks vides
                all_texts.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_strategy": strategy
                })
                all_ids.append(f"doc_{chunk_id}")
                chunk_id += 1

    return all_texts, all_metadatas, all_ids
