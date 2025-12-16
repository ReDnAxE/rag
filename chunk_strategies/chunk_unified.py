"""
Module unifié pour le chunking avec support de multiples méthodes.
"""

from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_METHOD, SEMANTIC_THRESHOLD


def chunk_text(text, method=None, chunk_size=None, overlap=None):
    """
    Divise un texte en chunks selon la méthode spécifiée.

    Args:
        text: Le texte à diviser
        method: Méthode de chunking ("fixed", "recursive", "token", "semantic", "langchain")
                Si None, utilise CHUNK_METHOD depuis config.py
        chunk_size: Taille des chunks (si None, utilise CHUNK_SIZE)
                   Note: Pour "token", ceci sera converti en nombre de tokens
        overlap: Chevauchement (si None, utilise CHUNK_OVERLAP)

    Returns:
        Liste de chunks de texte
    """
    # Utiliser les valeurs par défaut si non spécifiées
    if method is None:
        method = CHUNK_METHOD
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    # Appeler la méthode appropriée
    if method == "fixed":
        from .chunk_fixed import chunk_text as chunk_fixed
        return chunk_fixed(text, chunk_size, overlap)

    elif method == "recursive":
        from .chunk_recursive import chunk_text_recursive
        return chunk_text_recursive(text, chunk_size, overlap)

    elif method == "token":
        try:
            from .chunk_token import chunk_text_by_tokens
            # Convertir caractères en tokens (approximation : 1 token ≈ 4 caractères)
            chunk_size_tokens = max(chunk_size // 4, 50)
            overlap_tokens = max(overlap // 4, 10)
            return chunk_text_by_tokens(text, chunk_size_tokens, overlap_tokens)
        except ImportError:
            print("⚠️  Chunking par tokens non disponible. Installation requise :")
            print("   pip install transformers")
            print("   Utilisation de la méthode récursive à la place...")
            from .chunk_recursive import chunk_text_recursive
            return chunk_text_recursive(text, chunk_size, overlap)

    elif method == "semantic":
        try:
            from .chunk_semantic import chunk_text_semantic
            return chunk_text_semantic(text, chunk_size, SEMANTIC_THRESHOLD)
        except ImportError:
            print("⚠️  Chunking sémantique non disponible. Installation requise :")
            print("   pip install sentence-transformers")
            print("   Utilisation de la méthode récursive à la place...")
            from .chunk_recursive import chunk_text_recursive
            return chunk_text_recursive(text, chunk_size, overlap)

    elif method == "langchain":
        try:
            from .chunk_langchain import chunk_text_langchain
            return chunk_text_langchain(text, "recursive", chunk_size, overlap)
        except ImportError:
            print("⚠️  LangChain non disponible. Installation requise :")
            print("   pip install langchain langchain-text-splitters")
            print("   Utilisation de la méthode récursive à la place...")
            from .chunk_recursive import chunk_text_recursive
            return chunk_text_recursive(text, chunk_size, overlap)

    else:
        raise ValueError(f"Méthode de chunking inconnue : {method}. "
                        f"Options : 'fixed', 'recursive', 'token', 'semantic', 'langchain'")


def prepare_chunks_for_db(documents, method=None):
    """
    Prépare les documents en chunks pour insertion dans ChromaDB.

    Args:
        documents: Liste de tuples (nom_fichier, contenu)
        method: Méthode de chunking (si None, utilise CHUNK_METHOD)

    Returns:
        Tuple (texts, metadatas, ids) prêts pour ChromaDB
    """
    if method is None:
        method = CHUNK_METHOD

    all_texts = []
    all_metadatas = []
    all_ids = []

    chunk_id = 0
    for filename, content in documents:
        chunks = chunk_text(content, method=method)
        print(f"   → {filename}: {len(chunks)} chunks (méthode: {method})")

        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Ne pas ajouter de chunks vides
                all_texts.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_method": method
                })
                all_ids.append(f"doc_{chunk_id}")
                chunk_id += 1

    return all_texts, all_metadatas, all_ids
