"""
Chunking avec LangChain : solution production-ready avec plusieurs stratégies.

Installation requise :
    pip install langchain langchain-text-splitters
"""

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text_langchain(text, method="recursive", chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Divise un texte en chunks en utilisant LangChain TextSplitters.

    Args:
        text: Le texte à diviser
        method: Méthode de chunking
            - "recursive": RecursiveCharacterTextSplitter (recommandé)
            - "sentence": SentenceSplitter
            - "token": TokenTextSplitter
        chunk_size: Taille de chaque chunk
        overlap: Chevauchement entre chunks

    Returns:
        Liste de chunks de texte
    """
    try:
        if method == "recursive":
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len,
                separators=[
                    "\n\n",  # Paragraphes
                    "\n",    # Lignes
                    ". ",    # Phrases
                    "! ",
                    "? ",
                    "; ",
                    ", ",
                    " ",     # Mots
                    ""       # Caractères
                ],
                is_separator_regex=False,
            )

        elif method == "sentence":
            from langchain_text_splitters import SentenceTransformersTokenTextSplitter

            # Utilise le même modèle que ChromaDB
            splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=overlap,
                model_name="all-MiniLM-L6-v2",
                tokens_per_chunk=chunk_size // 4  # Approximation : ~4 chars par token
            )

        elif method == "token":
            from langchain_text_splitters import CharacterTextSplitter

            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separator="\n",
                length_function=len,
            )

        else:
            raise ValueError(f"Méthode inconnue : {method}")

        chunks = splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]

    except ImportError:
        print("⚠️  LangChain non installé. Utilisez : pip install langchain langchain-text-splitters")
        print("   Utilisation du chunking basique par défaut...")

        # Fallback sur le système basique
        from .chunk_fixed import chunk_text
        return chunk_text(text, chunk_size, overlap)


def prepare_chunks_for_db_langchain(documents, method="recursive", chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Prépare les documents en chunks LangChain pour insertion dans ChromaDB.

    Args:
        documents: Liste de tuples (nom_fichier, contenu)
        method: Méthode de chunking (recursive, sentence, token)
        chunk_size: Taille des chunks
        overlap: Chevauchement

    Returns:
        Tuple (texts, metadatas, ids) prêts pour ChromaDB
    """
    all_texts = []
    all_metadatas = []
    all_ids = []

    chunk_id = 0
    for filename, content in documents:
        chunks = chunk_text_langchain(content, method, chunk_size, overlap)
        print(f"   → {filename}: {len(chunks)} chunks (langchain-{method})")

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                all_texts.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_method": f"langchain-{method}"
                })
                all_ids.append(f"doc_{chunk_id}")
                chunk_id += 1

    return all_texts, all_metadatas, all_ids
