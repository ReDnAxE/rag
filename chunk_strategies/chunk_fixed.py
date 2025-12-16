"""
Chunking par taille fixe en caractères.
Méthode basique conservée pour compatibilité.
"""

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_RATIO


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Divise un texte en chunks de taille fixe avec chevauchement.

    Args:
        text: Le texte à diviser
        chunk_size: Taille de chaque chunk en caractères
        overlap: Nombre de caractères de chevauchement entre chunks

    Returns:
        Liste de chunks de texte
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        # Ne pas couper au milieu d'un mot si possible
        if end < text_length and not text[end].isspace():
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * MIN_CHUNK_RATIO:
                end = start + last_space
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def prepare_chunks_for_db(documents):
    """
    Prépare les documents en chunks pour insertion dans ChromaDB.

    Args:
        documents: Liste de tuples (nom_fichier, contenu)

    Returns:
        Tuple (texts, metadatas, ids) prêts pour ChromaDB
    """
    all_texts = []
    all_metadatas = []
    all_ids = []

    chunk_id = 0
    for filename, content in documents:
        chunks = chunk_text(content)
        print(f"   → {filename}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append({
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            all_ids.append(f"doc_{chunk_id}")
            chunk_id += 1

    return all_texts, all_metadatas, all_ids
