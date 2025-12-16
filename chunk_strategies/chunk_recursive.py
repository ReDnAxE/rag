"""
Chunking récursif : découpe intelligente par paragraphes, phrases, puis mots.
"""

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text_recursive(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Divise un texte en chunks en respectant la structure (paragraphes, phrases).

    Args:
        text: Le texte à diviser
        chunk_size: Taille maximale de chaque chunk en caractères
        overlap: Nombre de caractères de chevauchement entre chunks

    Returns:
        Liste de chunks de texte
    """
    # Séparateurs par ordre de préférence (du plus structuré au moins structuré)
    separators = [
        "\n\n",  # Paragraphes
        "\n",    # Sauts de ligne
        ". ",    # Phrases (avec point et espace)
        "! ",    # Phrases exclamatives
        "? ",    # Phrases interrogatives
        "; ",    # Points-virgules
        ", ",    # Virgules
        " ",     # Espaces (mots)
        ""       # Caractères (dernier recours)
    ]

    return _split_text_recursive(text, chunk_size, overlap, separators)


def _split_text_recursive(text, chunk_size, overlap, separators):
    """
    Fonction récursive pour découper le texte.
    """
    chunks = []

    if not text.strip():
        return chunks

    # Si le texte est assez petit, le retourner tel quel
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Essayer chaque séparateur dans l'ordre
    for separator in separators:
        if separator == "":
            # Dernier recours : découpage caractère par caractère
            return _chunk_by_size(text, chunk_size, overlap)

        if separator in text:
            # Découper le texte selon ce séparateur
            splits = text.split(separator)

            # Reconstituer les chunks
            current_chunk = []
            current_length = 0

            for i, split in enumerate(splits):
                # Ajouter le séparateur sauf pour le dernier élément
                piece = split + (separator if i < len(splits) - 1 else "")
                piece_length = len(piece)

                # Si un seul morceau dépasse la taille max, le découper avec le séparateur suivant
                if piece_length > chunk_size:
                    # Sauvegarder le chunk actuel s'il existe
                    if current_chunk:
                        chunks.append("".join(current_chunk).strip())
                        current_chunk = []
                        current_length = 0

                    # Découper récursivement ce gros morceau
                    sub_chunks = _split_text_recursive(
                        piece,
                        chunk_size,
                        overlap,
                        separators[separators.index(separator) + 1:]
                    )
                    chunks.extend(sub_chunks)
                    continue

                # Si ajouter ce morceau dépasse la taille, finaliser le chunk actuel
                if current_length + piece_length > chunk_size and current_chunk:
                    chunks.append("".join(current_chunk).strip())

                    # Gérer l'overlap : garder les derniers morceaux
                    overlap_text = "".join(current_chunk)[-overlap:] if overlap > 0 else ""
                    current_chunk = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text)

                # Ajouter le morceau au chunk actuel
                current_chunk.append(piece)
                current_length += piece_length

            # Ajouter le dernier chunk s'il existe
            if current_chunk:
                chunk_text = "".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)

            return chunks

    return chunks


def _chunk_by_size(text, chunk_size, overlap):
    """
    Découpage basique par taille (fallback).
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]


# Pour compatibilité avec l'ancien code
def prepare_chunks_for_db(documents, chunk_method="recursive"):
    """
    Prépare les documents en chunks pour insertion dans ChromaDB.

    Args:
        documents: Liste de tuples (nom_fichier, contenu)
        chunk_method: "recursive" ou "fixed" (ancien système)

    Returns:
        Tuple (texts, metadatas, ids) prêts pour ChromaDB
    """
    from config import CHUNK_SIZE, CHUNK_OVERLAP

    all_texts = []
    all_metadatas = []
    all_ids = []

    chunk_id = 0
    for filename, content in documents:
        if chunk_method == "recursive":
            chunks = chunk_text_recursive(content, CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            from .chunk_fixed import chunk_text
            chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)

        print(f"   → {filename}: {len(chunks)} chunks ({chunk_method})")

        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Ne pas ajouter de chunks vides
                all_texts.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_method": chunk_method
                })
                all_ids.append(f"doc_{chunk_id}")
                chunk_id += 1

    return all_texts, all_metadatas, all_ids
