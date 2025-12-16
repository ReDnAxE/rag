"""
Chunking sémantique : découpe basée sur la similarité sémantique.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from config import CHUNK_SIZE


def chunk_text_semantic(text, max_chunk_size=CHUNK_SIZE, threshold=0.5):
    """
    Divise un texte en chunks basés sur la similarité sémantique.

    Cette méthode découpe le texte en phrases, calcule leurs embeddings,
    et regroupe les phrases sémantiquement similaires.

    Args:
        text: Le texte à diviser
        max_chunk_size: Taille maximale d'un chunk en caractères
        threshold: Seuil de similarité (0-1, plus haut = chunks plus petits)

    Returns:
        Liste de chunks de texte
    """
    # Charger le modèle d'embedding (même que celui de ChromaDB)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Découper en phrases
    sentences = _split_into_sentences(text)

    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    # Calculer les embeddings de chaque phrase
    embeddings = model.encode(sentences)

    # Calculer les distances entre phrases consécutives
    distances = []
    for i in range(len(embeddings) - 1):
        # Similarité cosinus
        similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
        )
        # Convertir en distance (1 - similarité)
        distances.append(1 - similarity)

    # Trouver les points de rupture (haute distance = changement de sujet)
    avg_distance = np.mean(distances)
    breakpoints = [0]  # Toujours commencer au début

    for i, distance in enumerate(distances):
        if distance > avg_distance * (1 + threshold):
            breakpoints.append(i + 1)

    breakpoints.append(len(sentences))  # Toujours finir à la fin

    # Créer les chunks
    chunks = []
    for i in range(len(breakpoints) - 1):
        start = breakpoints[i]
        end = breakpoints[i + 1]
        chunk_sentences = sentences[start:end]
        chunk = " ".join(chunk_sentences)

        # Si le chunk est trop grand, le subdiviser
        if len(chunk) > max_chunk_size:
            sub_chunks = _split_large_chunk(chunk_sentences, max_chunk_size)
            chunks.extend(sub_chunks)
        else:
            chunks.append(chunk.strip())

    return [c for c in chunks if c.strip()]


def _split_into_sentences(text):
    """
    Découpe un texte en phrases de manière simple.
    """
    import re

    # Pattern pour détecter les fins de phrases
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ])')

    sentences = sentence_endings.split(text)

    # Nettoyer et filtrer
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def _split_large_chunk(sentences, max_size):
    """
    Divise un groupe de phrases trop long en chunks plus petits.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > max_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length + 1  # +1 pour l'espace

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def prepare_chunks_for_db_semantic(documents, max_chunk_size=CHUNK_SIZE, threshold=0.5):
    """
    Prépare les documents en chunks sémantiques pour insertion dans ChromaDB.

    Args:
        documents: Liste de tuples (nom_fichier, contenu)
        max_chunk_size: Taille maximale d'un chunk
        threshold: Seuil de rupture sémantique

    Returns:
        Tuple (texts, metadatas, ids) prêts pour ChromaDB
    """
    all_texts = []
    all_metadatas = []
    all_ids = []

    chunk_id = 0
    for filename, content in documents:
        chunks = chunk_text_semantic(content, max_chunk_size, threshold)
        print(f"   → {filename}: {len(chunks)} chunks (sémantique)")

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                all_texts.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_method": "semantic"
                })
                all_ids.append(f"doc_{chunk_id}")
                chunk_id += 1

    return all_texts, all_metadatas, all_ids
