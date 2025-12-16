"""
Chunking par tokens : découpe basée sur les tokens linguistiques.
Utilise le tokenizer du modèle d'embedding (all-MiniLM-L6-v2) pour cohérence.
"""

from transformers import AutoTokenizer
from config import CHUNK_SIZE, CHUNK_OVERLAP


# Tokenizer global (chargé une seule fois)
_tokenizer = None


def get_tokenizer():
    """Retourne le tokenizer (singleton)."""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            raise ImportError(
                f"Erreur lors du chargement du tokenizer: {e}\n"
                "Installation requise: pip install transformers"
            )
    return _tokenizer


def chunk_text_by_tokens(text, chunk_size_tokens=None, overlap_tokens=None):
    """
    Divise un texte en chunks basés sur les tokens.

    Cette méthode découpe le texte en utilisant le tokenizer du modèle d'embedding,
    garantissant que les chunks respectent les limites de tokens du modèle.

    Args:
        text: Le texte à diviser
        chunk_size_tokens: Nombre de tokens par chunk
                          Si None, convertit CHUNK_SIZE caractères en tokens (~4:1)
        overlap_tokens: Nombre de tokens de chevauchement
                       Si None, convertit CHUNK_OVERLAP caractères en tokens

    Returns:
        Liste de chunks de texte
    """
    tokenizer = get_tokenizer()

    # Calculer les tailles en tokens si non spécifiées
    if chunk_size_tokens is None:
        # Approximation : 1 token ≈ 4 caractères
        chunk_size_tokens = max(CHUNK_SIZE // 4, 50)  # Minimum 50 tokens

    if overlap_tokens is None:
        overlap_tokens = max(CHUNK_OVERLAP // 4, 10)  # Minimum 10 tokens

    # Vérifier que l'overlap n'est pas trop grand
    if overlap_tokens >= chunk_size_tokens:
        overlap_tokens = chunk_size_tokens // 4

    # Tokenizer le texte
    try:
        # add_special_tokens=False pour éviter [CLS], [SEP] dans le comptage
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except Exception as e:
        print(f"⚠️  Erreur de tokenization: {e}")
        # Fallback sur chunking par caractères
        from .chunk_fixed import chunk_text
        return chunk_text(text)

    if not tokens:
        return []

    chunks = []
    start = 0

    while start < len(tokens):
        # Extraire le chunk de tokens
        end = min(start + chunk_size_tokens, len(tokens))
        chunk_tokens = tokens[start:end]

        # Décoder les tokens en texte
        try:
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunk_text = chunk_text.strip()

            if chunk_text:
                chunks.append(chunk_text)
        except Exception as e:
            print(f"⚠️  Erreur de décodage: {e}")

        # Avancer avec overlap
        start = end - overlap_tokens
        if start >= len(tokens):
            break

    return chunks


def estimate_token_count(text):
    """
    Estime le nombre de tokens dans un texte.

    Args:
        text: Le texte à analyser

    Returns:
        Nombre de tokens estimé
    """
    try:
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception:
        # Fallback sur approximation : 1 token ≈ 4 caractères
        return len(text) // 4


def prepare_chunks_for_db_tokens(documents, chunk_size_tokens=None, overlap_tokens=None):
    """
    Prépare les documents en chunks par tokens pour insertion dans ChromaDB.

    Args:
        documents: Liste de tuples (nom_fichier, contenu)
        chunk_size_tokens: Nombre de tokens par chunk
        overlap_tokens: Nombre de tokens de chevauchement

    Returns:
        Tuple (texts, metadatas, ids) prêts pour ChromaDB
    """
    all_texts = []
    all_metadatas = []
    all_ids = []

    # Calculer les tailles par défaut si nécessaire
    if chunk_size_tokens is None:
        chunk_size_tokens = max(CHUNK_SIZE // 4, 50)
    if overlap_tokens is None:
        overlap_tokens = max(CHUNK_OVERLAP // 4, 10)

    chunk_id = 0
    for filename, content in documents:
        chunks = chunk_text_by_tokens(content, chunk_size_tokens, overlap_tokens)

        # Calculer statistiques
        total_tokens = estimate_token_count(content)
        avg_tokens_per_chunk = total_tokens // len(chunks) if chunks else 0

        print(f"   → {filename}: {len(chunks)} chunks "
              f"(~{avg_tokens_per_chunk} tokens/chunk, {total_tokens} tokens total)")

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                all_texts.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_method": "token",
                    "token_count": estimate_token_count(chunk)
                })
                all_ids.append(f"doc_{chunk_id}")
                chunk_id += 1

    return all_texts, all_metadatas, all_ids


def analyze_tokenization(text, max_examples=5):
    """
    Analyse la tokenization d'un texte (utile pour debug).

    Args:
        text: Texte à analyser
        max_examples: Nombre d'exemples de tokens à afficher

    Returns:
        Dictionnaire avec statistiques
    """
    tokenizer = get_tokenizer()

    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_strings = tokenizer.convert_ids_to_tokens(tokens[:max_examples])

    return {
        "total_tokens": len(tokens),
        "total_chars": len(text),
        "ratio_chars_per_token": len(text) / len(tokens) if tokens else 0,
        "example_tokens": token_strings,
        "example_text": text[:100] + "..." if len(text) > 100 else text
    }
