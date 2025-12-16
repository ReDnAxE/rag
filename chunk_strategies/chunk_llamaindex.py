"""
Chunking avec LlamaIndex : solution optimisée pour RAG avec plusieurs stratégies.

Installation requise :
    pip install llama-index-core

Pour chunking sémantique :
    pip install llama-index-embeddings-huggingface
"""


def _convert_to_tokens(chunk_size, overlap):
    """
    Convertit les tailles en caractères vers des tailles en tokens.

    Args:
        chunk_size: Taille en caractères
        overlap: Chevauchement en caractères

    Returns:
        Tuple (chunk_size_tokens, overlap_tokens)
    """
    chunk_size_tokens = max(chunk_size // 4, 50)
    overlap_tokens = max(overlap // 4, 10)
    return chunk_size_tokens, overlap_tokens


def _create_sentence_splitter(chunk_size_tokens, overlap_tokens):
    """
    Crée un SentenceSplitter (découpe par phrases).

    Args:
        chunk_size_tokens: Taille du chunk en tokens
        overlap_tokens: Chevauchement en tokens

    Returns:
        Instance de SentenceSplitter
    """
    from llama_index.core.node_parser import SentenceSplitter

    return SentenceSplitter(
        chunk_size=chunk_size_tokens,
        chunk_overlap=overlap_tokens,
        separator=" ",
        paragraph_separator="\n\n",
    )


def _create_token_splitter(chunk_size_tokens, overlap_tokens):
    """
    Crée un TokenTextSplitter (découpe par tokens tiktoken).

    Args:
        chunk_size_tokens: Taille du chunk en tokens
        overlap_tokens: Chevauchement en tokens

    Returns:
        Instance de TokenTextSplitter
    """
    from llama_index.core.node_parser import TokenTextSplitter

    return TokenTextSplitter(
        chunk_size=chunk_size_tokens,
        chunk_overlap=overlap_tokens,
        separator=" ",
    )


def _create_semantic_splitter(chunk_size_tokens, overlap_tokens):
    """
    Crée un SemanticSplitterNodeParser (découpe par ruptures sémantiques).

    Args:
        chunk_size_tokens: Taille du chunk en tokens (utilisé en fallback)
        overlap_tokens: Chevauchement en tokens (utilisé en fallback)

    Returns:
        Instance de SemanticSplitterNodeParser ou SentenceSplitter (fallback)
    """
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # Utiliser le même modèle d'embedding que ChromaDB
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        return SemanticSplitterNodeParser(
            buffer_size=1,  # Nombre de phrases à comparer
            breakpoint_percentile_threshold=95,  # Seuil de rupture
            embed_model=embed_model,
        )
    except ImportError:
        print("⚠️  SemanticSplitter nécessite : pip install llama-index-embeddings-huggingface")
        print("   Utilisation de SentenceSplitter à la place...")
        return _create_sentence_splitter(chunk_size_tokens, overlap_tokens)


def _create_window_splitter():
    """
    Crée un SentenceWindowNodeParser (fenêtres contextuelles).

    Returns:
        Instance de SentenceWindowNodeParser
    """
    from llama_index.core.node_parser import SentenceWindowNodeParser

    return SentenceWindowNodeParser(
        window_size=3,  # Nombre de phrases avant/après
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )


def _nodes_to_chunks(nodes):
    """
    Extrait le texte des nodes LlamaIndex.

    Args:
        nodes: Liste de nodes LlamaIndex

    Returns:
        Liste de chunks de texte (non vides)
    """
    return [node.text for node in nodes if node.text.strip()]


def chunk_text_llamaindex(text, method, chunk_size, overlap):
    """
    Divise un texte en chunks en utilisant LlamaIndex NodeParsers.

    Args:
        text: Le texte à diviser
        method: Méthode de chunking
            - "sentence": SentenceSplitter (recommandé pour RAG)
            - "token": TokenTextSplitter (par tokens tiktoken)
            - "semantic": SemanticSplitterNodeParser (par ruptures sémantiques)
            - "window": SentenceWindowNodeParser (fenêtres contextuelles)
        chunk_size: Taille de chaque chunk (en caractères, converti en tokens)
        overlap: Chevauchement entre chunks (en caractères, converti en tokens)

    Returns:
        Liste de chunks de texte
    """
    from llama_index.core.schema import Document

    # Convertir les tailles en tokens
    chunk_size_tokens, overlap_tokens = _convert_to_tokens(chunk_size, overlap)

    # Créer le splitter approprié
    if method == "sentence":
        splitter = _create_sentence_splitter(chunk_size_tokens, overlap_tokens)
    elif method == "token":
        splitter = _create_token_splitter(chunk_size_tokens, overlap_tokens)
    elif method == "semantic":
        splitter = _create_semantic_splitter(chunk_size_tokens, overlap_tokens)
    elif method == "window":
        splitter = _create_window_splitter()
    else:
        raise ValueError(f"Méthode inconnue : {method}")

    # Créer un document et le découper en nodes
    doc = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])

    # Extraire le texte des nodes
    return _nodes_to_chunks(nodes)