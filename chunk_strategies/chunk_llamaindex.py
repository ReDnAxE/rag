"""
Chunking avec LlamaIndex : solution optimisée pour RAG avec plusieurs stratégies.

Installation requise :
    pip install llama-index-core

Pour chunking sémantique :
    pip install llama-index-embeddings-huggingface
"""


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
        chunk_size: Taille de chaque chunk (en tokens pour LlamaIndex)
        overlap: Chevauchement entre chunks (en tokens)

    Returns:
        Liste de chunks de texte
    """

    # LlamaIndex travaille en tokens, convertir depuis caractères
    chunk_size_tokens = max(chunk_size // 4, 50)
    overlap_tokens = max(overlap // 4, 10)

    if method == "sentence":
        from llama_index.core.node_parser import SentenceSplitter

        # SentenceSplitter : découpe par phrases (meilleur pour RAG)
        splitter = SentenceSplitter(
            chunk_size=chunk_size_tokens,
            chunk_overlap=overlap_tokens,
            separator=" ",
            paragraph_separator="\n\n",
        )

    elif method == "token":
        from llama_index.core.node_parser import TokenTextSplitter

        # TokenTextSplitter : découpe par tokens (utilise tiktoken)
        splitter = TokenTextSplitter(
            chunk_size=chunk_size_tokens,
            chunk_overlap=overlap_tokens,
            separator=" ",
        )

    elif method == "semantic":
        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            # Utiliser le même modèle d'embedding que ChromaDB
            embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # SemanticSplitter : découpe par similarité sémantique
            splitter = SemanticSplitterNodeParser(
                buffer_size=1,  # Nombre de phrases à comparer
                breakpoint_percentile_threshold=95,  # Seuil de rupture
                embed_model=embed_model,
            )
        except ImportError:
            print("⚠️  SemanticSplitter nécessite : pip install llama-index-embeddings-huggingface")
            print("   Utilisation de SentenceSplitter à la place...")
            from llama_index.core.node_parser import SentenceSplitter
            splitter = SentenceSplitter(
                chunk_size=chunk_size_tokens,
                chunk_overlap=overlap_tokens,
            )

    elif method == "window":
        from llama_index.core.node_parser import SentenceWindowNodeParser

        # SentenceWindowNodeParser : crée des fenêtres de contexte autour des phrases
        # Utile pour conserver plus de contexte lors de la recherche
        splitter = SentenceWindowNodeParser(
            window_size=3,  # Nombre de phrases avant/après
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

    else:
        raise ValueError(f"Méthode inconnue : {method}")

    # LlamaIndex utilise des Documents, créer un document temporaire
    from llama_index.core.schema import Document

    doc = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])

    # Extraire le texte des nodes
    chunks = [node.text for node in nodes if node.text.strip()]

    return chunks