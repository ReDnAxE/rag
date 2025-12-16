"""
Stratégies de chunking pour le système RAG.

Ce package contient différentes stratégies de découpage de texte :
- chuk_llamaindex: intégration de LlamaIndex pour un chunking avancé.
"""

from .chunk_strategy import prepare_chunks_for_db

__all__ = [
    'prepare_chunks_for_db'
]
