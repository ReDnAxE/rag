"""
Stratégies de chunking pour le système RAG.

Ce package contient différentes méthodes de découpage de texte :
- chunk_fixed : Découpe par taille fixe en caractères
- chunk_recursive : Découpe récursive par structure (paragraphes, phrases, mots)
- chunk_token : Découpe par tokens linguistiques
- chunk_semantic : Découpe par ruptures sémantiques
- chunk_langchain : Intégration avec LangChain text splitters
- chunk_unified : Interface unifié pour toutes les stratégies
"""

from .chunk_fixed import chunk_text as chunk_fixed
from .chunk_recursive import chunk_text_recursive
from .chunk_unified import chunk_text, prepare_chunks_for_db

__all__ = [
    'chunk_fixed',
    'chunk_text_recursive',
    'chunk_text',
    'prepare_chunks_for_db',
]
