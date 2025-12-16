"""
Configuration pour la création de la base ChromaDB.
"""

# Paramètres des documents
DOCUMENTS_DIR = "documents"

# Paramètres ChromaDB
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
COLLECTION_DESCRIPTION = "Documents pour RAG avec Ollama"

# Paramètres de chunking
EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"  # Modèle d'embedding Sentence Transformers
# Méthode de chunking
# Options :
# - "llamaindex" : Avec LlamaIndex (optimisé pour RAG)
CHUNK_STRATEGY = "llamaindex"
# method: Méthode de chunking (ici pour LlamaIndex)
#   - "sentence": SentenceSplitter (recommandé pour RAG)
#   - "token": TokenTextSplitter (par tokens tiktoken)
#   - "semantic": SemanticSplitterNodeParser (par ruptures sémantiques)
#   - "window": SentenceWindowNodeParser (fenêtres contextuelles)
CHUNK_METHOD = "sentence"
CHUNK_SIZE = 500  # Taille des chunks en caractères
CHUNK_OVERLAP = 50  # Chevauchement entre chunks
MIN_CHUNK_RATIO = 0.7  # Ratio minimum pour éviter de couper les mots

# Paramètres de traitement
BATCH_SIZE = 100  # Taille des lots pour l'insertion dans ChromaDB

