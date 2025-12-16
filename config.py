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
CHUNK_SIZE = 500  # Taille des chunks en caractères
CHUNK_OVERLAP = 50  # Chevauchement entre chunks
MIN_CHUNK_RATIO = 0.7  # Ratio minimum pour éviter de couper les mots

# Méthode de chunking
# Options : "fixed", "recursive", "token", "semantic", "langchain"
# - "fixed" : Par caractères (basique, peut couper au milieu des phrases)
# - "recursive" : Par structure (paragraphes → phrases → mots) - RECOMMANDÉ
# - "token" : Par tokens linguistiques (précis, respect des limites du modèle)
# - "semantic" : Par ruptures sémantiques (haute qualité, plus lent)
# - "langchain" : Avec LangChain (production-ready)
CHUNK_METHOD = "recursive"

# Paramètres pour chunking sémantique
SEMANTIC_THRESHOLD = 0.5  # Seuil de rupture sémantique (0-1)

# Paramètres de traitement
BATCH_SIZE = 100  # Taille des lots pour l'insertion dans ChromaDB
