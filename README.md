# ChromaDB pour Ollama - Projet RAG

Ce projet permet de créer une base de données vectorielle ChromaDB à partir de documents texte, pour l'utiliser ensuite dans un système RAG (Retrieval-Augmented Generation) avec Ollama.

## Structure du projet

```
ia-ollama/
├── documents/                      # Dossier contenant les documents source
│   ├── intelligence_artificielle.txt
│   ├── programmation_python.txt
│   ├── bases_de_donnees.txt
│   ├── ollama_guide.txt
│   └── embeddings_rag.txt
├── chroma_db/                      # Base ChromaDB (créée après exécution)
├── config.py                       # Configuration du projet
├── document_loader.py              # Module de chargement des documents
├── text_utils.py                   # Utilitaires de traitement de texte
├── chroma_manager.py               # Gestionnaire ChromaDB
├── main.py                         # Script principal
├── example_rag_ollama.py           # Exemple d'utilisation avec Ollama
└── requirements.txt                # Dépendances Python
```

## Installation

1. Installez les dépendances :
```bash
pip3 install -r requirements.txt
```

2. Assurez-vous qu'Ollama est installé sur votre système :
```bash
# Sur Linux
curl -fsSL https://ollama.com/install.sh | sh

# Ou téléchargez depuis https://ollama.com
```

## Utilisation

### 1. Créer la base ChromaDB

Exécutez le script principal :
```bash
python3 main.py
```

Ce script va :
- Charger tous les fichiers `.txt` du dossier `documents/`
- Découper les documents en chunks
- Créer une base ChromaDB dans `./chroma_db/`
- Proposer un test de requête

### 2. Ajouter vos propres documents

Placez simplement vos fichiers `.txt` dans le dossier `documents/` et relancez `main.py`.

### 3. Utiliser avec Ollama

Utilisez l'exemple fourni :
```bash
python3 example_rag_ollama.py
```

## Configuration

Modifiez `config.py` pour ajuster les paramètres :

- `DOCUMENTS_DIR` : Dossier des documents source
- `CHROMA_DB_PATH` : Chemin de la base ChromaDB
- `COLLECTION_NAME` : Nom de la collection
- `CHUNK_SIZE` : Taille des chunks (en caractères)
- `CHUNK_OVERLAP` : Chevauchement entre chunks

## Utilisation dans un Modelfile Ollama

Vous pouvez créer un Modelfile personnalisé qui interroge la base :

```
FROM llama3.2

SYSTEM """
Tu es un assistant qui répond aux questions en utilisant le contexte fourni.
Si tu ne trouves pas la réponse dans le contexte, dis-le clairement.
"""
```

Puis dans votre code Python, récupérez le contexte pertinent depuis ChromaDB avant d'envoyer la requête à Ollama.

## Modules

### config.py
Contient tous les paramètres de configuration du projet.

### document_loader.py
- `load_documents(documents_dir)` : Charge les documents depuis un dossier
- `get_documents_summary(documents)` : Retourne les statistiques des documents

### text_utils.py
- `chunk_text(text, chunk_size, overlap)` : Découpe un texte en chunks
- `prepare_chunks_for_db(documents)` : Prépare les chunks pour ChromaDB

### chroma_manager.py
Classe `ChromaDBManager` pour gérer ChromaDB :
- `connect()` : Connexion à la base
- `create_collection()` : Création de la collection
- `insert_documents()` : Insertion des documents
- `query()` : Recherche dans la base
- `get_stats()` : Statistiques de la collection

### main.py
Script principal qui orchestre la création de la base.

## Exemple de code RAG

```python
import chromadb

# Charger la base
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("documents")

# Rechercher des documents pertinents
results = collection.query(
    query_texts=["Qu'est-ce que le RAG?"],
    n_results=3
)

# Utiliser avec Ollama
context = "\n".join(results['documents'][0])
# ... envoyer à Ollama
```

## Notes

- Les embeddings sont générés automatiquement par ChromaDB
- Par défaut, ChromaDB utilise un modèle d'embedding intégré
- Vous pouvez configurer ChromaDB pour utiliser des modèles d'embedding personnalisés
- La base est persistante et peut être réutilisée sans recréation
