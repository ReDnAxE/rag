# Système RAG avec ChromaDB et Ollama

Système de Retrieval-Augmented Generation (RAG) utilisant ChromaDB pour le stockage vectoriel et Ollama pour la génération de réponses. Ce projet permet de créer une base de connaissances à partir de documents texte et d'interroger cette base avec un modèle de langage local.

## Fonctionnalités

- **Chargement de documents** : Support des fichiers `.txt` et `.md`
- **Chunking avancé** : Utilisation de LlamaIndex avec plusieurs stratégies (sentence, token, semantic, window)
- **Stockage vectoriel** : ChromaDB avec embeddings Sentence Transformers
- **Génération de réponses** : Ollama avec streaming en temps réel
- **Modèle personnalisé** : Configuration via Modelfile pour optimiser le RAG
- **Mode interactif** : Interface en ligne de commande pour poser des questions

## Architecture

```
Documents (.txt, .md)
    ↓
Chunking (LlamaIndex)
    ↓
Embeddings (all-MiniLM-L6-v2)
    ↓
ChromaDB (base vectorielle)
    ↓
Recherche de similarité
    ↓
Ollama (génération avec contexte)
```

## Prérequis

- Python 3.8+
- [Ollama](https://ollama.ai/) installé et en cours d'exécution
- 4 Go de RAM minimum (8 Go recommandés pour les modèles plus grands)

## Installation

### 1. Cloner le projet

```bash
git clone <votre-repo>
cd ia-ollama
```

### 2. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

**Dépendances principales :**
- `chromadb` : Base de données vectorielle
- `pysqlite3-binary` : Support SQLite pour ChromaDB
- `llama-index-core` : Framework de chunking avancé

**Optionnel** (pour chunking sémantique) :
```bash
pip install llama-index-embeddings-huggingface
```

### 3. Installer Ollama

Téléchargez et installez Ollama depuis [ollama.ai](https://ollama.ai/)

### 4. Télécharger un modèle de base

```bash
ollama pull mistral
```

### 5. Créer le modèle personnalisé

```bash
ollama create rag-assistant -f Modelfile
```

## Configuration

Modifiez le fichier `config.py` selon vos besoins

### Stratégies de chunking disponibles

| Méthode | Description | Usage recommandé |
|---------|-------------|------------------|
| `sentence` | Découpe par phrases | **Recommandé pour RAG** - Préserve la cohérence sémantique |
| `token` | Découpe par tokens (tiktoken) | Documents techniques, contrôle précis de la taille |
| `semantic` | Découpe par ruptures sémantiques | Documents longs avec changements de sujets |
| `window` | Fenêtres contextuelles (3 phrases avant/après) | Maximum de contexte pour chaque chunk |

## Utilisation

### Étape 1 : Préparer vos documents

Placez vos fichiers `.txt` ou `.md` dans le répertoire `documents/` :

```bash
mkdir -p documents
cp mes_documents/*.txt documents/
```

### Étape 2 : Créer la base ChromaDB

```bash
python main.py
```

Ce script :
1. Charge tous les documents du répertoire
2. Les découpe en chunks selon la stratégie configurée
3. Génère les embeddings
4. Crée la base ChromaDB

### Étape 3 : Interroger le système RAG

```bash
python example_rag_ollama.py
```

Le script :
1. Exécute 3 questions de démonstration
2. Lance un mode interactif pour vos propres questions

## Structure du projet

```
ia-ollama/
├── README.md                      # Ce fichier
├── requirements.txt               # Dépendances Python
├── config.py                      # Configuration globale
├── Modelfile                      # Configuration du modèle Ollama
│
├── main.py                        # Script de création de la base
├── example_rag_ollama.py          # Script d'interrogation (avec streaming)
│
├── document_loader.py             # Chargement des documents
├── chroma_manager.py              # Gestion de ChromaDB
│
├── chunk_strategies/              # Stratégies de chunking
│   ├── __init__.py
│   ├── chunk_strategy.py          # Interface commune
│   └── chunk_llamaindex.py        # Implémentation LlamaIndex
│
├── documents/                     # Vos documents (à créer)
│   ├── document1.txt
│   └── document2.md
│
└── chroma_db/                     # Base ChromaDB (générée)
```

## Personnalisation

### Modifier le prompt du modèle

Éditez le fichier `Modelfile` :

```dockerfile
FROM mistral:latest

SYSTEM """
Tu es un assistant spécialisé dans...
[Personnalisez les instructions]
"""

PARAMETER temperature 0.3
PARAMETER num_ctx 4096
PARAMETER num_predict 512
```

Puis recréez le modèle :
```bash
ollama create rag-assistant -f Modelfile
```

### Changer de modèle de base

Dans `Modelfile`, modifiez la ligne `FROM` :

```dockerfile
# Modèles plus rapides
FROM llama3.2:1b      # Très rapide, 1B paramètres
FROM phi3:mini        # Bon compromis

# Modèles plus puissants
FROM llama3.1:8b      # Plus précis
FROM mixtral:8x7b     # Excellent pour RAG
```

### Ajuster les paramètres de chunking

Si vos chunks sont trop grands ou trop petits, modifiez `config.py` :

```python
# Pour des chunks plus petits (réponses plus précises)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30

# Pour des chunks plus grands (plus de contexte)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
```

## Dépannage

### Erreur : "Impossible de se connecter à Ollama"

Vérifiez qu'Ollama est lancé :
```bash
ollama serve
```

### Erreur : "Model 'rag-assistant' not found"

Créez le modèle personnalisé :
```bash
ollama create rag-assistant -f Modelfile
```

### La génération est très lente

Solutions :
1. Utilisez un modèle plus léger dans le Modelfile : `FROM llama3.2:1b`
2. Réduisez `num_predict` dans le Modelfile : `PARAMETER num_predict 256`
3. Vérifiez que vous utilisez un GPU si disponible

### Les chunks sont trop grands

La lenteur peut venir de chunks trop volumineux. Pour vérifier :

```python
# Dans un terminal Python
import chromadb
from config import CHROMA_DB_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
results = collection.get()

# Analyser les tailles
sizes = [len(doc) for doc in results['documents']]
print(f"Taille moyenne: {sum(sizes)/len(sizes):.0f} caractères")
print(f"Taille max: {max(sizes)} caractères")
```

Si les chunks sont trop grands, réduisez `CHUNK_SIZE` dans `config.py` et recréez la base.

### Erreur SQLite

Le projet inclut un workaround pour SQLite < 3.35. Si vous avez toujours des erreurs :

```bash
pip install --upgrade pysqlite3-binary
```

## Performance et optimisation

### Taille des chunks vs qualité des réponses

- **Petits chunks (200-400 caractères)** : Réponses plus précises, risque de manquer du contexte
- **Chunks moyens (500-800 caractères)** : **Recommandé** - Bon équilibre
- **Grands chunks (1000+ caractères)** : Plus de contexte, génération plus lente

### Nombre de documents récupérés

Dans `example_rag_ollama.py`, vous pouvez ajuster le nombre de documents :

```python
def search_documents(self, query, n_results=3):  # Modifier ici
    # ...
```

- `n_results=1-2` : Plus rapide, moins de contexte
- `n_results=3-5` : **Recommandé**
- `n_results=5+` : Plus de contexte, génération plus lente

## Améliorations futures

- [ ] Support d'autres formats (PDF, DOCX)
- [ ] Interface web avec Gradio/Streamlit
- [ ] Métriques de qualité des réponses
- [ ] Cache des embeddings pour accélérer
- [ ] Support de plusieurs collections ChromaDB
- [ ] Filtrage par métadonnées

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Ressources

- [Documentation Ollama](https://ollama.ai/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Guide sur le RAG](https://www.pinecone.io/learn/retrieval-augmented-generation/)
