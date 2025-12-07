# Guide d'utilisation

## Installation et configuration

### 1. Installer les dépendances Python

```bash
pip3 install -r requirements.txt
```

### 2. Créer la base ChromaDB

```bash
python3 main.py
```

Cette commande va :
- Charger les documents du dossier `documents/`
- Les découper en chunks
- Créer la base vectorielle ChromaDB

### 3. Créer le modèle Ollama personnalisé

```bash
ollama create rag-assistant -f Modelfile
```

Cette commande crée un nouveau modèle nommé `rag-assistant` basé sur votre Modelfile.

**Note:** Assurez-vous d'avoir d'abord téléchargé le modèle de base :
```bash
ollama pull llama3.2
```

## Utilisation

### Option 1 : Utiliser le script Python avec le système RAG complet

```bash
python3 example_rag_ollama.py
```

Ce script :
- Se connecte à ChromaDB
- Recherche les documents pertinents pour chaque question
- Envoie le contexte + question à Ollama
- Affiche la réponse

**Modifiez le modèle utilisé dans `example_rag_ollama.py`:**
```python
rag = RAGSystem(
    chroma_path=CHROMA_DB_PATH,
    collection_name=COLLECTION_NAME,
    ollama_model="rag-assistant"  # Utilisez votre modèle personnalisé
)
```

### Option 2 : Utiliser directement le modèle Ollama (sans ChromaDB)

```bash
ollama run rag-assistant
```

**Note:** Dans ce mode, vous devez fournir manuellement le contexte dans votre prompt.

Exemple de prompt :
```
Contexte:
[Collez ici le texte pertinent]

Question: Qu'est-ce que le RAG?
```

### Option 3 : Intégration dans votre propre code

```python
import chromadb
import requests

# 1. Charger ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("documents")

# 2. Rechercher les documents pertinents
results = collection.query(
    query_texts=["votre question"],
    n_results=3
)

# 3. Construire le contexte
context = "\n\n".join(results['documents'][0])

# 4. Créer le prompt
prompt = f"""Contexte pertinent:
{context}

Question: votre question

Instructions: Réponds en te basant sur le contexte."""

# 5. Appeler Ollama
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "rag-assistant",
        "prompt": prompt,
        "stream": False
    }
)

print(response.json()['response'])
```

## Personnalisation du Modelfile

Vous pouvez modifier le `Modelfile` pour :

### Changer le modèle de base
```
FROM llama3.2        # ou mistral, phi, etc.
```

### Ajuster les paramètres
```
PARAMETER temperature 0.3    # 0-1 : 0 = déterministe, 1 = créatif
PARAMETER top_p 0.9          # Diversité des réponses
PARAMETER num_ctx 4096       # Taille du contexte (important pour RAG)
```

### Modifier les instructions système
Éditez la section `SYSTEM` pour changer le comportement du modèle.

### Recréer le modèle après modification
```bash
ollama create rag-assistant -f Modelfile
```

## Commandes utiles Ollama

```bash
# Lister les modèles
ollama list

# Supprimer un modèle
ollama rm rag-assistant

# Voir les détails d'un modèle
ollama show rag-assistant

# Tester rapidement
ollama run rag-assistant "test question"
```

## Ajouter de nouveaux documents

1. Ajoutez vos fichiers `.txt` dans le dossier `documents/`
2. Relancez la création de la base :
```bash
python3 main.py
```

## Dépannage

### ChromaDB ne trouve pas la collection
Vérifiez que vous avez bien exécuté `python3 main.py` avant.

### Erreur de connexion Ollama
Vérifiez qu'Ollama est lancé :
```bash
ollama serve
```

### Le modèle n'existe pas
Créez-le avec :
```bash
ollama create rag-assistant -f Modelfile
```

### Performances lentes
- Réduisez `num_ctx` dans le Modelfile
- Réduisez `CHUNK_SIZE` dans `config.py`
- Utilisez un modèle plus petit (ex: `phi`)

## Architecture du système RAG

```
Question utilisateur
    ↓
[Script Python]
    ↓
Recherche sémantique → [ChromaDB]
    ↓
Documents pertinents trouvés
    ↓
Construction du prompt avec contexte
    ↓
Envoi à [Ollama + Modelfile]
    ↓
Réponse générée
```

Le Modelfile définit **comment** le modèle répond, tandis que ChromaDB fournit **le contexte** pertinent.
