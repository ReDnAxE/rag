# Guide des méthodes de Chunking

Ce guide explique les différentes méthodes de chunking disponibles et comment les utiliser.

## Vue d'ensemble

Le chunking est l'étape qui découpe vos documents en morceaux plus petits pour l'insertion dans ChromaDB. La qualité du chunking impacte directement la qualité des résultats de recherche du système RAG.

## Méthodes disponibles

### 1️⃣ **Récursive** (RECOMMANDÉ) ⭐

```python
CHUNK_METHOD = "recursive"
```

**Description** : Découpe le texte en respectant la structure naturelle (paragraphes → phrases → mots)

**Avantages** :
- ✅ Respecte la structure du document
- ✅ Chunks sémantiquement cohérents
- ✅ Pas de dépendances supplémentaires
- ✅ Bon compromis performance/qualité

**Inconvénients** :
- Taille de chunks variable

**Utilisation** :
```python
from config import CHUNK_SIZE, CHUNK_OVERLAP
from text_utils_recursive import chunk_text_recursive

chunks = chunk_text_recursive(text, CHUNK_SIZE, CHUNK_OVERLAP)
```

---

### 2️⃣ **Fixe** (Méthode actuelle)

```python
CHUNK_METHOD = "fixed"
```

**Description** : Découpe par taille fixe avec détection basique des limites de mots

**Avantages** :
- ✅ Simple et rapide
- ✅ Taille prévisible

**Inconvénients** :
- ❌ Peut couper au milieu des phrases
- ❌ Ne respecte pas la structure sémantique
- ❌ Qualité de recherche inférieure

**Utilisation** :
```python
from text_utils import chunk_text

chunks = chunk_text(text, chunk_size=500, overlap=50)
```

---

### 3️⃣ **Sémantique** (Avancé) 🚀

```python
CHUNK_METHOD = "semantic"
```

**Description** : Détecte automatiquement les ruptures sémantiques en analysant la similarité entre phrases

**Installation** :
```bash
pip install sentence-transformers
```

**Avantages** :
- ✅ Meilleure qualité : détecte les changements de sujet
- ✅ Chunks naturellement cohérents
- ✅ Idéal pour documents complexes

**Inconvénients** :
- ⚠️ Plus lent (calcul d'embeddings)
- ⚠️ Dépendance supplémentaire
- ⚠️ Taille de chunks très variable

**Utilisation** :
```python
from text_utils_semantic import chunk_text_semantic

chunks = chunk_text_semantic(text, max_chunk_size=500, threshold=0.5)
```

**Paramètres** :
- `threshold` : Seuil de rupture sémantique (0-1)
  - Plus haut (0.7-0.9) = chunks plus petits, plus focalisés
  - Plus bas (0.3-0.5) = chunks plus grands, plus de contexte

---

### 4️⃣ **LangChain** (Production) 🏭

```python
CHUNK_METHOD = "langchain"
```

**Description** : Utilise les text splitters de LangChain, solution production-ready

**Installation** :
```bash
pip install langchain langchain-text-splitters
```

**Avantages** :
- ✅ Bien testé et maintenu
- ✅ Compatible écosystème LangChain
- ✅ Multiple stratégies disponibles

**Inconvénients** :
- ⚠️ Dépendance supplémentaire (assez lourde)

**Utilisation** :
```python
from text_utils_langchain import chunk_text_langchain

# Méthode récursive
chunks = chunk_text_langchain(text, method="recursive")

# Par tokens
chunks = chunk_text_langchain(text, method="token")
```

---

## Configuration

### Méthode simple : Éditer `config.py`

```python
# config.py

# Choisir la méthode
CHUNK_METHOD = "recursive"  # ou "fixed", "semantic", "langchain"

# Paramètres généraux
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Pour chunking sémantique
SEMANTIC_THRESHOLD = 0.5
```

### Utilisation du module unifié

```python
from text_utils_unified import chunk_text, prepare_chunks_for_db

# Utilise automatiquement la méthode de config.py
chunks = chunk_text(text)

# Ou spécifier explicitement
chunks = chunk_text(text, method="recursive")
```

---

## Comparaison des méthodes

Pour comparer toutes les méthodes sur vos documents :

```bash
python3 compare_chunking_methods.py
```

Ce script affichera :
- Nombre de chunks par méthode
- Taille moyenne/min/max
- Exemples de chunks
- Recommandations

---

## Migration depuis le système actuel

### Option 1 : Utiliser le module unifié (Recommandé)

1. **Éditer `main.py`** :

```python
# Remplacer
from text_utils import prepare_chunks_for_db

# Par
from text_utils_unified import prepare_chunks_for_db
```

2. **Configurer dans `config.py`** :
```python
CHUNK_METHOD = "recursive"
```

3. **Recréer la base** :
```bash
python3 main.py
```

### Option 2 : Modification manuelle

Éditer directement `main.py` pour utiliser une méthode spécifique :

```python
# Importer la méthode choisie
from text_utils_recursive import prepare_chunks_for_db

# Ou pour sémantique
from text_utils_semantic import prepare_chunks_for_db_semantic as prepare_chunks_for_db

# Le reste du code reste identique
```

---

## Recommandations par type de document

| Type de document | Méthode recommandée | Raison |
|------------------|---------------------|---------|
| Documentation technique | **Récursive** | Structure claire (sections, paragraphes) |
| Articles de blog | **Récursive** | Bon équilibre qualité/performance |
| Livres, longs textes | **Sémantique** | Détecte les changements de chapitres/thèmes |
| Code source | **LangChain-token** | Respect de la syntaxe |
| Conversations, chats | **Fixe** | Pas de structure forte |
| Documents scientifiques | **Sémantique** | Transitions complexes entre sujets |

---

## Paramètres recommandés

### Pour documents courts (< 5000 caractères)
```python
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
CHUNK_METHOD = "recursive"
```

### Pour documents moyens (5000-50000 caractères)
```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNK_METHOD = "recursive"
```

### Pour documents longs (> 50000 caractères)
```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNK_METHOD = "semantic"
SEMANTIC_THRESHOLD = 0.6
```

### Pour performance maximale
```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNK_METHOD = "recursive"  # Éviter "semantic"
```

---

## Dépannage

### Chunks trop grands
- Réduire `CHUNK_SIZE`
- Utiliser méthode "recursive" ou "semantic"

### Chunks trop petits
- Augmenter `CHUNK_SIZE`
- Pour sémantique : réduire `SEMANTIC_THRESHOLD`

### Résultats de recherche de mauvaise qualité
- Essayer méthode "recursive" ou "semantic"
- Augmenter `CHUNK_OVERLAP` (50-100)
- Vérifier que les chunks ne coupent pas les phrases

### Performance lente
- Éviter méthode "semantic" pour gros volumes
- Utiliser "recursive" (meilleur compromis)
- Réduire le nombre de documents

---

## Exemple complet

```python
#!/usr/bin/env python3
from document_loader import load_documents
from text_utils_unified import prepare_chunks_for_db
from chroma_manager import ChromaDBManager
from config import DOCUMENTS_DIR, CHROMA_DB_PATH, COLLECTION_NAME

# Charger les documents
documents = load_documents(DOCUMENTS_DIR)

# Créer les chunks (utilise CHUNK_METHOD de config.py)
texts, metadatas, ids = prepare_chunks_for_db(documents)

# Ou spécifier la méthode
texts, metadatas, ids = prepare_chunks_for_db(documents, method="semantic")

# Insérer dans ChromaDB
db = ChromaDBManager(CHROMA_DB_PATH, COLLECTION_NAME)
db.connect()
db.create_collection(reset=True)
db.insert_documents(texts, metadatas, ids)
db.close()
```

---

## Questions fréquentes

**Q : Dois-je recréer la base si je change de méthode ?**
R : Oui, les chunks seront différents donc il faut recréer avec `python3 main.py`

**Q : Quelle méthode est la plus rapide ?**
R : "fixed" > "recursive" >> "langchain" >> "semantic"

**Q : Quelle méthode donne les meilleurs résultats ?**
R : "semantic" ≥ "recursive" > "langchain" > "fixed"

**Q : Puis-je combiner plusieurs méthodes ?**
R : Non, mais vous pouvez créer plusieurs collections avec différentes méthodes

**Q : La méthode affecte-t-elle les embeddings ?**
R : Non, seul le découpage change. Les embeddings sont toujours générés par all-MiniLM-L6-v2
