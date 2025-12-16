# Stratégies de Chunking

Ce package contient les différentes stratégies de découpage de texte pour le système RAG.

## Structure des fichiers

| Fichier | Description |
|---------|-------------|
| `chunk_fixed.py` | Découpage par taille fixe en caractères (méthode basique) |
| `chunk_recursive.py` | Découpage récursif par structure (paragraphes → phrases → mots) - **RECOMMANDÉ** |
| `chunk_token.py` | Découpage par tokens linguistiques (précision maximale) |
| `chunk_semantic.py` | Découpage par ruptures sémantiques (haute qualité, plus lent) |
| `chunk_langchain.py` | Intégration avec LangChain text splitters (production-ready) |
| `chunk_unified.py` | Interface unifiée pour toutes les stratégies |
| `__init__.py` | Exports du package |

## Utilisation

### Import simple

```python
from chunk_strategies import prepare_chunks_for_db

documents = [("doc1.txt", "contenu..."), ("doc2.txt", "contenu...")]
texts, metadatas, ids = prepare_chunks_for_db(documents)
```

### Import de fonctions spécifiques

```python
from chunk_strategies.chunk_recursive import chunk_text_recursive
from chunk_strategies.chunk_token import chunk_text_by_tokens

# Utiliser une stratégie spécifique
chunks = chunk_text_recursive(text, chunk_size=500, overlap=50)
```

### Configuration via config.py

```python
# Dans config.py
CHUNK_METHOD = "recursive"  # ou "token", "semantic", "langchain", "fixed"

# Dans votre code
from chunk_strategies import chunk_text

# Utilisera automatiquement la méthode configurée
chunks = chunk_text(text)
```

## Choix de la stratégie

| Cas d'usage | Stratégie recommandée |
|-------------|----------------------|
| Usage général | `recursive` |
| Contrôle précis de la taille | `token` |
| Documents complexes avec changements de sujet | `semantic` |
| Intégration LangChain existante | `langchain` |
| Performance maximale (non recommandé) | `fixed` |

## Dépendances

### Obligatoires (toujours installées)
- Aucune dépendance supplémentaire pour `fixed` et `recursive`

### Optionnelles
- **Token** : `pip install transformers`
- **Semantic** : `pip install sentence-transformers`
- **LangChain** : `pip install langchain langchain-text-splitters`

## Comparaison

Pour comparer toutes les stratégies sur vos documents :

```bash
python3 compare_chunking_methods.py
```

## Architecture interne

Chaque stratégie implémente deux fonctions principales :

1. **`chunk_text(...)`** ou équivalent : Découpe un texte en chunks
2. **`prepare_chunks_for_db(documents, ...)`** : Prépare les chunks avec métadonnées pour ChromaDB

Le module `chunk_unified.py` centralise l'accès à toutes les stratégies via une interface commune.
