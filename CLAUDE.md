# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Vue d'ensemble du projet

Il s'agit d'un système RAG (Retrieval-Augmented Generation) qui utilise ChromaDB pour le stockage vectoriel et Ollama pour l'inférence LLM. Le système charge des documents texte, les découpe en chunks, les stocke dans une base de données vectorielle, et permet une recherche sémantique pour fournir du contexte pertinent lors de questions-réponses avec Ollama.

## Commandes essentielles

### Installation et dépendances
```bash
# Installer les dépendances Python
pip3 install -r requirements.txt

# Installer Ollama (si pas déjà installé)
curl -fsSL https://ollama.com/install.sh | sh
```

### Création de la base de données
```bash
# Créer/reconstruire la base de données vectorielle ChromaDB à partir des documents dans documents/
python3 main.py
```

Ce script effectue :
- Chargement de tous les fichiers `.txt` depuis `documents/`
- Découpage des documents (par défaut : 500 caractères avec 50 de chevauchement)
- Création de la collection ChromaDB dans `./chroma_db/`
- Proposition d'une requête de test interactive

### Configuration du modèle Ollama
```bash
# Télécharger le modèle de base (requis avant de créer le modèle personnalisé)
ollama pull llama3.2
# ou
ollama pull mistral

# Créer un modèle RAG personnalisé depuis le Modelfile
ollama create rag-assistant -f Modelfile

# Lister les modèles disponibles
ollama list

# Tester le modèle directement (sans contexte RAG)
ollama run rag-assistant

# Supprimer un modèle
ollama rm rag-assistant
```

### Exécution du système RAG
```bash
# Exécuter le système RAG complet (ChromaDB + Ollama)
python3 example_rag_ollama.py
```

Ce script démontre :
- Chargement de la collection ChromaDB
- Recherche de documents pertinents
- Construction de prompts avec contexte
- Génération de réponses via l'API Ollama

## Architecture

### Composants principaux

**1. Pipeline de documents (main.py)**
- Orchestre le workflow d'ingestion des documents
- Flux : documents/ → document_loader → text_utils → chroma_manager

**2. Chargeur de documents (document_loader.py)**
- `load_documents()` : Charge les fichiers `.txt` depuis un répertoire
- `get_documents_summary()` : Retourne les statistiques des documents

**3. Utilitaires de texte (text_utils.py)**
- `chunk_text()` : Divise le texte en chunks avec chevauchement, en évitant de couper les mots
- `prepare_chunks_for_db()` : Formate les chunks avec métadonnées (source, chunk_index) pour ChromaDB

**4. Gestionnaire ChromaDB (chroma_manager.py)**
- Encapsule les opérations ChromaDB avec workaround de compatibilité SQLite
- Méthodes clés :
  - `connect()` : Initialise le client persistant
  - `create_collection(reset=True)` : Crée/réinitialise la collection
  - `insert_documents()` : Insertion par lots avec suivi de progression
  - `query()` : Recherche sémantique
  - `get_stats()` : Métadonnées de la collection

**Important** : Ce module inclut un workaround SQLite en haut (substitution pysqlite3) qui doit rester en place pour la compatibilité avec les anciennes versions de SQLite.

**5. Système RAG (example_rag_ollama.py)**
- Implémentation RAG complète combinant recherche ChromaDB et génération Ollama
- Classe `RAGSystem` :
  - `search_documents()` : Récupère les top-k chunks pertinents
  - `generate_response()` : Construit le prompt avec contexte et interroge l'API Ollama
  - `ask()` : Question-réponse de bout en bout

**6. Configuration (config.py)**
- Paramètres centralisés :
  - `DOCUMENTS_DIR` : Emplacement des documents source
  - `CHROMA_DB_PATH` : Chemin de la base vectorielle
  - `CHUNK_SIZE`, `CHUNK_OVERLAP` : Paramètres de découpage du texte
  - `BATCH_SIZE` : Taille de lot pour l'insertion ChromaDB

**7. Modelfile**
- Définit le comportement du modèle Ollama personnalisé
- Utilise actuellement `mistral:latest` comme modèle de base
- Configuré pour RAG avec température basse (0.3) pour des réponses factuelles
- Grande fenêtre de contexte (4096 tokens) pour accommoder les chunks de documents
- Le prompt système impose des réponses basées sur le contexte et la citation des sources

**8. Package de chunking (chunk_strategies/)**
Package Python contenant toutes les stratégies de découpage de texte :
- **chunk_fixed.py** : Méthode fixe par caractères (ancienne, conservée pour compatibilité)
- **chunk_recursive.py** : Méthode récursive par structure (recommandée, par défaut)
- **chunk_token.py** : Méthode par tokens linguistiques (précision maximale)
- **chunk_semantic.py** : Méthode sémantique (avancée, détection de ruptures)
- **chunk_langchain.py** : Intégration LangChain (production-ready)
- **chunk_unified.py** : Module unifié qui sélectionne la méthode selon `CHUNK_METHOD`
- **__init__.py** : Exports du package pour imports simplifiés
- **README.md** : Documentation détaillée du package

**9. Outils de comparaison**
- **compare_chunking_methods.py** : Compare toutes les méthodes sur vos documents
- **CHUNKING_GUIDE.md** : Documentation complète des stratégies de chunking

### Flux de données

```
[documents/*.txt]
      ↓
document_loader.load_documents()
      ↓
chunk_strategies.prepare_chunks_for_db()
      ↓ (selon CHUNK_METHOD)
      ├─ recursive (défaut) → chunk_strategies.chunk_recursive
      ├─ token → chunk_strategies.chunk_token
      ├─ fixed → chunk_strategies.chunk_fixed
      ├─ semantic → chunk_strategies.chunk_semantic
      └─ langchain → chunk_strategies.chunk_langchain
      ↓
Chunks avec embeddings (all-MiniLM-L6-v2)
      ↓
chroma_manager.insert_documents()
      ↓
[chroma_db/] ← Base de données vectorielle persistante
      ↓
Question utilisateur → RAGSystem.search_documents()
      ↓
Contexte récupéré + Question → RAGSystem.generate_response()
      ↓
API Ollama (modèle rag-assistant)
      ↓
Réponse
```

## Détails d'implémentation importants

### Compatibilité SQLite
`chroma_manager.py` et `example_rag_ollama.py` incluent tous deux un workaround SQLite critique en haut :
```python
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```
Ceci substitue pysqlite3-binary plus récent pour les systèmes avec SQLite < 3.35. Ne pas supprimer ce code.

### Télémétrie ChromaDB
La télémétrie ChromaDB est désactivée via variable d'environnement pour la compatibilité Python 3.8 :
```python
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
```

### Communication API Ollama
- Point de terminaison par défaut : `http://localhost:11434/api/generate`
- Mode non-streaming utilisé (`"stream": False`)
- Timeout de 180 secondes pour les modèles lourds
- Retourne du JSON avec clé `response` contenant le texte généré

### Stratégies de chunking

Le système supporte plusieurs méthodes de chunking configurables via `CHUNK_METHOD` dans `config.py` :

**1. Récursive (par défaut - RECOMMANDÉ)**
- Découpe intelligente : paragraphes → phrases → mots
- Respecte la structure naturelle du texte
- Meilleur compromis qualité/performance
- Utilise `text_utils_recursive.py`

**2. Tokens (précision linguistique)**
- Découpe par tokens du modèle d'embedding (all-MiniLM-L6-v2)
- Garantit respect exact des limites du modèle
- Idéal pour optimisation fine et contrôle précis
- Requiert : `pip install transformers`
- Utilise `text_utils_token.py`

**3. Fixe (ancienne méthode - par caractères)**
- Découpe par taille fixe en caractères avec détection basique des limites de mots
- Simple mais peut couper au milieu des phrases
- Utilise `text_utils.py`

**4. Sémantique (avancée)**
- Détecte automatiquement les ruptures sémantiques
- Qualité supérieure mais plus lent (calcul d'embeddings)
- Requiert : `pip install sentence-transformers`
- Utilise `text_utils_semantic.py`

**5. LangChain (production)**
- Utilise les text splitters de LangChain
- Requiert : `pip install langchain langchain-text-splitters`
- Utilise `text_utils_langchain.py`

Paramètres communs :
- `CHUNK_SIZE` : Taille maximale des chunks en caractères
- `CHUNK_OVERLAP` : Chevauchement entre chunks pour continuité du contexte
- Chaque chunk inclut des métadonnées : source, chunk_index, total_chunks, chunk_method

Pour comparer les méthodes : `python3 compare_chunking_methods.py`
Voir documentation complète : `CHUNKING_GUIDE.md`

### Sources de documents
Placer les fichiers `.txt` dans le répertoire `documents/`. Documents d'exemple inclus :
- intelligence_artificielle.txt
- programmation_python.txt
- bases_de_donnees.txt
- ollama_guide.txt
- embeddings_rag.txt

## Points de personnalisation

### Changer la méthode de chunking
Éditer `config.py` :
```python
CHUNK_METHOD = "recursive"  # Options : "recursive", "fixed", "semantic", "langchain"
```
Puis reconstruire : `python3 main.py`

Recommandations par type de document :
- Documents structurés (technique, blog) : `"recursive"` (par défaut)
- Optimisation fine de la taille : `"token"`
- Documents longs et complexes : `"semantic"`
- Projets avec LangChain existant : `"langchain"`
- Performance maximale : `"fixed"` (non recommandé pour la qualité)

### Changer la taille des chunks
Éditer `config.py` :
```python
CHUNK_SIZE = 500        # Augmenter pour plus de contexte par chunk
CHUNK_OVERLAP = 50      # Augmenter pour assurer une meilleure continuité
```
Puis reconstruire : `python3 main.py`

### Utiliser différents modèles Ollama
Éditer `example_rag_ollama.py` :
```python
ollama_model="llama3.2"  # ou "mistral", "phi", etc.
```

Ou modifier le `Modelfile` :
```
FROM llama3.2  # Changer le modèle de base
```
Puis recréer : `ollama create rag-assistant -f Modelfile`

### Ajuster les paramètres du modèle
Éditer le `Modelfile` :
- `temperature` : 0-1 (plus bas = plus déterministe)
- `top_p` : Seuil de nucleus sampling
- `num_ctx` : Taille de fenêtre de contexte (important pour RAG)
- `num_predict` : Longueur maximale de réponse

### Changer le nombre de documents récupérés
Dans `example_rag_ollama.py` :
```python
docs = self.search_documents(question, n_results=3)  # Ajuster n_results
```

## Tests

### Vérifier la création ChromaDB
Après avoir exécuté `python3 main.py` :
- Vérifier que le répertoire `./chroma_db/` existe
- Exécuter la requête de test interactive proposée
- Vérifier que le nombre de chunks correspond à la taille attendue des documents

### Tester la connexion Ollama
```bash
# Vérifier qu'Ollama est lancé
curl http://localhost:11434/api/tags

# Tester le modèle directement
ollama run rag-assistant "Question de test"
```

### Déboguer le pipeline RAG
Dans `example_rag_ollama.py`, la méthode `ask()` affiche :
- Documents récupérés depuis ChromaDB
- Noms des fichiers sources
- Aperçus des chunks
- Réponse générée complète
