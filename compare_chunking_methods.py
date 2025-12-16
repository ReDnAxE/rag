#!/usr/bin/env python3
"""
Script pour comparer les différentes méthodes de chunking.
"""

from document_loader import load_documents
from config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

# Import des différentes méthodes
from chunk_strategies.chunk_fixed import chunk_text as chunk_fixed
from chunk_strategies.chunk_recursive import chunk_text_recursive
# Méthodes optionnelles
try:
    from chunk_strategies.chunk_token import chunk_text_by_tokens
    TOKEN_AVAILABLE = True
except ImportError:
    TOKEN_AVAILABLE = False
    print("⚠️  Chunking par tokens non disponible (transformers requis)")

try:
    from chunk_strategies.chunk_semantic import chunk_text_semantic
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("⚠️  Chunking sémantique non disponible (sentence-transformers requis)")

try:
    from chunk_strategies.chunk_langchain import chunk_text_langchain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain non disponible")


def analyze_chunks(chunks, method_name):
    """Analyse les caractéristiques des chunks."""
    if not chunks:
        return None

    sizes = [len(c) for c in chunks]
    return {
        "method": method_name,
        "count": len(chunks),
        "avg_size": sum(sizes) / len(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "total_chars": sum(sizes)
    }


def compare_methods():
    """Compare toutes les méthodes de chunking disponibles."""
    print("=" * 80)
    print("COMPARAISON DES MÉTHODES DE CHUNKING")
    print("=" * 80)

    # Charger un document de test
    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        print("\n✗ Aucun document trouvé dans", DOCUMENTS_DIR)
        return

    # Prendre le premier document pour la comparaison
    filename, content = documents[0]
    print(f"\n📄 Document test : {filename}")
    print(f"   Taille : {len(content):,} caractères")
    print(f"\n⚙️  Paramètres : CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={CHUNK_OVERLAP}")

    results = []

    # 1. Méthode fixe (actuelle)
    print("\n" + "-" * 80)
    print("1️⃣  Méthode FIXE (actuelle)")
    print("-" * 80)
    chunks = chunk_fixed(content, CHUNK_SIZE, CHUNK_OVERLAP)
    stats = analyze_chunks(chunks, "Fixe")
    results.append(stats)
    print_stats(stats)
    print_sample(chunks[0] if chunks else "")

    # 2. Méthode récursive
    print("\n" + "-" * 80)
    print("2️⃣  Méthode RÉCURSIVE")
    print("-" * 80)
    chunks = chunk_text_recursive(content, CHUNK_SIZE, CHUNK_OVERLAP)
    stats = analyze_chunks(chunks, "Récursive")
    results.append(stats)
    print_stats(stats)
    print_sample(chunks[0] if chunks else "")

    # 3. Méthode par tokens (si disponible)
    if TOKEN_AVAILABLE:
        print("\n" + "-" * 80)
        print("3️⃣  Méthode PAR TOKENS")
        print("-" * 80)
        chunk_size_tokens = max(CHUNK_SIZE // 4, 50)
        overlap_tokens = max(CHUNK_OVERLAP // 4, 10)
        print(f"   Configuration : ~{chunk_size_tokens} tokens/chunk, overlap={overlap_tokens} tokens")
        chunks = chunk_text_by_tokens(content, chunk_size_tokens, overlap_tokens)
        stats = analyze_chunks(chunks, "Tokens")
        results.append(stats)
        print_stats(stats)
        print_sample(chunks[0] if chunks else "")

    # 4. Méthode sémantique (si disponible)
    if SEMANTIC_AVAILABLE:
        print("\n" + "-" * 80)
        print("4️⃣  Méthode SÉMANTIQUE")
        print("-" * 80)
        print("⏳ Calcul des embeddings en cours...")
        chunks = chunk_text_semantic(content, CHUNK_SIZE, threshold=0.5)
        stats = analyze_chunks(chunks, "Sémantique")
        results.append(stats)
        print_stats(stats)
        print_sample(chunks[0] if chunks else "")

    # 5. LangChain récursif (si disponible)
    if LANGCHAIN_AVAILABLE:
        print("\n" + "-" * 80)
        print("5️⃣  LangChain RÉCURSIVE")
        print("-" * 80)
        chunks = chunk_text_langchain(content, "recursive", CHUNK_SIZE, CHUNK_OVERLAP)
        stats = analyze_chunks(chunks, "LangChain-Recursive")
        results.append(stats)
        print_stats(stats)
        print_sample(chunks[0] if chunks else "")

    # Résumé comparatif
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ COMPARATIF")
    print("=" * 80)
    print(f"\n{'Méthode':<25} {'Chunks':<10} {'Taille moy.':<15} {'Min':<10} {'Max':<10}")
    print("-" * 80)

    for stats in results:
        print(f"{stats['method']:<25} {stats['count']:<10} "
              f"{stats['avg_size']:<15.1f} {stats['min_size']:<10} {stats['max_size']:<10}")

    # Recommandations
    print("\n" + "=" * 80)
    print("💡 RECOMMANDATIONS")
    print("=" * 80)
    print("""
1. 🏆 RÉCURSIVE : Meilleur compromis qualité/performance
   - Respecte la structure du texte (paragraphes, phrases)
   - Chunks cohérents sémantiquement
   - Pas de dépendances supplémentaires
   - ✅ RECOMMANDÉ pour la plupart des cas

2. 🎯 TOKENS : Précision linguistique maximale
   - Découpe exacte par tokens du modèle d'embedding
   - Garantit respect des limites du modèle (384 dimensions)
   - Idéal pour contrôle précis de la taille
   - ✅ BON pour optimisation fine

3. 🚀 SÉMANTIQUE : Meilleure qualité, mais plus lent
   - Détecte les ruptures de sujet automatiquement
   - Idéal pour documents longs et complexes
   - Coût : calcul d'embeddings supplémentaire
   - ⚠️  Utiliser pour des documents de haute valeur

4. 🏭 LANGCHAIN : Production-ready, bien testé
   - Solution éprouvée et maintenue
   - Compatible avec l'écosystème LangChain
   - Dépendance supplémentaire
   - ✅ BON pour projets avec LangChain existant

5. 📏 FIXE : Simple mais limité
   - Rapide et prévisible
   - Peut couper au milieu des phrases
   - ❌ À éviter sauf contraintes spécifiques
""")


def print_stats(stats):
    """Affiche les statistiques."""
    print(f"   Nombre de chunks : {stats['count']}")
    print(f"   Taille moyenne   : {stats['avg_size']:.1f} caractères")
    print(f"   Taille min/max   : {stats['min_size']} / {stats['max_size']} caractères")
    print(f"   Total            : {stats['total_chars']:,} caractères")


def print_sample(chunk, max_length=200):
    """Affiche un échantillon de chunk."""
    sample = chunk[:max_length] + ("..." if len(chunk) > max_length else "")
    print(f"\n   📝 Exemple de chunk :")
    print(f"   {sample}")


if __name__ == "__main__":
    compare_methods()
