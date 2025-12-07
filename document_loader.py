"""
Module pour charger les documents depuis le système de fichiers.
"""

from pathlib import Path


def load_documents(documents_dir):
    """
    Charge tous les documents texte d'un répertoire.

    Args:
        documents_dir: Chemin vers le répertoire contenant les documents

    Returns:
        Liste de tuples (nom_fichier, contenu)
    """
    documents = []
    doc_path = Path(documents_dir)

    if not doc_path.exists():
        print(f"Erreur: Le répertoire {documents_dir} n'existe pas")
        return documents

    # Charger tous les fichiers .txt
    for file_path in doc_path.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append((file_path.name, content))
                print(f"✓ Chargé: {file_path.name}")
        except Exception as e:
            print(f"✗ Erreur lors de la lecture de {file_path.name}: {e}")

    return documents


def get_documents_summary(documents):
    """
    Retourne un résumé des documents chargés.

    Args:
        documents: Liste de tuples (nom_fichier, contenu)

    Returns:
        Dictionnaire avec les statistiques
    """
    total_chars = sum(len(content) for _, content in documents)
    return {
        "count": len(documents),
        "total_chars": total_chars,
        "avg_chars": total_chars // len(documents) if documents else 0,
        "files": [filename for filename, _ in documents]
    }
