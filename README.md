# Agent Arxiv (LangGraph + OpenAI)

Ce projet est un petit agent qui :
- interroge Arxiv en fonction d'une requête,
- analyse les papiers retournés avec un LLM OpenAI,
- attribue un score aux papiers,
- affiche les meilleurs résultats en sortie console.

Le flux est orchestré avec **LangGraph** et **LangChain** dans `app.py`.

## Installation

1. **Cloner le dépôt** :

```bash
git clone <URL_DU_DEPOT>
cd agent-arxiv
```

2. **Créer et activer un environnement virtuel (recommandé)** :

```bash
python -m venv agent-arxiv-env
source agent-arxiv-env/bin/activate
```

3. **Installer les dépendances** :

```bash
pip install -r requirements.txt
```

## Configuration des clés API

L'agent utilise `ChatOpenAI` (modèle `gpt-4o-mini`). Tu dois définir ta clé OpenAI dans l'environnement :

```bash
export OPENAI_API_KEY="ta_cle_openai"
```

Tu peux aussi utiliser un fichier `.env` (géré par `python-dotenv`) si tu préfères, par exemple :

```env
OPENAI_API_KEY=ta_cle_openai
```

## Utilisation

Le point d'entrée principal est `app.py`. Par défaut, la requête utilisateur est définie dans le `if __name__ == "__main__":` :

```python
user_query = "LLM training OR RAG OR agents OR retrieval-augmented"
```

Pour lancer l'agent :

```bash
python app.py
```

Le script :
- lance la recherche Arxiv,
- analyse les papiers,
- calcule un score global,
- affiche les **5 meilleurs papiers** (titre, score, URL) dans le terminal.

## Personnalisation

- **Changer la requête de recherche** : modifie la variable `user_query` dans `app.py`.
- **Changer le modèle OpenAI** : modifie la ligne suivante dans `app.py` :

```python
llm = ChatOpenAI(model="gpt-4o-mini")
```

## Structure du projet

- `app.py` : définition de l'état, des nœuds LangGraph (recherche Arxiv, analyse, scoring) et exécution du graphe.
- `requirements.txt` : dépendances Python du projet.
- `.gitignore` : ignore l'environnement virtuel, les fichiers temporaires, etc.

## License

À définir (MIT, Apache 2.0, …) selon ton choix.


