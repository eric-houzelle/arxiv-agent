
# Agent Arxiv (LangGraph)

Agent LangGraph clé en main pour surveiller les soumissions ArXiv des dernières 24 h, générer des analyses structurées, scorer automatiquement chaque papier et proposer un post LinkedIn prêt à publier.

## Fonctionnalités clés
- Veille quotidienne configurable par catégories ArXiv ou requête libre.
- Génération d’analyses détaillées + scoring multi-critères via un LLM compatible OpenAI.
- Mise en cache locale (contenu PDF, analyses, scores) pour éviter de reconsommer les mêmes appels.
- Production d’un brief des top 5 papiers et d’un post LinkedIn rédigé à partir de prompts personnalisables.

## Installation rapide
```bash
git clone <URL_DU_DEPOT>
cd agent-arxiv
python -m venv agent-arxiv-env
source agent-arxiv-env/bin/activate
pip install -r requirements.txt
```

## Configuration
Créer un fichier `.env` à la racine (chargé automatiquement) :

```env
AI_ENDPOINTS_ACCESS_TOKEN=votre_token
MODEL=gpt-oss-120b
BASE_URL=https://oai.endpoints.preprod.ai.cloud.ovh.net/v1
LINKEDIN_POST_LANGUAGE=fr
LINKEDIN_POST_TEMPERATURE=0.4
```

- `AI_ENDPOINTS_ACCESS_TOKEN`, `MODEL`, `BASE_URL` : paramètres d’accès à votre fournisseur compatible OpenAI.
- `LINKEDIN_POST_LANGUAGE` : langue du post final (ex. `fr`, `en`).
- `LINKEDIN_POST_TEMPERATURE` : créativité appliquée uniquement à la génération LinkedIn.

## Exécution
```bash
python app.py
```

La CLI affiche les papiers triés par score global avec leurs scores détaillés, puis imprime la proposition de post LinkedIn. La requête ArXiv peut être surchargée en modifiant l’argument de `run_workflow()` dans `app.py`.

## Flux opérationnel
1. **Recherche ArXiv** (`agent_arxiv.nodes.search_arxiv`) : récupère les soumissions récentes dans les catégories par défaut `cs.CL`, `cs.AI`, `cs.IR`, `cs.MA` (modifiable).
2. **Récupération PDF** (`fetch_pdf_content`) : télécharge les PDF, extrait le texte et le stocke dans le cache.
3. **Analyse LLM** (`analyze_papers`) : produit une synthèse détaillée injectée ensuite dans le scoring.
4. **Scoring** (`score_papers`) : applique les critères définis dans `prompts/*.md`.
5. **Curation LinkedIn** (`write_linkedin_post`) : assemble les 5 meilleurs papiers, formate un brief et rédige un post conforme aux consignes.

L’orchestration est réalisée via `agent_arxiv.workflow` qui compile un `StateGraph` LangGraph.

## Personnalisation
- **Prompts de scoring** : éditer `prompts/originality.md`, `prompts/impact.md`, etc. pour changer les guidelines.
- **Prompt système LinkedIn** : mettre à jour `prompts/linkedin_system.md`.
- **Langue / température** : ajuster les variables d’environnement listées plus haut.
- **Catégories par défaut** : modifier `DEFAULT_CATEGORIES` dans `agent_arxiv/config.py`.

## Structure du projet
- `app.py` : point d’entrée CLI qui exécute le workflow et affiche les résultats.
- `agent_arxiv/config.py` : constantes (catégories, prompts, repo).
- `agent_arxiv/state.py` : état partagé entre les nœuds LangGraph.
- `agent_arxiv/nodes.py` : implémentation des nœuds (search, PDF, analyse, scoring, LinkedIn).
- `agent_arxiv/prompts.py` : chargement et assemblage des prompts.
- `agent_arxiv/papers.py` : utilitaires de scoring et de mise en forme.
- `agent_arxiv/workflow.py` : construction et compilation du graphe LangGraph.
- `cache.py` : persistance locale pour éviter de relancer les traitements sur les mêmes papiers.

## License
Apache 2.0
