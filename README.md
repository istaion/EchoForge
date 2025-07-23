# 🎭 EchoForge

<div align="center">
  <img src="logo_echoforge.jpg" alt="EchoForge Logo" width="200"/>
  
  **Framework d'agents conversationnels intelligents avec RAG et mémoire avancée**
  
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.dev/)
  [![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange.svg)](https://langchain-ai.github.io/langgraph/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 📋 Table des matières

- [🎯 Vue d'ensemble](#-vue-densemble)
- [✨ Fonctionnalités principales](#-fonctionnalités-principales)
- [🚀 Installation rapide](#-installation-rapide)
- [🎮 Démonstration](#-démonstration)
- [🏗️ Architecture](#️-architecture)
- [📖 Guide d'utilisation](#-guide-dutilisation)
- [🎨 Exemples](#-exemples)
- [🤝 Contribution](#-contribution)
- [📄 License](#-license)

---

## 🎯 Vue d'ensemble

**EchoForge** est un framework avancé pour créer des personnages de jeu vidéo et chatbots avec une **personnalité profonde** et **cohérente**. Utilisant les dernières technologies d'IA (RAG, LangGraph, LangChain), EchoForge permet de développer des agents conversationnels qui se souviennent, apprennent et évoluent au fil des interactions.

### 🔥 Points forts

- **🧠 Mémoire avancée** : Système de résumés automatiques et persistance des conversations
- **🎭 Personnalités complexes** : Traits de caractère, émotions et cohérence comportementale
- **🔗 Architecture modulaire** : Graphes LangGraph pour des workflows conversationnels sophistiqués
- **📚 RAG intégré** : Récupération de connaissances contextuelles avec FAISS
- **🎮 Prêt pour le jeu** : Système de triggers, actions et gestion d'état
- **🔧 Multi-LLM** : Support Ollama, Groq, OpenAI, Mistral

---

## ✨ Fonctionnalités principales

### 🤖 Agents intelligents
- **Graphes conversationnels** avec LangGraph
- **Routing adaptatif** selon la complexité
- **Triggers d'entrée/sortie** pour actions de jeu
- **Workflows personnalisables**

### 🧠 Système de mémoire
- **Résumés automatiques** des conversations longues
- **Persistance PostgreSQL** avec checkpoints
- **Contexte de session** et filtrage multi-utilisateur
- **Intégration transparente** dans les réponses

### 🎭 Personnalités riches
- **Traits psychologiques** quantifiés
- **États émotionnels** dynamiques
- **Relations interpersonnelles** évolutives
- **Cohérence comportementale** garantie

### 📚 RAG avancé
- **Embeddings multilingues** avec Ollama
- **Vector stores FAISS** optimisés
- **Recherche contextuelle** monde + personnage
- **Chunking intelligent** par type de contenu

---

## 🚀 Installation rapide

### Prérequis
- **Python 3.12+**
- **PostgreSQL** (pour la persistance)
- **Ollama** (optionnel, pour les modèles locaux)

### 1. Cloner le projet
```bash
git clone https://github.com/votre-repo/echoforge.git
cd echoforge
```

### 2. Installer les dépendances
```bash
pip install -e .
```

### 3. Configuration
```bash
cp .env.example .env
# Éditer .env avec vos clés API et configuration DB
```

### 4. Initialiser la base de données
```bash
python scripts/setup_environment.py
```

### 5. Lancer l'exemple
```bash
python examples/island_game/ui.py
```

🎉 **Interface web disponible sur http://localhost:7860**

---

## 🎮 Démonstration

L'exemple principal est un **jeu d'aventure sur une île** où votre montgolfière s'est écrasée. Vous devez interagir avec 4 personnages uniques pour réparer votre véhicule :

### 🏝️ Personnages du jeu

| Personnage | Rôle | Personnalité | Capacités spéciales |
|------------|------|--------------|-------------------|
| 👑 **Martine** | Maire | Diplomate, garde les secrets | Donne de l'or, révèle le trésor |
| 🔨 **Claude** | Forgeron | Pragmatique, aime les cookies | Répare la montgolfière |
| ✂️ **Azzedine** | Styliste | Artistique, perfectionniste | Vend du tissu, recoud la toile |
| 👩‍🍳 **Roberte** | Cuisinière | Généreuse mais territoriale | Fournit des cookies |

### 🎯 Fonctionnalités démo

- **💬 Conversations naturelles** avec mémoire long-terme
- **🎯 Système de quêtes** avec triggers automatiques  
- **📊 Gestion des ressources** (or, cookies, tissu, alcool)
- **💾 Sauvegarde par session** avec restauration
- **🧠 Debug de mémoire** en temps réel

---

## 🏗️ Architecture

### 📦 Structure modulaire

```
echoforge/
├── agents/          # 🤖 Système LangGraph (graphes, nœuds, conditions)
├── core/            # ⚙️ Moteur RAG et providers LLM
├── personality/     # 🎭 Traits, émotions, cohérence
├── conversation/    # 💬 Dialogue, contexte, triggers
├── knowledge/       # 📚 Base de connaissances, relations
├── data/           # 📄 Gestion données et schémas
└── utils/          # 🔧 Configuration, logging, monitoring
```

### 🔄 Flux de conversation

```mermaid
graph TD
    A[Message utilisateur] --> B[Chargement mémoire]
    B --> C[Analyse triggers d'entrée]
    C --> D[Évaluation complexité]
    D --> E{Besoin RAG ?}
    E -->|Oui| F[Recherche connaissances]
    E -->|Non| G[Génération directe]
    F --> H[Génération avec contexte]
    G --> I[Analyse triggers de sortie]
    H --> I
    I --> J[Mise à jour mémoire]
    J --> K[Réponse finale]
```

### 🧠 Système de mémoire

- **Messages récents** : Gardés en mémoire vive (configurable)
- **Résumés automatiques** : Déclenchés par seuil ou "bye"
- **Persistance DB** : PostgreSQL avec métadonnées de session
- **Checkpoints LangGraph** : État complet des graphes

---

## 📖 Guide d'utilisation

### 🎨 Créer un personnage simple

```python
from echoforge.personality import Character
from echoforge.agents.graphs.character_graph import CharacterGraphManager

# Définir la personnalité
character_data = {
    "name": "Alice",
    "role": "Guide touristique",
    "personality": {
        "traits": {"friendliness": 0.9, "knowledge": 0.8},
        "speech_style": "Enthousiaste et informatif"
    },
    "triggers": {
        "input": {
            "ask_directions": {
                "trigger": "Demande d'itinéraire",
                "threshold": 0.7,
                "effect": "Donne des directions détaillées"
            }
        }
    }
}

# Créer le gestionnaire
manager = CharacterGraphManager()

# Traiter une conversation
async def chat():
    response = await manager.process_message(
        user_message="Pouvez-vous m'indiquer le chemin vers le musée ?",
        character_data=character_data,
        player_data={"inventory": []},
        session_id="session_001"
    )
    print(response["response"])
```

### 🔧 Configuration avancée

```python
from echoforge.utils.config import get_config

config = get_config()

# Paramètres mémoire  
config.max_messages_without_summary = 50
config.keep_recent_messages = 15

# Provider LLM
config.llm_provider = "groq"  # ou "ollama", "openai", "mistral"
config.llm_model = "llama-3.1-8b-instant"

# Base de données
config.database_url = "postgresql://user:pass@localhost/echoforge"
```

### 📊 Monitoring et debug

```python
# Activer le debug
import os
os.environ['ECHOFORGE_DEBUG'] = 'true'

# Statistiques de mémoire
memory_stats = response.get("memory_stats", {})
print(f"Messages totaux: {memory_stats['total_messages']}")
print(f"Résumés: {memory_stats['summaries']}")

# Informations de traitement
debug_info = response.get("debug_info", {})
print(f"Temps de traitement: {debug_info['final_stats']['total_processing_time']:.3f}s")
```

---

## 🎨 Exemples

### 🏝️ Jeu d'aventure (Complet)
```bash
python examples/island_game/ui.py
```
Interface web complète avec :
- 4 personnages uniques avec mémoire
- Système de quêtes et ressources  
- Sauvegarde multi-session
- Debug en temps réel

### 🤖 Chatbot simple
```bash
python examples/chatbot/simple_bot.py
```
Bot conversationnel basique pour tester les fonctionnalités.

### 📚 Histoire interactive
```bash
python examples/interactive_story/story_engine.py
```
Narrateur adaptatif qui se souvient de vos choix.

### 👥 Galerie de personnages
```bash
python examples/character_gallery/historical_figures.py
```
Collection de personnages historiques prêts à l'emploi.

---

## 🛠️ Développement

### 🧪 Tests
```bash
# Tests unitaires
pytest tests/unit/

# Tests d'intégration
pytest tests/integration/

# Coverage
pytest --cov=echoforge tests/
```

### 📝 Scripts utiles
```bash
# Créer un personnage via CLI
python scripts/create_character.py --name "Merlin" --role "Wizard"

# Import de connaissances externes
python scripts/import_knowledge.py --source "world_data.json"

# Benchmark des modèles
python scripts/benchmark_models.py
```

### 🏗️ Étendre EchoForge

1. **Nouveaux nœuds LangGraph** : `echoforge/agents/nodes/`
2. **Providers LLM** : `echoforge/core/llm_providers.py`
3. **Triggers personnalisés** : `echoforge/conversation/triggers.py`
4. **Workflows** : `echoforge/agents/workflows/`

---

## 🤝 Contribution

Nous accueillons toutes les contributions ! 

### 🐛 Signaler un bug
[Créer une issue](https://github.com/votre-repo/echoforge/issues/new?template=bug_report.md)

### 💡 Proposer une fonctionnalité
[Créer une issue](https://github.com/votre-repo/echoforge/issues/new?template=feature_request.md)

### 🔧 Contribuer au code
1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

### 📋 Guidelines
- **Code** : Suivre PEP 8, typage avec mypy
- **Tests** : Coverage > 80%
- **Documentation** : Docstrings + exemples
- **Commits** : [Conventional Commits](https://conventionalcommits.org/)

---

## 🔗 Liens utiles

- **🏠 [Site web](https://echoforge.dev)** (à venir)
- **📖 [Documentation complète](https://docs.echoforge.dev)** (à venir)  
- **💬 [Discord](https://discord.gg/echoforge)** (à venir)
- **🐦 [Twitter](https://twitter.com/echoforge_ai)** (à venir)

## 📊 Roadmap

### 🎯 Version 0.2.0
- [ ] Interface graphique pour création de personnages
- [ ] Support Redis pour mise à l'échelle
- [ ] Plugins communautaires
- [ ] API REST complète

### 🚀 Version 0.3.0
- [ ] Agents multi-modaux (vision, audio)
- [ ] Workflows visuels drag & drop
- [ ] Déploiement cloud (Docker, K8s)
- [ ] Marketplace de personnages

---

## 📄 License

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **LangChain** pour l'écosystème LLM
- **LangGraph** pour les workflows d'agents
- **Ollama** pour les modèles locaux
- **Groq** pour l'inférence rapide
- **FAISS** pour la recherche vectorielle
- **Gradio** pour l'interface utilisateur

---

<div align="center">
  
  **⭐ N'oubliez pas de mettre une étoile si ce projet vous plaît ! ⭐**
  
  Fait avec ❤️ par la communauté EchoForge
  
</div>