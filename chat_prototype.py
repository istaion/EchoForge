import gradio as gr
import json
from datetime import datetime, timezone
import os
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import math
import asyncio
import time
import uuid
import shutil
from langsmith import traceable
from echoforge.utils.config import get_config

config = get_config()

# Import du système d'agents LangGraph avec mémoire avancée
try:
    from echoforge.agents.graphs.character_graph import CharacterGraphManager
    from echoforge.agents.state.character_state import CharacterState
    from echoforge.agents.checkpointers.postgres_checkpointer import PostgreSQLCheckpointSaver
    AGENTS_AVAILABLE = True
    print("✅ Système d'agents LangGraph avec mémoire avancée chargé avec succès!")
except ImportError as e:
    print(f"⚠️ Erreur: Impossible d'importer le système d'agents: {e}")
    print("📝 Utilisation du système RAG de base comme fallback")
    AGENTS_AVAILABLE = False
    CharacterGraphManager = None
    PostgreSQLCheckpointSaver = None

# Fallback vers le système RAG existant si les agents ne sont pas disponibles
if not AGENTS_AVAILABLE:
    try:
        from main import EchoForgeRAG, ActionParsed
        RAG_AVAILABLE = True
        print("✅ Système RAG de base chargé comme fallback")
    except ImportError:
        print("❌ Aucun système de dialogue disponible!")
        EchoForgeRAG = None
        ActionParsed = None
        RAG_AVAILABLE = False
else:
    RAG_AVAILABLE = False
    EchoForgeRAG = None
    ActionParsed = None

# 🆕 Chemins des fichiers
CHARACTERS_TEMPLATE_PATH = "data/game_data/characters.json"
PLAYER_TEMPLATE_PATH = "data/game_data/player.json"
SESSIONS_DIR = "data/game_data/sessions"
PLAYER_SESSIONS_DIR = f"{SESSIONS_DIR}/player"
CHARACTERS_SESSIONS_DIR = f"{SESSIONS_DIR}/characters"

# S'assurer que les dossiers existent
for directory in [SESSIONS_DIR, PLAYER_SESSIONS_DIR, CHARACTERS_SESSIONS_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_game_data():
    """Charge les données du jeu (templates des personnages et joueur)"""
    # 🆕 Chargement du template des personnages (jamais modifié)
    with open(CHARACTERS_TEMPLATE_PATH, "r") as f:
        characters_template = json.load(f)
    
    # Chargement du template player (jamais modifié)
    try:
        with open(PLAYER_TEMPLATE_PATH, "r") as f:
            player_template = json.load(f)
    except FileNotFoundError:
        print("⚠️ Fichier player.json template non trouvé, création du template")
    
    return characters_template, player_template

def save_player_template(player_data):
    """Sauvegarde le template player (pour mise à jour du template seulement)"""
    try:
        with open(PLAYER_TEMPLATE_PATH, "w") as f:
            json.dump(player_data, f, indent=2, ensure_ascii=False)
        print("💾 Template joueur sauvegardé")
    except Exception as e:
        print(f"❌ Erreur sauvegarde template: {e}")

def save_characters_template(characters_data):
    """Sauvegarde le template des personnages (pour mise à jour du template seulement)"""
    try:
        with open(CHARACTERS_TEMPLATE_PATH, "w") as f:
            json.dump(characters_data, f, indent=2, ensure_ascii=False)
        print("💾 Template personnages sauvegardé")
    except Exception as e:
        print(f"❌ Erreur sauvegarde template personnages: {e}")

# 🆕 Fonctions pour la gestion des personnages par session
def get_characters_session_path(session_id: str) -> str:
    """Retourne le chemin du fichier de sauvegarde des personnages pour une session"""
    return f"{CHARACTERS_SESSIONS_DIR}/characters_{session_id}.json"

def get_player_session_path(session_id: str) -> str:
    """Retourne le chemin du fichier de sauvegarde du joueur pour une session"""
    return f"{PLAYER_SESSIONS_DIR}/player_{session_id}.json"

def load_characters_data_for_session(session_id: str) -> dict:
    """Charge les données des personnages pour une session spécifique"""
    session_file = get_characters_session_path(session_id)
    
    try:
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                characters_data = json.load(f)
            print(f"💾 Données personnages chargées pour session {session_id}")
            return characters_data
        else:
            # Nouvelle session : copie du template
            with open(CHARACTERS_TEMPLATE_PATH, "r") as f:
                template_data = json.load(f)
            
            print(f"🆕 Nouvelles données personnages créées pour session {session_id} (depuis template)")
            return template_data
            
    except Exception as e:
        print(f"❌ Erreur chargement personnages session {session_id}: {e}")
        # Fallback sur template
        with open(CHARACTERS_TEMPLATE_PATH, "r") as f:
            return json.load(f)

def save_characters_data_for_session(characters_data: dict, session_id: str):
    """
    Sauvegarde les données des personnages pour une session spécifique.
    🆕 Parcourt toutes les clés du template et sauvegarde le contenu correspondant du state.
    """
    try:
        session_file = get_characters_session_path(session_id)
        
        # Chargement du template pour s'assurer qu'on a toutes les clés
        with open(CHARACTERS_TEMPLATE_PATH, "r") as f:
            template_data = json.load(f)
        
        # 🆕 Construction des données à sauvegarder en parcourant le template
        session_characters_data = {}
        
        for character_id, template_character in template_data.items():
            if character_id in characters_data:
                # Copie toutes les clés du template avec les valeurs du state
                session_characters_data[character_id] = {}
                
                # Parcours de toutes les clés du template
                for key, template_value in template_character.items():
                    if key in characters_data[character_id]:
                        session_characters_data[character_id][key] = characters_data[character_id][key]
                    else:
                        # Si la clé n'existe pas dans le state, garde la valeur du template
                        session_characters_data[character_id][key] = template_value
                        
                print(f"💾 Personnage {character_id} sauvegardé ({len(session_characters_data[character_id])} clés)")
            else:
                # Si le personnage n'existe pas dans le state, garde le template
                session_characters_data[character_id] = template_character
                print(f"💾 Personnage {character_id} sauvegardé (depuis template)")
        
        # Ajout des métadonnées de session
        for character_id in session_characters_data:
            if "meta" not in session_characters_data[character_id]:
                session_characters_data[character_id]["meta"] = {}
            
            session_characters_data[character_id]["meta"]["last_updated"] = datetime.now(timezone.utc).isoformat()
            session_characters_data[character_id]["meta"]["session_id"] = session_id
        
        # Sauvegarde
        with open(session_file, "w") as f:
            json.dump(session_characters_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Données personnages sauvegardées pour session {session_id} ({len(session_characters_data)} personnages)")
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde personnages session {session_id}: {e}")

def load_player_data_for_session(session_id: str) -> dict:
    """Charge les données joueur pour une session spécifique"""
    session_file = get_player_session_path(session_id)
    
    try:
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                player_data = json.load(f)
            print(f"💾 Données joueur chargées pour session {session_id}")
            return player_data
        else:
            # Nouvelle session : copie du template
            with open(PLAYER_TEMPLATE_PATH, "r") as f:
                template_data = json.load(f)
            
            # Mise à jour des métadonnées pour la nouvelle session
            template_data["meta"]["created"] = datetime.now(timezone.utc).isoformat()
            template_data["meta"]["last_updated"] = datetime.now(timezone.utc).isoformat()
            template_data["meta"]["save_count"] = 0
            template_data["meta"]["session_id"] = session_id
            
            print(f"🆕 Nouvelles données joueur créées pour session {session_id}")
            return template_data
            
    except Exception as e:
        print(f"❌ Erreur chargement joueur session {session_id}: {e}")
        # Fallback sur template
        with open(PLAYER_TEMPLATE_PATH, "r") as f:
            return json.load(f)

def save_player_data_for_session(player_data: dict, session_id: str):
    """Sauvegarde les données joueur pour une session spécifique"""
    try:
        session_file = get_player_session_path(session_id)
        
        # Mise à jour des métadonnées
        if "meta" not in player_data:
            player_data["meta"] = {}
        
        player_data["meta"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        player_data["meta"]["save_count"] = player_data.get("meta", {}).get("save_count", 0) + 1
        player_data["meta"]["session_id"] = session_id
        
        with open(session_file, "w") as f:
            json.dump(player_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Données joueur sauvegardées pour session {session_id}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde joueur session {session_id}: {e}")

# 🆕 Fonction de sauvegarde complète de la session
def save_complete_session(session_id: str):
    """Sauvegarde complète de la session (joueur + personnages)"""
    if not game_state.get("session_initialized", False):
        print("⚠️ Aucune session active à sauvegarder")
        return
    
    try:
        # Sauvegarde du joueur
        if CURRENT_PLAYER_DATA:
            save_player_data_for_session(CURRENT_PLAYER_DATA, session_id)
        
        # 🆕 Sauvegarde des personnages
        save_characters_data_for_session(CHARACTERS, session_id)
        
        print(f"✅ Session complète sauvegardée: {session_id}")
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde session complète: {e}")


# Chargement initial des templates
CHARACTERS_TEMPLATE, PLAYER_TEMPLATE = load_game_data()

# 🆕 Variables globales pour la session courante
CURRENT_PLAYER_DATA = None
CHARACTERS = CHARACTERS_TEMPLATE.copy()  # Copie du template au démarrage

# Position de la montgolfière
BALLOON_POSITION = {"x": 120, "y": 120}

# État global du jeu avec session management
game_state = {
    "current_character": None,
    "current_session_id": None,
    "session_name": None,
    "player_position": BALLOON_POSITION.copy(),
    "chat_open": False,
    "chat_locked": False,
    "conversation_ending": False,
    "game_events": [],
    "start_time": time.time(),
    "memory_stats": {},
    "last_bye_score": 0.0,
    "session_initialized": False
}

# Synchronisation avec les données joueur - 🆕 Ajout de l'alcool
def sync_game_state_with_player_data():
    """Synchronise game_state avec CURRENT_PLAYER_DATA"""
    if CURRENT_PLAYER_DATA:
        game_state.update({
            "player_gold": CURRENT_PLAYER_DATA["player_stats"]["gold"],
            "player_cookies": CURRENT_PLAYER_DATA["player_stats"]["cookies"], 
            "player_fabric": CURRENT_PLAYER_DATA["player_stats"]["fabric"],
            "player_alcool": CURRENT_PLAYER_DATA["player_stats"]["alcool"],  
            "montgolfiere_repaired": CURRENT_PLAYER_DATA["montgolfiere_status"]["fully_operational"]
        })

# Instances globales
graph_manager = None
rag_system = None

# 🆕 Gestion des sessions
class SessionManager:
    """Gestionnaire des sessions de jeu."""
    
    @staticmethod
    def generate_session_id() -> str:
        """Génère un nouvel ID de session."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    @staticmethod
    def generate_session_name() -> str:
        """Génère un nom de session par défaut."""
        return f"Partie du {datetime.now().strftime('%d/%m/%Y à %H:%M')}"
    
    @staticmethod
    def get_available_sessions() -> List[Dict[str, str]]:
        """Récupère la liste des sessions disponibles."""
        sessions = []
        
        # 🆕 Sessions à partir des fichiers de sauvegarde joueur ET personnages
        player_files = []
        character_files = []
        
        if os.path.exists(PLAYER_SESSIONS_DIR):
            player_files = [f for f in os.listdir(PLAYER_SESSIONS_DIR) if f.startswith("player_session_") and f.endswith(".json")]
        
        if os.path.exists(CHARACTERS_SESSIONS_DIR):
            character_files = [f for f in os.listdir(CHARACTERS_SESSIONS_DIR) if f.startswith("characters_session_") and f.endswith(".json")]
        
        # Union des sessions depuis les deux sources
        all_session_files = set()
        for f in player_files:
            session_id = f.replace("player_", "").replace(".json", "")
            all_session_files.add(session_id)
        
        for f in character_files:
            session_id = f.replace("characters_", "").replace(".json", "")
            all_session_files.add(session_id)
        
        for session_id in all_session_files:
            display_name = session_id.replace("session_", "").replace("_", " ")
            
            # 🆕 Essaie de récupérer plus d'infos depuis les fichiers
            try:
                player_file = get_player_session_path(session_id)
                characters_file = get_characters_session_path(session_id)
                
                # Informations sur la session
                info_parts = []
                if os.path.exists(player_file):
                    with open(player_file, "r") as f:
                        player_data = json.load(f)
                    last_updated = player_data.get("meta", {}).get("last_updated", "")
                    if last_updated:
                        info_parts.append(last_updated[:16])  # YYYY-MM-DD HH:MM
                
                if os.path.exists(characters_file):
                    info_parts.append("Personnages")
                
                if info_parts:
                    display_name += f" - {' | '.join(info_parts)}"
                
            except:
                pass
            
            sessions.append({
                "session_id": session_id,
                "display_name": display_name,
                "has_player": os.path.exists(get_player_session_path(session_id)),
                "has_characters": os.path.exists(get_characters_session_path(session_id))
            })
        
        # Sessions à partir de la base de données (si agents disponibles)
        if AGENTS_AVAILABLE and graph_manager and hasattr(graph_manager, 'graph_manager'):
            try:
                for character_name in CHARACTERS_TEMPLATE.keys():
                    checkpointer = PostgreSQLCheckpointSaver(character_name)
                    db_session_ids = checkpointer.list_session_ids()
                    
                    for session_id in db_session_ids:
                        # Vérifie si on a déjà cette session depuis les fichiers
                        if not any(s["session_id"] == session_id for s in sessions):
                            sessions.append({
                                "session_id": session_id,
                                "display_name": session_id.replace("session_", "").replace("_", " ") + " (DB)",
                                "has_player": False,
                                "has_characters": False
                            })
                
            except Exception as e:
                print(f"⚠️ Erreur récupération sessions DB: {e}")
        
        # Trie par session_id (plus récent en premier)
        sessions.sort(key=lambda x: x["session_id"], reverse=True)
        
        return sessions
    
    @staticmethod
    def load_session(session_id: str) -> bool:
        """Charge une session existante."""
        try:
            global CURRENT_PLAYER_DATA, CHARACTERS
            
            # 🆕 Charge les données joueur ET personnages pour cette session
            CURRENT_PLAYER_DATA = load_player_data_for_session(session_id)
            CHARACTERS = load_characters_data_for_session(session_id)
            
            game_state["current_session_id"] = session_id
            game_state["session_name"] = session_id.replace("session_", "").replace("_", " ")
            game_state["session_initialized"] = True
            
            # Reset de l'état de jeu pour la nouvelle session
            game_state["current_character"] = None
            game_state["chat_open"] = False
            game_state["chat_locked"] = False
            game_state["conversation_ending"] = False
            game_state["game_events"] = []
            game_state["memory_stats"] = {}
            game_state["start_time"] = time.time()
            
            # Synchronisation avec les données joueur
            sync_game_state_with_player_data()
            
            print(f"✅ Session chargée: {session_id} (joueur + {len(CHARACTERS)} personnages)")
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement session: {e}")
            return False
    
    @staticmethod
    def create_new_session(session_name: str = None) -> str:
        """Crée une nouvelle session."""
        global CURRENT_PLAYER_DATA, CHARACTERS
        
        session_id = SessionManager.generate_session_id()
        
        if not session_name:
            session_name = SessionManager.generate_session_name()
        
        # 🆕 Charge les données template pour la nouvelle session
        CURRENT_PLAYER_DATA = load_player_data_for_session(session_id)
        CHARACTERS = load_characters_data_for_session(session_id)
        
        game_state["current_session_id"] = session_id
        game_state["session_name"] = session_name
        game_state["session_initialized"] = True
        
        # Reset complet du jeu
        game_state["current_character"] = None
        game_state["chat_open"] = False
        game_state["chat_locked"] = False
        game_state["conversation_ending"] = False
        game_state["game_events"] = []
        game_state["memory_stats"] = {}
        game_state["start_time"] = time.time()
        
        # Synchronisation avec les nouvelles données joueur
        sync_game_state_with_player_data()
        
        # 🆕 Sauvegarde initiale complète
        save_complete_session(session_id)
        
        print(f"✅ Nouvelle session créée: {session_id} ({session_name}) - joueur + {len(CHARACTERS)} personnages")
        return session_id


class EchoForgeAgentWrapper:
    """Wrapper pour intégrer les agents LangGraph avec mémoire avancée."""
    
    def __init__(self):
        self.graph_manager = None
        self.initialized = False
        self.error_message = None
        
        if AGENTS_AVAILABLE:
            try:
                # Teste d'abord avec checkpointer activé
                self.graph_manager = CharacterGraphManager(enable_checkpointer=True)
                self.initialized = True
                print("✅ Système d'agents initialisé avec checkpointer")
            except Exception as e:
                print(f"⚠️ Erreur avec checkpointer: {e}")
                try:
                    # Fallback sans checkpointer
                    self.graph_manager = CharacterGraphManager(enable_checkpointer=False)
                    self.initialized = True
                    print("✅ Système d'agents initialisé sans checkpointer")
                except Exception as e2:
                    print(f"❌ Erreur système d'agents: {e2}")
                    self.error_message = str(e2)
                    self.initialized = False
        else:
            self.error_message = "Système d'agents non disponible"
        
    async def get_character_response(self, character_key: str, user_message: str) -> str:
        """Obtient une réponse du personnage via le système d'agents avec mémoire."""
        
        if not self.initialized:
            error_msg = f"❌ Système d'agents non disponible"
            if self.error_message:
                error_msg += f": {self.error_message}"
            return error_msg
        
        # Vérification de l'initialisation de session
        if not game_state.get("session_initialized", False):
            return "❌ Aucune session active. Veuillez sélectionner ou créer une session."
        
        try:
            # 🆕 Utilise les données du personnage depuis le state de session
            character_data = CHARACTERS[character_key].copy()
            player_data = CURRENT_PLAYER_DATA.copy()
            
            # Utilisation du session_id actuel
            session_id = game_state.get("current_session_id")
            thread_id = f"game_conversation_{character_key}"
            
            # Traitement du message avec l'agent et mémoire avancée
            result = await self.graph_manager.process_message(
                user_message=user_message,
                character_data=character_data,
                player_data=player_data,
                thread_id=thread_id,
                session_id=session_id
            )
            
            # Vérification du résultat
            if not result or 'response' not in result:
                return f"❌ Erreur: Réponse invalide du système d'agents"
            
            # 🆕 Mise à jour des données du personnage dans le state global
            CHARACTERS[character_key]['conversation_history'] = result.get('conversation_history', [])
            if 'current_emotion' in result:
                CHARACTERS[character_key]['current_emotion'] = result['current_emotion']
            
            # Mise à jour des statistiques de mémoire
            if 'memory_stats' in result:
                game_state["memory_stats"][character_key] = result['memory_stats']
            
            # Traitement des triggers de sortie et actions
            await self._process_agent_actions(character_key, result, user_message)
            
            # Gestion du trigger bye
            output_triggers = result.get('output_trigger_probs', {})
            if output_triggers and isinstance(output_triggers, dict):
                bye_info = output_triggers.get('bye', {})
                if isinstance(bye_info, dict):
                    bye_score = bye_info.get('prob', 0.0)
                    game_state["last_bye_score"] = bye_score
                    
                    if bye_score > 0.9:
                        game_state["chat_locked"] = True
                        game_state["conversation_ending"] = True
                        # 🆕 Sauvegarde automatique complète quand bye est détecté
                        save_complete_session(session_id)
                        print(f"💾 Sauvegarde automatique complète déclenchée par bye (score: {bye_score:.2f})")
            
            # Récupération de la réponse
            response = result.get('response', '')
            
            # Informations sur le mode de fonctionnement
            fallback_info = result.get('fallback_info', {})
            emergency_fallback = result.get('emergency_fallback', False)
            
            # Ajout d'informations de debug en mode développement
            if os.getenv('ECHOFORGE_DEBUG', 'false').lower() == 'true':
                debug_info = result.get('debug_info', {})
                complexity = result.get('complexity_level', 'unknown')
                input_prob = result.get('input_trigger_probs')
                output_prob = result.get('output_trigger_probs')
                rag_used = bool(result.get('rag_results', []))
                processing_time = debug_info.get('final_stats', {}).get('total_processing_time', 0)
                
                memory_info = ""
                if 'memory_stats' in result:
                    stats = result['memory_stats']
                    memory_info = f"\n📊 Mémoire: {stats.get('total_messages', 0)} msgs | {stats.get('summaries', 0)} résumés"
                
                session_info = f"\n🔗 Session: {session_id}"
                
                fallback_debug = ""
                if fallback_info:
                    fallback_debug = f"\n⚠️ Mode fallback: {fallback_info.get('reason', 'unknown')}"
                elif emergency_fallback:
                    fallback_debug = f"\n🚨 Mode urgence: {result.get('error_info', {}).get('error', 'unknown')}"
                
                response += f"\n\n🐛 Debug: {complexity} | RAG: {rag_used} | {processing_time:.3f}s{memory_info}{session_info}{fallback_debug}\n input_probs : {input_prob} \n output_probs : {output_prob}"
            
            return response
            
        except Exception as e:
            print(f"❌ Erreur dans l'agent pour {character_key}: {str(e)}")
            # Fallback d'urgence local
            character_name = CHARACTERS[character_key].get('name', character_key)
            return f"*{character_name} semble troublé*\n\nExcusez-moi, je rencontre des difficultés techniques. Pouvez-vous reformuler votre message ?\n\n🔧 Erreur: {str(e)}"
    
    # 🆕 Fonction mise à jour pour gérer tous les nouveaux triggers
    async def _process_agent_actions(self, character_key: str, result: dict, user_message: str):
        """Traite les actions spéciales basées sur la réponse de l'agent avec tous les nouveaux triggers."""
        global CURRENT_PLAYER_DATA
        
        character_data = CHARACTERS[character_key]
        output_triggers = result.get('output_trigger_probs', {})
        
        # Vérification que output_triggers est un dict
        if not isinstance(output_triggers, dict):
            print(f"⚠️ output_triggers invalide pour {character_key}: {type(output_triggers)}")
            output_triggers = {}
        
        # Traitement des triggers de sortie avec valeurs
        for trigger_name, trigger_data in output_triggers.items():
            if not isinstance(trigger_data, dict):
                print(f"⚠️ Trigger data invalide pour {trigger_name}: {type(trigger_data)}")
                continue
                
            prob = trigger_data.get('prob', 0.0)
            value = trigger_data.get('value', 0)
            
            # Vérification que prob est un nombre
            if not isinstance(prob, (int, float)):
                print(f"⚠️ Probabilité invalide pour {trigger_name}: {type(prob)}")
                continue
            
            # Récupération du seuil depuis la config du personnage
            output_config = character_data.get('triggers', {}).get('output', {})
            trigger_config = output_config.get(trigger_name, {})
            threshold = trigger_config.get('threshold', 0.8)
            
            if prob >= threshold:
                print(f"🎯 Trigger de sortie activé: {trigger_name} (prob: {prob:.2f}, value: {value})")
                
                # Actions spécifiques par trigger
                try:
                    # === Triggers d'actions existantes ===
                    if trigger_name == "give_gold" and character_key == "martine":
                        await self._give_gold(value if isinstance(value, (int, float)) and value > 0 else 10)
                    elif trigger_name == "give_cookies" and character_key == "roberte":
                        await self._give_cookies(value if isinstance(value, (int, float)) and value > 0 else 3)
                    elif trigger_name == "sell_fabric" and character_key == "azzedine":
                        await self._sell_fabric()
                    elif trigger_name == "fix_mongolfière" and character_key == "claude":
                        await self._repair_balloon()
                    
                    # === 🆕 Nouveaux triggers d'alcool ===
                    elif trigger_name == "give_alcool":
                        if character_key == "claude":
                            await self._give_alcool(value if isinstance(value, (int, float)) and value > 0 else 2)
                    elif character_key == "martine":  # 🆕 Gestion pour Martine
                        await self._give_alcool(value if isinstance(value, (int, float)) and value > 0 else 1)
                        # 🆕 Martine devient saoule
                        CHARACTERS[character_key]["personality"]['current_alcohol_level'] = "drunk"
                        print(f"🍷 Martine a bu et est maintenant saoule!")
                    
                    # === 🆕 Triggers de quêtes ===
                    elif trigger_name.startswith("quest_"):
                        await self._discover_quest(trigger_name, character_key, prob)
                    
                    # === 🆕 Triggers de réparation ===
                    elif trigger_name == "repair_montgolfiere":
                        await self._repair_balloon()
                        # 🆕 Complétion de la quête cookies pour Claude
                        CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_cookies_for_claude"]["completed"] = True
                        CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_cookies_for_claude"]["active"] = False
                        print("✅ Quête 'find_cookies_for_claude' complétée!")
                        # Vérifier si la quête principale peut être complétée
                        await self._check_main_quest_completion()
                    elif trigger_name == "fabric_repair":
                        await self._sew_fabric()
                        # 🆕 Complétion de la quête or pour Azzedine
                        CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_gold_for_azzedine"]["completed"] = True
                        CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_gold_for_azzedine"]["active"] = False
                        print("✅ Quête 'find_gold_for_azzedine' complétée!")
                        # Vérifier si la quête principale peut être complétée
                        await self._check_main_quest_completion()
                        
                except Exception as e:
                    print(f"❌ Erreur lors de l'exécution du trigger {trigger_name}: {e}")
        
        # Enregistrement de l'événement avec session_id
        try:
            event = {
                "timestamp": time.time(),
                "session_id": game_state.get("current_session_id"),
                "type": "conversation",  # 🆕 Ajout du type pour cohérence
                "character": character_key,
                "user_message": user_message,
                "response_summary": result.get('response', '')[:100] + "...",
                "complexity": result.get('complexity_level', 'unknown'),
                "rag_used": bool(result.get('rag_results', [])),
                "memory_summarized": result.get('memory_summarized', False),
                "output_triggers": output_triggers,
                "fallback_mode": result.get('fallback_info', {}).get('reason') is not None,
                "emergency_fallback": result.get('emergency_fallback', False)
            }
            game_state["game_events"].append(event)
            
            # Limite l'historique des événements
            if len(game_state["game_events"]) > 50:
                game_state["game_events"] = game_state["game_events"][-50:]
        except Exception as e:
            print(f"⚠️ Erreur lors de l'enregistrement de l'événement: {e}")
    
    # === 🆕 Nouvelles fonctions d'action ===
    
    async def _give_alcool(self, amount: int = 2):
        """Donne de l'alcool au joueur."""
        CURRENT_PLAYER_DATA["player_stats"]["alcool"] += amount
        game_state["player_alcool"] = CURRENT_PLAYER_DATA["player_stats"]["alcool"]
        print(f"🍷 +{amount} alcool! Total: {CURRENT_PLAYER_DATA['player_stats']['alcool']}")
    
    async def _discover_quest(self, quest_trigger: str, character_key: str, probability: float):
        """Découvre une quête basée sur le trigger."""
        quest_discovered = False
        
        if quest_trigger == "quest_main_001_claude":
            if not CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_cookies_for_claude"]["discovered"]:
                CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_cookies_for_claude"]["discovered"] = True
                CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_cookies_for_claude"]["active"] = True
                quest_discovered = True
                print("🎯 Quête découverte: Trouver des cookies pour Claude")
        
        elif quest_trigger == "quest_main_001_azzedine":
            if not CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_gold_for_azzedine"]["discovered"]:
                CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_gold_for_azzedine"]["discovered"] = True
                CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_gold_for_azzedine"]["active"] = True
                quest_discovered = True
                print("🎯 Quête découverte: Trouver de l'or pour Azzedine")
        
        elif quest_trigger == "quest_side_001":
            if not CURRENT_PLAYER_DATA["quests"]["side_quests"]["find_island_treasure"]["discovered"]:
                CURRENT_PLAYER_DATA["quests"]["side_quests"]["find_island_treasure"]["discovered"] = True
                CURRENT_PLAYER_DATA["quests"]["side_quests"]["find_island_treasure"]["active"] = True
                quest_discovered = True
                print("🎯 Quête découverte: Découvrir le trésor de l'île")
        
        elif quest_trigger == "quest_side_001_position":
            # Révélation de l'emplacement du trésor
            treasure_quest = CURRENT_PLAYER_DATA["quests"]["side_quests"]["find_island_treasure"]
            if treasure_quest["discovered"] and not treasure_quest["completion_conditions"]["treasure_location_discovered"]:
                treasure_quest["completion_conditions"]["treasure_location_discovered"] = True
                treasure_quest["progress"] = 1
                quest_discovered = True
                print("🗺️ Emplacement du trésor révélé!")
        
        if quest_discovered:
            # Met à jour les événements de jeu pour notifier la découverte
            game_state["game_events"].append({
                "timestamp": time.time(),
                "session_id": game_state.get("current_session_id"),
                "type": "quest_discovery",
                "quest_trigger": quest_trigger,
                "character": character_key,
                "probability": probability
            })
    
    async def _sew_fabric(self):
        """Coud le tissu de la montgolfière."""
        if CURRENT_PLAYER_DATA["player_stats"]["fabric"] >= 1:
            CURRENT_PLAYER_DATA["player_stats"]["fabric"] -= 1
            CURRENT_PLAYER_DATA["montgolfiere_status"]["fabric_sewn"] = True
            
            # Vérifie si la montgolfière est complètement réparée
            if (CURRENT_PLAYER_DATA["montgolfiere_status"]["motor_repaired"] and 
                CURRENT_PLAYER_DATA["montgolfiere_status"]["fabric_sewn"]):
                CURRENT_PLAYER_DATA["montgolfiere_status"]["fully_operational"] = True
                CURRENT_PLAYER_DATA["quests"]["main_quests"]["repair_montgolfiere"]["completed"] = True
            
            sync_game_state_with_player_data()
            print(f"🧵 Tissu cousu! Montgolfière: {'✅ Réparée' if CURRENT_PLAYER_DATA['montgolfiere_status']['fully_operational'] else '🔧 En cours'}")
    
    # === Fonctions existantes mises à jour ===
    
    async def _give_gold(self, amount: int = 10):
        """Donne de l'or au joueur."""
        CURRENT_PLAYER_DATA["player_stats"]["gold"] += amount
        game_state["player_gold"] = CURRENT_PLAYER_DATA["player_stats"]["gold"]
        print(f"💰 +{amount} or! Total: {CURRENT_PLAYER_DATA['player_stats']['gold']}")
        # Découverte de la sous-quête si pas encore découverte
        if not CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_gold_for_azzedine"]["discovered"]:
            CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_gold_for_azzedine"]["discovered"] = True
    
    async def _give_cookies(self, amount: int = 3):
        """Donne des cookies au joueur."""
        CURRENT_PLAYER_DATA["player_stats"]["cookies"] += amount
        game_state["player_cookies"] = CURRENT_PLAYER_DATA["player_stats"]["cookies"]
        print(f"🍪 +{amount} cookies! Total: {CURRENT_PLAYER_DATA['player_stats']['cookies']}")
        # Découverte de la sous-quête si pas encore découverte
        if not CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_cookies_for_claude"]["discovered"]:
            CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_cookies_for_claude"]["discovered"] = True
    
    async def _sell_fabric(self, cost: int = 15):
        """Vend du tissu au joueur."""
        if CURRENT_PLAYER_DATA["player_stats"]["gold"] >= cost:
            CURRENT_PLAYER_DATA["player_stats"]["gold"] -= cost
            CURRENT_PLAYER_DATA["player_stats"]["fabric"] += 1
            game_state["player_gold"] = CURRENT_PLAYER_DATA["player_stats"]["gold"]
            game_state["player_fabric"] = CURRENT_PLAYER_DATA["player_stats"]["fabric"]
            print(f"🧶 Tissu acheté pour {cost} or! Or: {CURRENT_PLAYER_DATA['player_stats']['gold']}, Tissu: {CURRENT_PLAYER_DATA['player_stats']['fabric']}")
    
    async def _repair_balloon(self):
        """Répare la montgolfière."""
        cookies_needed = 5
        fabric_needed = 1
        
        if (CURRENT_PLAYER_DATA["player_stats"]["cookies"] >= cookies_needed and 
            CURRENT_PLAYER_DATA["player_stats"]["fabric"] >= fabric_needed):
            
            CURRENT_PLAYER_DATA["player_stats"]["cookies"] -= cookies_needed
            CURRENT_PLAYER_DATA["player_stats"]["fabric"] -= fabric_needed
            CURRENT_PLAYER_DATA["montgolfiere_status"]["motor_repaired"] = True
            CURRENT_PLAYER_DATA["montgolfiere_status"]["fabric_sewn"] = True
            CURRENT_PLAYER_DATA["montgolfiere_status"]["fully_operational"] = True
            
            # Mise à jour des quêtes
            CURRENT_PLAYER_DATA["quests"]["main_quests"]["repair_montgolfiere"]["completed"] = True
            CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_cookies_for_claude"]["completed"] = True
            CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_gold_for_azzedine"]["completed"] = True
            
            # Synchronisation
            sync_game_state_with_player_data()
            
            print(f"🎈 Montgolfière complètement réparée! Vous pouvez repartir!")

    async def _check_main_quest_completion(self):
        """Vérifie et met à jour la complétion de la quête principale."""
        # Vérifier si les deux sous-quêtes sont complétées
        cookies_completed = CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_cookies_for_claude"]["completed"]
        gold_completed = CURRENT_PLAYER_DATA["quests"]["sub_quests"]["find_gold_for_azzedine"]["completed"]
        
        if cookies_completed and gold_completed:
            # Marquer la quête principale comme complétée
            CURRENT_PLAYER_DATA["quests"]["main_quests"]["repair_montgolfiere"]["completed"] = True
            CURRENT_PLAYER_DATA["quests"]["main_quests"]["repair_montgolfiere"]["active"] = False
            CURRENT_PLAYER_DATA["quests"]["main_quests"]["repair_montgolfiere"]["progress"] = 2  # Max progress
            
            # Marquer la montgolfière comme complètement réparée
            CURRENT_PLAYER_DATA["montgolfiere_status"]["fully_operational"] = True
            sync_game_state_with_player_data()
            
            print("🎉 QUÊTE PRINCIPALE COMPLÉTÉE! La montgolfière est entièrement réparée!")
            
            # Ajouter un événement spécial
            game_state["game_events"].append({
                "timestamp": time.time(),
                "session_id": game_state.get("current_session_id"),
                "type": "quest_completion",
                "quest_id": "repair_montgolfiere",
                "description": "Quête principale complétée - Montgolfière réparée!"
            })
            
            # 🆕 Sauvegarde automatique lors de la complétion de la quête principale
            if game_state.get("current_session_id"):
                save_complete_session(game_state["current_session_id"])
                print("💾 Sauvegarde automatique complète déclenchée par complétion de quête!")


def create_character_avatar(emoji: str, size: int = 60, active: bool = False) -> Image.Image:
    """Crée un avatar circulaire pour un personnage."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Couleur du fond selon l'état
    if active:
        bg_color = (100, 200, 100, 220)  # Vert actif
        border_color = (50, 150, 50, 255)
    else:
        bg_color = (255, 255, 255, 200)  # Blanc normal
        border_color = (100, 100, 100, 255)
    
    # Fond coloré circulaire
    margin = 5
    draw.ellipse([margin, margin, size-margin, size-margin], 
                fill=bg_color, outline=border_color, width=3)
    
    # Emoji au centre
    font_size = size // 2
    try:
        text_bbox = draw.textbbox((0, 0), emoji)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (size - text_width) // 2
        y = (size - text_height) // 2
        
        draw.text((x, y), emoji, fill=(0, 0, 0, 255))
    except:
        draw.text((size//4, size//4), "?", fill=(0, 0, 0, 255))
    
    return img


def load_map_image(map_path: str = "data/img/board.png") -> Image.Image:
    """Charge l'image de la carte ou crée une carte placeholder."""
    try:
        return Image.open(map_path)
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la carte: {str(e)}")
        # Crée une image placeholder
        img = Image.new('RGB', (1000, 700), color='lightblue')
        draw = ImageDraw.Draw(img)
        draw.text((400, 350), "Carte non trouvée", fill='black')
        return img


def generate_interactive_map(active_character: str = None) -> Image.Image:
    """Génère la carte avec les personnages positionnés."""
    map_img = load_map_image("data/img/board.png")
    
    # Ajouter les avatars des personnages
    for char_id, char_data in CHARACTERS.items():
        pos = char_data["position"]
        is_active = (char_id == active_character)
        avatar = create_character_avatar(char_data["emoji"], 50, is_active)
        map_img.paste(avatar, (pos["x"]-25, pos["y"]-25), avatar)
    
    # Ajouter l'avatar du joueur avec la montgolfière
    balloon_emoji = "🎈" if not game_state.get("montgolfiere_repaired", False) else "✨"
    player_avatar = create_character_avatar(balloon_emoji, 60)
    player_pos = game_state["player_position"]
    map_img.paste(player_avatar, (player_pos["x"]-30, player_pos["y"]-30), player_avatar)
    
    return map_img


# 🆕 Fonctions de gestion des sessions pour l'interface
def get_session_list() -> List[str]:
    """Récupère la liste des sessions pour le dropdown."""
    sessions = SessionManager.get_available_sessions()
    session_choices = [f"{s['session_id']} - {s['display_name']}" for s in sessions]
    
    # Ajoute l'option par défaut si la liste est vide
    if not session_choices:
        session_choices = ["Aucune session disponible"]
    
    return session_choices


def handle_session_selection(session_choice: str) -> Tuple[str, gr.update, str, Image.Image, gr.update, str, str]:
    """
    Gère la sélection d'une session existante.
    
    Returns:
        message, session_selection_update, session_info, map_image, game_interface_update, game_status, memory_info
    """
    if not session_choice or session_choice in ["Sélectionnez une session...", "Aucune session disponible"]:
        return (
            "Veuillez sélectionner une session valide.",
            gr.update(visible=True),    # session_selection_container visible
            "",
            generate_interactive_map(),
            gr.update(visible=False),   # game_interface_container caché
            "",
            ""
        )
    
    # Extraction du session_id
    session_id = session_choice.split(" - ")[0]
    
    # Chargement de la session
    if SessionManager.load_session(session_id):
        message = f"✅ Session chargée: {session_id}"
        session_info = f"**Session active:** {game_state['session_name']}\n**ID:** {game_state['current_session_id']}"
        
        return (
            message,
            gr.update(visible=False),   # Cache la sélection de session
            session_info,
            generate_interactive_map(),
            gr.update(visible=True),    # Montre l'interface de jeu
            get_game_status(),
            get_memory_debug_info()
        )
    else:
        return (
            "❌ Erreur lors du chargement de la session.",
            gr.update(visible=True),    # session_selection_container visible
            "",
            generate_interactive_map(),
            gr.update(visible=False),   # game_interface_container caché
            "",
            ""
        )


def handle_new_session(session_name: str = None) -> Tuple[str, gr.update, str, Image.Image, gr.update, str, str]:
    """
    Crée une nouvelle session.
    
    Returns:
        message, session_selection_update, session_info, map_image, game_interface_update, game_status, memory_info
    """
    if not session_name or session_name.strip() == "":
        session_name = None  # Utilisera le nom par défaut
    
    session_id = SessionManager.create_new_session(session_name)
    
    message = f"✅ Nouvelle session créée: {session_id}"
    session_info = f"**Session active:** {game_state['session_name']}\n**ID:** {game_state['current_session_id']}"
    
    return (
        message,
        gr.update(visible=False),
        session_info,
        generate_interactive_map(),
        gr.update(visible=False), 
        get_game_status(),
        get_memory_debug_info(),
        True  # show_intro_screen: actif
    )


def handle_map_click(evt: gr.SelectData) -> Tuple[str, bool, str, Image.Image, bool]:
    """Gère les clics sur la carte."""
    # Vérification de session
    if not game_state.get("session_initialized", False):
        return "❌ Aucune session active. Veuillez d'abord sélectionner ou créer une session.", False, "", generate_interactive_map(), True
    
    if not evt.index:
        return "Cliquez sur un personnage pour lui parler!", False, "", generate_interactive_map(), True
    
    click_x, click_y = evt.index
    
    # Vérifier si le clic est proche d'un personnage
    clicked_character = None
    min_distance = float('inf')
    
    for char_id, char_data in CHARACTERS.items():
        pos = char_data["position"]
        distance = math.sqrt((click_x - pos["x"])**2 + (click_y - pos["y"])**2)
        
        if distance < 40 and distance < min_distance:
            min_distance = distance
            clicked_character = char_id
    
    if clicked_character:
        game_state["current_character"] = clicked_character
        game_state["chat_open"] = True
        game_state["chat_locked"] = False
        game_state["conversation_ending"] = False
        game_state["last_bye_score"] = 0.0
        
        char_data = CHARACTERS[clicked_character]
        welcome_message = f"Vous approchez de {char_data['emoji']} {char_data['name']} ({char_data['role']})"
        
        # Générer la carte avec le personnage actif mis en évidence
        updated_map = generate_interactive_map(clicked_character)
        
        return welcome_message, True, clicked_character, updated_map, False
    else:
        return "Cliquez sur un personnage pour lui parler!", False, "", generate_interactive_map(), True


def chat_interface(message: str, history: List[Dict[str, str]], character_id: str) -> Tuple[List[Dict[str, str]], str, bool]:
    """Interface de chat avec un personnage (format messages pour Gradio 5+)."""
    
    # Vérification de session
    if not game_state.get("session_initialized", False):
        error_msg = "❌ Aucune session active."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, "", False
    
    # Vérification du verrouillage du chat
    if game_state.get("chat_locked", False):
        return history, "", True
    
    if not message.strip() or not character_id:
        return history, "", False
    
    # Fonction synchrone qui lance la fonction async
    def run_async_response():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(get_character_response(character_id, message))
        finally:
            loop.close()
    
    # Obtient la réponse du personnage
    character_response = run_async_response()
    
    # Met à jour l'historique d'affichage (format messages)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": character_response})
    
    # Retourne l'état du verrouillage
    return history, "", game_state.get("chat_locked", False)


def close_chat() -> Tuple[bool, List, str, Image.Image, bool, bool]:
    """Ferme la fenêtre de chat."""
    game_state["chat_open"] = False
    game_state["current_character"] = None
    game_state["chat_locked"] = False
    game_state["conversation_ending"] = False
    # Sauvegarde lors de la fermeture manuelle
    if game_state.get("current_session_id"):
        save_player_data_for_session(CURRENT_PLAYER_DATA, game_state["current_session_id"])
    return False, [], "", generate_interactive_map(), True, False


# 🆕 Fonction mise à jour pour inclure l'alcool
def get_game_status() -> str:
    """Retourne l'état actuel du jeu."""
    if not CURRENT_PLAYER_DATA:
        return "❌ Aucune session active"
    
    repair_status = "✅ Réparée" if CURRENT_PLAYER_DATA["montgolfiere_status"]["fully_operational"] else "❌ Endommagée"
    
    # Calcul du temps de jeu
    play_time = int(time.time() - game_state["start_time"])
    play_time_str = f"{play_time // 60}m {play_time % 60}s"
    
    # Système de dialogue actif
    dialogue_system = "🤖 Agents + Mémoire" if AGENTS_AVAILABLE else ("📚 RAG Basique" if RAG_AVAILABLE else "❌ Aucun")
    
    # Informations de session
    session_info = ""
    if game_state.get("session_initialized", False):
        session_info = f"\n\n**Session:**\n- 📋 {game_state.get('session_name', 'Sans nom')}\n- 🔗 {game_state.get('current_session_id', 'Aucun ID')}"
    else:
        session_info = "\n\n**Session:** ❌ Aucune session active"
    
    # Informations de mémoire
    memory_info = ""
    if game_state["memory_stats"]:
        total_messages = sum(stats.get('total_messages', 0) for stats in game_state["memory_stats"].values())
        total_summaries = sum(stats.get('summaries', 0) for stats in game_state["memory_stats"].values())
        memory_info = f"\n\n**Mémoire:**\n- 💬 Messages: {total_messages}\n- 📝 Résumés: {total_summaries}"
    
    # 🆕 Ajout de l'alcool dans l'affichage des ressources
    status = f"""## 🎮 État du Jeu
    
**Ressources:**
- 💰 Or: {CURRENT_PLAYER_DATA['player_stats']['gold']}
- 🍪 Cookies: {CURRENT_PLAYER_DATA['player_stats']['cookies']}
- 🧶 Tissu: {CURRENT_PLAYER_DATA['player_stats']['fabric']}
- 🍷 Alcool: {CURRENT_PLAYER_DATA['player_stats']['alcool']}

**Montgolfière:** {repair_status}

**Temps de jeu:** {play_time_str}

**Système:** {dialogue_system}{session_info}{memory_info}

**Objectif:** Réparer votre montgolfière pour quitter l'île !
"""
    return status


def get_quests_info() -> str:
    """Retourne les informations sur les quêtes."""
    if not CURRENT_PLAYER_DATA:
        return "❌ Aucune session active"
    
    quests_text = "## 🎯 Quêtes\n\n"
    
    # Quêtes principales
    quests_text += "**Quêtes principales:**\n"
    for quest_id, quest in CURRENT_PLAYER_DATA["quests"]["main_quests"].items():
        if quest.get("discovered", False):
            if quest.get("completed", False):
                status = "🎉"  # Emoji spécial pour les quêtes principales complétées
            elif quest.get("active", False):
                status = "🔄"
            else:
                status = "⏸️"
            quests_text += f"{status} {quest.get('title', quest_id)}\n"
    
    # Sous-quêtes
    sub_quests_discovered = [q for q in CURRENT_PLAYER_DATA["quests"]["sub_quests"].values() if q.get("discovered", False)]
    if sub_quests_discovered:
        quests_text += "\n**Sous-quêtes:**\n"
        for quest in sub_quests_discovered:
            status = "✅" if quest.get("completed", False) else "🔄" if quest.get("active", False) else "⏸️"
            quests_text += f"{status} {quest.get('title', 'Quête inconnue')}\n"
    
    # Quêtes annexes
    side_quests_discovered = [q for q in CURRENT_PLAYER_DATA["quests"]["side_quests"].values() if q.get("discovered", False)]
    if side_quests_discovered:
        quests_text += "\n**Quêtes annexes:**\n"
        for quest in side_quests_discovered:
            status = "✅" if quest.get("completed", False) else "🔄" if quest.get("active", False) else "⏸️"
            quests_text += f"{status} {quest.get('title', 'Quête inconnue')}\n"
    
    if len(sub_quests_discovered) == 0 and len(side_quests_discovered) == 0:
        quests_text += "\n*Explorez et parlez aux habitants pour découvrir de nouvelles quêtes!*"
    
    return quests_text


def get_memory_debug_info() -> str:
    """Retourne les informations de debug sur la mémoire."""
    if not game_state["memory_stats"]:
        return "## 🧠 Mémoire\n\nAucune conversation active."
    
    debug_text = "## 🧠 État de la Mémoire\n\n"
    
    for char_id, stats in game_state["memory_stats"].items():
        char_name = CHARACTERS[char_id]['name']
        debug_text += f"**{char_name}:**\n"
        debug_text += f"- Messages: {stats.get('total_messages', 0)}\n"
        debug_text += f"- Résumés: {stats.get('summaries', 0)}\n"
        debug_text += f"- Checkpoints: {stats.get('checkpoints', 0)}\n"
        debug_text += f"- Dernière activité: {stats.get('last_activity', 'N/A')}\n\n"
    
    # Configuration mémoire
    memory_config = config.get_memory_config()
    debug_text += f"\n**Configuration:**\n"
    debug_text += f"- Auto-résumé après: {memory_config['max_messages_without_summary']} msgs\n"
    debug_text += f"- Messages gardés: {memory_config['keep_recent_messages']}\n"
    debug_text += f"- Sauvegarde auto: {'✅' if memory_config['auto_backup_messages'] else '❌'}\n"
    
    # Informations de session
    if game_state.get("session_initialized", False):
        debug_text += f"\n**Session actuelle:**\n"
        debug_text += f"- ID: {game_state.get('current_session_id', 'N/A')}\n"
        debug_text += f"- Nom: {game_state.get('session_name', 'N/A')}\n"
        
        # Chemin du fichier de sauvegarde
        if game_state.get('current_session_id'):
            save_path = get_player_session_path(game_state['current_session_id'])
            save_exists = os.path.exists(save_path)
            debug_text += f"- Fichier: {'✅' if save_exists else '❌'} {save_path}\n"
    
    return debug_text


def get_debug_info() -> str:
    """Retourne les informations de debug."""
    if not game_state["game_events"]:
        return "## 🐛 Debug\n\nAucun événement enregistré."
    
    recent_events = game_state["game_events"][-5:]
    
    debug_text = "## 🐛 Debug - Derniers Événements\n\n"
    
    # Statut du système
    if graph_manager and hasattr(graph_manager, 'graph_manager'):
        manager_status = graph_manager.graph_manager.get_status() if graph_manager.graph_manager else {}
        debug_text += f"**Statut système:**\n"
        debug_text += f"- Base de données: {'✅' if manager_status.get('database_available', False) else '❌'}\n"
        debug_text += f"- Checkpointer: {'✅' if manager_status.get('checkpointer_enabled', False) else '❌'}\n"
        debug_text += f"- Mode fallback: {'⚠️ Oui' if manager_status.get('fallback_mode', False) else '✅ Non'}\n"
        debug_text += f"- Graphes créés: {manager_status.get('graphs_created', 0)}\n\n"
    
    # Informations de session
    debug_text += f"**Session:**\n"
    debug_text += f"- Initialisée: {'✅' if game_state.get('session_initialized', False) else '❌'}\n"
    debug_text += f"- ID actuel: {game_state.get('current_session_id', 'Aucun')}\n\n"
    
    # Informations sur les fichiers de session
    if CURRENT_PLAYER_DATA:
        debug_text += f"**Sauvegarde:**\n"
        debug_text += f"- Données chargées: ✅\n"
        debug_text += f"- Dernière MAJ: {CURRENT_PLAYER_DATA.get('meta', {}).get('last_updated', 'N/A')}\n"
        debug_text += f"- Nb sauvegardes: {CURRENT_PLAYER_DATA.get('meta', {}).get('save_count', 0)}\n\n"
    
    debug_text += "**Événements récents:**\n"
    
    for i, event in enumerate(recent_events, 1):
        timestamp = datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S")
        session_id = event.get("session_id", "N/A")
        event_type = event.get("type", "conversation")
        character = event.get("character", "Unknown")
        
        debug_text += f"**{i}. {timestamp}** - {character.title()} (Session: {session_id})\n"
        
        # Gestion différente selon le type d'événement
        if event_type == "quest_discovery":
            debug_text += f"   - Type: 🎯 Découverte de quête\n"
            debug_text += f"   - Trigger: {event.get('quest_trigger', 'N/A')}\n"
            debug_text += f"   - Probabilité: {event.get('probability', 0):.2f}\n"
        else:
            # Événement de conversation normale
            user_msg = event.get('user_message', 'Message non disponible')
            debug_text += f"   - Message: {user_msg[:50]}...\n"
            debug_text += f"   - Complexité: {event.get('complexity', 'N/A')}\n"
            debug_text += f"   - RAG: {'✅' if event.get('rag_used', False) else '❌'}\n"
            debug_text += f"   - Résumé mémoire: {'✅' if event.get('memory_summarized', False) else '❌'}\n"
        # Informations sur les modes de fallback (seulement pour les conversations)
        if event_type != "quest_discovery":
            if event.get('fallback_mode', False):
                debug_text += f"   - ⚠️ Mode fallback actif\n"
            if event.get('emergency_fallback', False):
                debug_text += f"   - 🚨 Mode urgence utilisé\n"
            
            # Affichage des triggers de sortie
            output_triggers = event.get('output_triggers', {})
            if output_triggers and isinstance(output_triggers, dict):
                triggers_str = []
                for k, v in output_triggers.items():
                    if isinstance(v, dict) and 'prob' in v:
                        triggers_str.append(f'{k}({v["prob"]:.2f})')
                    else:
                        triggers_str.append(f'{k}(?)')
                if triggers_str:
                    debug_text += f"   - Triggers: {', '.join(triggers_str)}\n"
        debug_text += "\n"
    
    return debug_text


def reset_game() -> Tuple[List, str, Image.Image, str, str, str, bool, bool, bool, str]:
    """Remet à zéro le jeu tout en gardant la session."""
    global CURRENT_PLAYER_DATA
    
    # Reset des données de personnages
    for char_data in CHARACTERS.values():
        char_data['conversation_history'] = []
        char_data['current_emotion'] = 'neutral'
    
    # Reset des données joueur pour la session courante
    if game_state.get("current_session_id"):
        CURRENT_PLAYER_DATA = load_player_data_for_session(game_state["current_session_id"])
        # Force le reload du template
        with open(PLAYER_TEMPLATE_PATH, "r") as f:
            template_data = json.load(f)
        CURRENT_PLAYER_DATA.update(template_data)
        save_player_data_for_session(CURRENT_PLAYER_DATA, game_state["current_session_id"])
    
    # Reset game_state mais garde la session
    session_id = game_state.get("current_session_id")
    session_name = game_state.get("session_name")
    session_initialized = game_state.get("session_initialized", False)
    
    game_state.update({
        "current_character": None,
        "player_position": BALLOON_POSITION.copy(),
        "chat_open": False,
        "chat_locked": False,
        "conversation_ending": False,
        "game_events": [],
        "start_time": time.time(),
        "memory_stats": {},
        "last_bye_score": 0.0,
        # Préserve les informations de session
        "current_session_id": session_id,
        "session_name": session_name,
        "session_initialized": session_initialized
    })
    
    sync_game_state_with_player_data()
    
    session_info = ""
    if session_initialized:
        session_info = f"**Session préservée:** {session_name}\n**ID:** {session_id}"
    
    return (
        [],  # chatbot history
        get_game_status(),
        generate_interactive_map(),
        get_debug_info(),
        get_memory_debug_info(),
        get_quests_info(),
        True,   # map_visible
        False,  # chat_locked
        session_initialized,  # game_interface_visible
        session_info
    )


def initialize_dialogue_system():
    """Initialise le système de dialogue (agents ou RAG)."""
    global graph_manager, rag_system
    
    if AGENTS_AVAILABLE:
        print("🤖 Initialisation du système d'agents LangGraph avec mémoire avancée...")
        graph_manager = EchoForgeAgentWrapper()
        return True
    elif RAG_AVAILABLE:
        print("📚 Initialisation du système RAG de base...")
        return False
    else:
        print("❌ Aucun système de dialogue disponible!")
        return False


async def get_character_response(character_key: str, user_message: str) -> str:
    """Interface unifiée pour obtenir une réponse de personnage."""
    
    if graph_manager:
        return await graph_manager.get_character_response(character_key, user_message)
    elif rag_system:
        return await rag_system.get_character_response(character_key, user_message)
    else:
        return "❌ Aucun système de dialogue disponible. Veuillez vérifier la configuration."


def create_interface():
    """Crée l'interface Gradio avec agents, mémoire avancée et gestion des sessions."""
    
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald",
        neutral_hue="slate",
    )
    
    with gr.Blocks(theme=theme, title="🎈 EchoForge - Système de Quêtes avec Sessions") as demo:
        
        # Variables d'état pour l'interface
        show_intro_screen = gr.State(True)
        chat_visible = gr.State(False)
        current_char = gr.State("")
        map_visible = gr.State(True)
        chat_locked = gr.State(False)
        game_interface_visible = gr.State(False)
        
        # En-tête avec statut système
        system_info = "🤖 LangGraph + Mémoire" if AGENTS_AVAILABLE else ("📚 RAG" if RAG_AVAILABLE else "❌ Aucun")
        
        # Vérification du statut détaillé
        system_status = ""
        if graph_manager and hasattr(graph_manager, 'graph_manager') and graph_manager.graph_manager:
            status = graph_manager.graph_manager.get_status()
            if status.get('fallback_mode', False):
                system_status = " ⚠️ (Mode fallback - DB indisponible)"
            elif not status.get('checkpointer_enabled', True):
                system_status = " ⚠️ (Sans persistance)"
        elif graph_manager and not graph_manager.initialized:
            system_status = f" ❌ (Erreur: {graph_manager.error_message})"
        
        gr.HTML(f"""
        <div style="text-align: center; padding: 20px;">
            <h1>🎈 EchoForge - Agents Intelligents avec Sessions et Quêtes</h1>
            <h3>Système: {system_info}{system_status} | Sauvegarde par session</h3>
            <p><em>Sélectionnez une session existante ou créez-en une nouvelle pour commencer !</em></p>
        </div>
        """)

        with gr.Column(visible=False) as intro_screen:
            gr.Markdown("""
            ## 🌪️ Tempête en vue...

            Vous étiez seul dans une montgolfière, flottant au-dessus de l’océan.  
            Soudain, une tempête violente vous a emporté vers une île mystérieuse...

            Votre montgolfière est **endommagée** :
            - La toile est **déchirée**
            - Le moteur est **hors service**

            Pour repartir, vous devrez explorer l’île, obtenir du tissu et trouver de l’aide pour réparer votre machine.

            Bonne chance, aventurier.
            """)

            continue_button = gr.Button("🎮 Continuer l'aventure")

        with gr.Column(visible=False) as victory_screen:
            gr.Markdown("""
            ## 🏆 Victoire !

            Grâce à vos efforts, la montgolfière est enfin réparée.  
            Le tissu recousu, le moteur rugit à nouveau.

            Vous quittez l'île, emportant avec vous des souvenirs... et peut-être des secrets.

            **Félicitations !**
            """)
        
        # Interface de sélection de session (visible au démarrage)
        with gr.Column(visible=True) as session_selection_container:
            gr.HTML("<h2>🔗 Gestion des Sessions</h2>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>📂 Charger une session existante</h3>")
                    session_dropdown = gr.Dropdown(
                        choices=get_session_list(),
                        label="Sessions disponibles",
                        interactive=True,
                        allow_custom_value=False  # 🔧 Fixe le warning
                    )
                    load_session_btn = gr.Button("🔄 Charger la session", variant="primary")
                    refresh_sessions_btn = gr.Button("🔄 Actualiser la liste", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.HTML("<h3>➕ Créer une nouvelle session</h3>")
                    new_session_name = gr.Textbox(
                        label="Nom de la session (optionnel)",
                        placeholder="Ex: Ma première aventure",
                        lines=1
                    )
                    create_session_btn = gr.Button("✨ Nouvelle session", variant="primary")
            
            session_status_msg = gr.Markdown("Sélectionnez une session pour commencer à jouer.")
        
        # Informations de session (visible quand une session est active)
        session_info_display = gr.Markdown("", visible=False)
        
        # Interface de jeu principale (masquée au démarrage)
        with gr.Column(visible=False) as game_interface_container:
            with gr.Row():
                # Colonne principale - Carte et Chat
                with gr.Column(scale=2):
                    
                    # Message d'instruction
                    instruction_msg = gr.Markdown(
                        "🗺️ **Cliquez sur un personnage sur la carte pour commencer une conversation !**",
                        visible=True
                    )
                    
                    # Carte interactive
                    map_image = gr.Image(
                        value=generate_interactive_map(),
                        interactive=True,
                        label="🎈 Carte de l'île - Votre montgolfière est endommagée!",
                        show_label=True,
                        height=480,
                        visible=True
                    )
                    
                    # Interface de chat (initialement masquée)
                    with gr.Column(visible=False) as chat_container:
                        
                        character_title = gr.Markdown("## Conversation", visible=False)
                        
                        # 🔧 Chatbot avec format messages pour Gradio 5+
                        chatbot = gr.Chatbot(
                            label="Conversation avec IA avancée et mémoire",
                            height=300,
                            show_label=True,
                            container=True,
                            type="messages"  # 🔧 Fixe le warning deprecated
                        )
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                label="Votre message",
                                placeholder="Tapez votre message... (L'IA se souvient de vos conversations précédentes)",
                                lines=2,
                                scale=4,
                                interactive=True
                            )
                            send_btn = gr.Button("📤 Envoyer", scale=1, variant="primary", interactive=True)
                        
                        with gr.Row():
                            leave_btn = gr.Button("🚪 Partir", variant="secondary")
                            clear_chat_btn = gr.Button("🗑️ Effacer chat", variant="secondary")
                        
                        # Message de fin de conversation
                        end_conversation_msg = gr.Markdown(
                            "**💬 Conversation terminée. Fermeture automatique dans 2 secondes...**",
                            visible=False
                        )
                
                # Colonne latérale - État du jeu et infos
                with gr.Column(scale=1):
                    
                    # État du jeu
                    game_status = gr.Markdown(get_game_status())
                    
                    # Tabs pour les différentes infos
                    with gr.Tabs():
                        with gr.TabItem("🎯 Quêtes"):
                            quests_info = gr.Markdown(get_quests_info())
                        
                        with gr.TabItem("👥 Personnages"):
                            # 🆕 Mise à jour des infos personnages avec les nouveaux triggers
                            personality_info = f"""
**👑 Martine** - Maire  
*Donne de l'or, connaît les secrets, évoque les quêtes*
*Peut donner de l'alcool si vous gagnez sa confiance*

**🔨 Claude** - Forgeron  
*Répare la montgolfière contre des cookies*
*Peut vous donner de l'alcool*

**✂️ Azzedine** - Styliste  
*Vend du tissu contre de l'or*
*Évoque des quêtes d'amélioration*

**👩‍🍳 Roberte** - Cuisinière  
*Donne des cookies pendant ses pauses*
*Évoque des quêtes culinaires*

💡 **IA Avancée:** Les personnages gardent en mémoire vos interactions et détectent automatiquement vos intentions !

🆕 **Nouveaux éléments:**
- 🍷 **Alcool** : Nouvelle ressource obtenue auprès de Claude et Martine
- 🎯 **Triggers de quêtes** : Les personnages évoquent naturellement les quêtes
- 🔄 **Actions conditionnelles** : Certaines actions dépendent de vos relations

🔗 **Sessions:** Vos conversations sont sauvegardées par session (fichier `player_session_XXX.json`)
"""
                            gr.Markdown(personality_info)
                        
                        with gr.TabItem("🧠 Mémoire"):
                            memory_info = gr.Markdown(get_memory_debug_info())
                        
                        with gr.TabItem("🐛 Debug"):
                            debug_info = gr.Markdown(get_debug_info())
                    
                    # Boutons d'action
                    with gr.Column():
                        refresh_btn = gr.Button("🔄 Actualiser État", variant="secondary")
                        reset_btn = gr.Button("🆕 Reset Partie", variant="stop")
                        save_btn = gr.Button("💾 Sauvegarder", variant="primary")
                        
                        # Boutons de gestion de session
                        with gr.Row():
                            change_session_btn = gr.Button("🔄 Changer Session", variant="secondary")
                            session_info_btn = gr.Button("ℹ️ Info Session", variant="secondary")
                        
                        if AGENTS_AVAILABLE:
                            toggle_debug_btn = gr.Button("🐛 Toggle Debug", variant="secondary")
        
        # Fonctions de gestion des événements
        def update_chat_visibility(visible: bool, char_id: str, map_vis: bool, locked: bool):
            """Met à jour la visibilité du chat."""
            if visible and char_id:
                char_data = CHARACTERS[char_id]
                system_type = "🤖 Agent + Mémoire" if AGENTS_AVAILABLE else "📚 RAG"
                title = f"## 💬 {system_type} - {char_data['emoji']} {char_data['name']}"
                
                # Contrôle de l'interactivité selon le verrouillage
                msg_interactive = not locked
                send_interactive = not locked
                
                return {
                    chat_container: gr.update(visible=True),
                    character_title: gr.update(value=title, visible=True),
                    instruction_msg: gr.update(visible=False),
                    map_image: gr.update(visible=not visible),
                    msg: gr.update(interactive=msg_interactive),
                    send_btn: gr.update(interactive=send_interactive)
                }
            else:
                return {
                    chat_container: gr.update(visible=False),
                    character_title: gr.update(visible=False),
                    instruction_msg: gr.update(visible=True),
                    map_image: gr.update(visible=True),
                    msg: gr.update(interactive=True),
                    send_btn: gr.update(interactive=True)
                }
        
        def refresh_all_stats():
            """Actualise toutes les statistiques."""
            return get_game_status(), get_memory_debug_info(), get_debug_info(), get_quests_info()
        
        def manual_save():
            """Sauvegarde manuelle."""
            if game_state.get("current_session_id") and CURRENT_PLAYER_DATA:
                save_player_data_for_session(CURRENT_PLAYER_DATA, game_state["current_session_id"])
                status = get_game_status()
                status += "\n\n💾 **Jeu sauvegardé manuellement !**"
                return status
            else:
                return get_game_status() + "\n\n❌ **Aucune session active pour sauvegarder**"
        
        def check_conversation_end():
            """Vérifie si la conversation doit se terminer automatiquement."""
            if game_state.get("conversation_ending", False):
                import threading
                def delayed_close():
                    time.sleep(2)
                
                threading.Thread(target=delayed_close).start()
                return gr.update(visible=True)
            return gr.update(visible=False)
        
        def check_victory_condition():
            if CURRENT_PLAYER_DATA and CURRENT_PLAYER_DATA["quests"]["main_quests"]["repair_montgolfiere"]["completed"]:
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

        
        def update_session_dropdown():
            """Met à jour la liste des sessions."""
            return gr.update(choices=get_session_list())
        
        def show_session_info():
            """Affiche les informations de la session actuelle."""
            if game_state.get("session_initialized", False):
                session_id = game_state.get('current_session_id', 'Aucun')
                save_path = get_player_session_path(session_id) if session_id != 'Aucun' else 'N/A'
                save_exists = os.path.exists(save_path) if save_path != 'N/A' else False
                
                info = f"""**Session active:**
- **Nom:** {game_state.get('session_name', 'Sans nom')}
- **ID:** {session_id}
- **Temps de jeu:** {int(time.time() - game_state["start_time"]) // 60}m {int(time.time() - game_state["start_time"]) % 60}s

**Fichiers:**
- **Sauvegarde:** {'✅' if save_exists else '❌'} `{save_path}`

**Statistiques:**
- **Événements:** {len(game_state["game_events"])}
- **Personnages rencontrés:** {len(game_state["memory_stats"])}
"""
                return info
            else:
                return "❌ Aucune session active"
        
        def return_to_session_selection():
            """Retourne à la sélection de session."""
            return (
                True,   # session_selection_visible
                False,  # game_interface_visible
                "Sélectionnez une nouvelle session ou rechargez la session actuelle.",
                ""      # session_info_display
            )
        
        # Connexions des événements - Gestion des sessions
        continue_button.click(
            lambda: (
                gr.update(visible=False),  # intro_screen caché
                gr.update(visible=True),   # game_interface_container visible
                False                      # show_intro_screen à False
            ),
            outputs=[intro_screen, game_interface_container, show_intro_screen]
        )
        
        refresh_sessions_btn.click(
            update_session_dropdown,
            outputs=[session_dropdown]
        )
        
        load_session_btn.click(
            handle_session_selection,
            inputs=[session_dropdown],
            outputs=[session_status_msg, session_selection_container, session_info_display, map_image, game_interface_container, game_status, memory_info]
        ).then(
            refresh_all_stats,
            outputs=[game_status, memory_info, debug_info, quests_info]
        )
        
        create_session_btn.click(
            handle_new_session,
            inputs=[new_session_name],
            outputs=[session_status_msg, session_selection_container, session_info_display, map_image, game_interface_container, game_status, memory_info, show_intro_screen]
        ).then(
            # Cette fonction utilise la valeur de show_intro_screen pour afficher l'intro
            lambda show_intro: gr.update(visible=show_intro),
            inputs=[show_intro_screen],
            outputs=[intro_screen]
        ).then(
            refresh_all_stats,
            outputs=[game_status, memory_info, debug_info, quests_info]
        )
        
        change_session_btn.click(
            return_to_session_selection,
            outputs=[session_selection_container, game_interface_container, session_status_msg, session_info_display]
        ).then(
            update_session_dropdown,
            outputs=[session_dropdown]
        )
        
        session_info_btn.click(
            show_session_info,
            outputs=[session_info_display]
        )
        
        # Connexions des événements - Interface de jeu
        map_image.select(
            handle_map_click,
            outputs=[instruction_msg, chat_visible, current_char, map_image, map_visible]
        ).then(
            update_chat_visibility,
            inputs=[chat_visible, current_char, map_visible, chat_locked],
            outputs=[chat_container, character_title, instruction_msg, map_image, msg, send_btn]
        ).then(
            refresh_all_stats,
            outputs=[game_status, memory_info, debug_info, quests_info]
        )
        
        # Chat interface avec gestion du verrouillage
        msg.submit(
            chat_interface,
            inputs=[msg, chatbot, current_char],
            outputs=[chatbot, msg, chat_locked]
        ).then(
            refresh_all_stats,
            outputs=[game_status, memory_info, debug_info, quests_info]
        ).then(
            check_conversation_end,
            outputs=[end_conversation_msg]
        ).then(
            check_victory_condition,
            outputs=[victory_screen, map_image, instruction_msg]
        ).then(
            update_chat_visibility,
            inputs=[chat_visible, current_char, map_visible, chat_locked],
            outputs=[chat_container, character_title, instruction_msg, map_image, msg, send_btn]
        )
        
        send_btn.click(
            chat_interface,
            inputs=[msg, chatbot, current_char],
            outputs=[chatbot, msg, chat_locked]
        ).then(
            refresh_all_stats,
            outputs=[game_status, memory_info, debug_info, quests_info]
        ).then(
            check_conversation_end,
            outputs=[end_conversation_msg]
        ).then(
            check_victory_condition,
            outputs=[victory_screen, map_image, instruction_msg]
        ).then(
            update_chat_visibility,
            inputs=[chat_visible, current_char, map_visible, chat_locked],
            outputs=[chat_container, character_title, instruction_msg, map_image, msg, send_btn]
        )
        
        leave_btn.click(
            close_chat,
            outputs=[chat_visible, chatbot, current_char, map_image, map_visible, chat_locked]
        ).then(
            update_chat_visibility,
            inputs=[chat_visible, current_char, map_visible, chat_locked],
            outputs=[chat_container, character_title, instruction_msg, map_image, msg, send_btn]
        ).then(
            refresh_all_stats,
            outputs=[game_status, memory_info, debug_info, quests_info]
        )
        
        clear_chat_btn.click(lambda: [], outputs=[chatbot])
        
        refresh_btn.click(
            refresh_all_stats,
            outputs=[game_status, memory_info, debug_info, quests_info]
        )
        
        save_btn.click(
            manual_save,
            outputs=[game_status]
        )
        
        reset_btn.click(
            reset_game,
            outputs=[chatbot, game_status, map_image, debug_info, memory_info, quests_info, map_visible, chat_locked, game_interface_container, session_info_display]
        ).then(
            lambda: (False, ""),
            outputs=[chat_visible, current_char]
        ).then(
            update_chat_visibility,
            inputs=[chat_visible, current_char, map_visible, chat_locked],
            outputs=[chat_container, character_title, instruction_msg, map_image, msg, send_btn]
        )
        
        # Toggle debug button (seulement si agents disponibles)
        if AGENTS_AVAILABLE:
            def toggle_debug_mode():
                """Active/désactive le mode debug."""
                current = os.getenv('ECHOFORGE_DEBUG', 'false').lower()
                new_value = 'false' if current == 'true' else 'true'
                os.environ['ECHOFORGE_DEBUG'] = new_value
                
                status = get_game_status()
                status += f"\n\n🐛 Mode debug: {'✅ Activé' if new_value == 'true' else '❌ Désactivé'}"
                
                return status, get_memory_debug_info(), get_debug_info()
            
            toggle_debug_btn.click(
                toggle_debug_mode,
                outputs=[game_status, memory_info, debug_info]
            )
        
        # Instructions mises à jour
        gr.HTML(f"""
        <div style="text-align: center; padding: 20px; margin-top: 20px; background-color: #f0f0f0; border-radius: 10px;">
            <h4>🎯 Comment jouer avec Sessions et IA avancée</h4>
            <p><strong>🔗 Sessions:</strong> Créez ou chargez une session pour conserver vos progrès et conversations</p>
            <p><strong>💾 Sauvegarde:</strong> Template `player.json` préservé, données par session `player_session_XXX.json`</p>
            <p><strong>🧠 Mémoire:</strong> L'IA se souvient de toutes vos interactions précédentes dans la session</p>
            <p><strong>🎈 Objectif:</strong> Réparez votre montgolfière en parlant aux habitants de l'île</p>
            <p><strong>🎮 Navigation:</strong> Cliquez sur les personnages, suivez les quêtes, explorez !</p>
            <hr>
            <p>🆕 <strong>Nouveautés:</strong></p>
            <p>🍷 <strong>Alcool:</strong> Nouvelle ressource obtenue auprès de Claude et Martine</p>
            <p>🎯 <strong>Triggers avancés:</strong> Les personnages évoquent naturellement les quêtes et proposent de l'aide</p>
            <p>🔄 <strong>Actions conditionnelles:</strong> Certaines actions dépendent de vos relations et possessions</p>
            <p>💡 <strong>Astuce:</strong> Explorez toutes les possibilités de conversation pour découvrir de nouveaux triggers !</p>
            <hr>
            <p>💡 <strong>Astuce:</strong> Utilisez différentes sessions pour explorer différentes stratégies de jeu</p>
            <p>🤖 <strong>IA Avancée:</strong> Les personnages comprennent le contexte et réagissent de façon cohérente</p>
            <p>📁 <strong>Fichiers:</strong> `data/game_data/sessions/player_session_XXX.json` pour chaque session</p>
        </div>
        """)
    
    return demo


def main():
    """Lance l'application avec le système d'agents, mémoire avancée et gestion des sessions."""
    
    print("🎈 Démarrage d'EchoForge avec Sessions et Système de Quêtes...")
    print("=" * 70)
    print(config.debug_info())
    print("=" * 70)
    
    # Vérification des systèmes disponibles
    if AGENTS_AVAILABLE:
        print("✅ Système d'agents LangGraph avec mémoire avancée disponible")
    elif RAG_AVAILABLE:
        print("⚠️ Fallback vers le système RAG de base")
    else:
        print("❌ Aucun système de dialogue disponible")
        print("Vérifiez l'installation des dépendances:")
        print("  pip install langgraph langchain langchain-community")
        return
    
    # Initialisation du système de dialogue
    if not initialize_dialogue_system():
        print("❌ Impossible d'initialiser le système de dialogue.")
        return
    
    print("✅ Système de dialogue initialisé avec succès !")
    print(f"💾 Template joueur: {PLAYER_TEMPLATE['player_stats']}")
    print(f"📁 Dossier sessions: {SESSIONS_DIR}")
    print("🔗 Sessions disponibles:", len(SessionManager.get_available_sessions()))
    print("🆕 Nouvelles fonctionnalités:")
    print("  - 🍷 Ressource alcool ajoutée")
    print("  - 🎯 Nouveaux triggers de quêtes")
    print("  - 🔄 Actions conditionnelles avancées")
    print("🎮 Lancement de l'interface avec gestion des sessions...")
    
    # Création et lancement de l'interface
    demo = create_interface()
    
    # Configuration du lancement
    demo.launch(
        server_name=config.gradio_server_name,
        server_port=config.gradio_server_port,
        share=config.gradio_share,
        debug=config.debug,
        show_error=True
    )


if __name__ == "__main__":
    main()