import gradio as gr
import json
from datetime import datetime
import os
from typing import List, Tuple, Dict
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import math
import asyncio
import time
from langsmith import traceable
from echoforge.utils.config import get_config

config = get_config()

# Import du système d'agents LangGraph avec mémoire avancée
try:
    from echoforge.agents.graphs.character_graph import CharacterGraphManager
    from echoforge.agents.state.character_state import CharacterState
    AGENTS_AVAILABLE = True
    print("✅ Système d'agents LangGraph avec mémoire avancée chargé avec succès!")
except ImportError as e:
    print(f"⚠️ Erreur: Impossible d'importer le système d'agents: {e}")
    print("📝 Utilisation du système RAG de base comme fallback")
    AGENTS_AVAILABLE = False
    CharacterGraphManager = None

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

# Chargement des données
def load_game_data():
    """Charge les données du jeu (personnages et joueur)"""
    # Chargement des personnages
    with open("data/game_data/characters.json", "r") as f:
        characters = json.load(f)
    
    # Chargement des données joueur
    try:
        with open("data/game_data/player.json", "r") as f:
            player_data = json.load(f)
    except FileNotFoundError:
        print("⚠️ Fichier player.json non trouvé, création avec valeurs par défaut")
        player_data = create_default_player_data()
        save_player_data(player_data)
    
    return characters, player_data

def create_default_player_data():
    """Crée des données de joueur par défaut"""
    return {
        "player_stats": {"gold": 0, "cookies": 0, "fabric": 0},
        "montgolfiere_status": {
            "motor_repaired": False,
            "fabric_sewn": False, 
            "fully_operational": False
        },
        "quests": {
            "main_quests": {
                "repair_montgolfiere": {
                    "discovered": True,
                    "completed": False,
                    "active": True
                }
            },
            "sub_quests": {
                "find_cookies_for_claude": {
                    "discovered": False,
                    "completed": False,
                    "active": False
                },
                "find_gold_for_azzedine": {
                    "discovered": False,
                    "completed": False,
                    "active": False
                }
            },
            "side_quests": {
                "find_island_treasure": {
                    "discovered": False,
                    "completed": False,
                    "active": False
                }
            }
        },
        "game_state": {
            "reputation": {"martine": 0, "claude": 0, "azzedine": 0, "roberte": 0}
        }
    }

def save_player_data(player_data):
    """Sauvegarde les données du joueur"""
    try:
        player_data["meta"]["last_updated"] = datetime.utcnow().isoformat()
        with open("data/game_data/player.json", "w") as f:
            json.dump(player_data, f, indent=2, ensure_ascii=False)
        print("💾 Données joueur sauvegardées")
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")

# Chargement initial
CHARACTERS, PLAYER_DATA = load_game_data()

# Position de la montgolfière
BALLOON_POSITION = {"x": 120, "y": 120}

# État global du jeu
game_state = {
    "current_character": None,
    "player_position": BALLOON_POSITION.copy(),
    "chat_open": False,
    "chat_locked": False,  # 🆕 Pour bloquer le chat quand bye>0.9
    "conversation_ending": False,  # 🆕 Pour gérer la fermeture automatique
    "game_events": [],
    "start_time": time.time(),
    "memory_stats": {},
    "last_bye_score": 0.0  # 🆕 Score bye de la dernière réponse
}

# Synchronisation avec les données joueur
def sync_game_state_with_player_data():
    """Synchronise game_state avec player_data"""
    game_state.update({
        "player_gold": PLAYER_DATA["player_stats"]["gold"],
        "player_cookies": PLAYER_DATA["player_stats"]["cookies"], 
        "player_fabric": PLAYER_DATA["player_stats"]["fabric"],
        "montgolfiere_repaired": PLAYER_DATA["montgolfiere_status"]["fully_operational"]
    })

sync_game_state_with_player_data()

# Instances globales
graph_manager = None
rag_system = None

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
        
        try:
            character_data = CHARACTERS[character_key].copy()
            
            # Traitement du message avec l'agent et mémoire avancée
            result = await self.graph_manager.process_message(
                user_message=user_message,
                character_data=character_data,
                thread_id=f"game_conversation_{character_key}"
            )
            
            # Vérification du résultat
            if not result or 'response' not in result:
                return f"❌ Erreur: Réponse invalide du système d'agents"
            
            # Mise à jour des données du personnage
            CHARACTERS[character_key]['conversation_history'] = result.get('conversation_history', [])
            if 'current_emotion' in result:
                CHARACTERS[character_key]['current_emotion'] = result['current_emotion']
            
            # Mise à jour des statistiques de mémoire
            if 'memory_stats' in result:
                game_state["memory_stats"][character_key] = result['memory_stats']
            
            # 🆕 Traitement des triggers de sortie et actions
            await self._process_agent_actions(character_key, result, user_message)
            
            # 🆕 Gestion du trigger bye
            output_triggers = result.get('output_trigger_probs', {})
            if output_triggers and isinstance(output_triggers, dict):
                bye_info = output_triggers.get('bye', {})
                if isinstance(bye_info, dict):
                    bye_score = bye_info.get('prob', 0.0)
                    game_state["last_bye_score"] = bye_score
                    
                    if bye_score > 0.9:
                        game_state["chat_locked"] = True
                        game_state["conversation_ending"] = True
                        # Sauvegarde automatique quand bye est détecté
                        save_player_data(PLAYER_DATA)
                        print(f"💾 Sauvegarde automatique déclenchée par bye (score: {bye_score:.2f})")
            
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
                
                fallback_debug = ""
                if fallback_info:
                    fallback_debug = f"\n⚠️ Mode fallback: {fallback_info.get('reason', 'unknown')}"
                elif emergency_fallback:
                    fallback_debug = f"\n🚨 Mode urgence: {result.get('error_info', {}).get('error', 'unknown')}"
                
                response += f"\n\n🐛 Debug: {complexity} | RAG: {rag_used} | {processing_time:.3f}s{memory_info}{fallback_debug}\n input_probs : {input_prob} \n output_probs : {output_prob}"
            
            return response
            
        except Exception as e:
            print(f"❌ Erreur dans l'agent pour {character_key}: {str(e)}")
            # Fallback d'urgence local
            character_name = CHARACTERS[character_key].get('name', character_key)
            return f"*{character_name} semble troublé*\n\nExcusez-moi, je rencontre des difficultés techniques. Pouvez-vous reformuler votre message ?\n\n🔧 Erreur: {str(e)}"
    
    async def _process_agent_actions(self, character_key: str, result: dict, user_message: str):
        """Traite les actions spéciales basées sur la réponse de l'agent."""
        
        character_data = CHARACTERS[character_key]
        output_triggers = result.get('output_trigger_probs', {})
        
        # Vérification que output_triggers est un dict
        if not isinstance(output_triggers, dict):
            print(f"⚠️ output_triggers invalide pour {character_key}: {type(output_triggers)}")
            output_triggers = {}
        
        # 🆕 Traitement des triggers de sortie avec valeurs
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
                    if trigger_name == "give_gold" and character_key == "martine":
                        await self._give_gold(value if isinstance(value, (int, float)) and value > 0 else 10)
                    elif trigger_name == "give_cookies" and character_key == "roberte":
                        await self._give_cookies(value if isinstance(value, (int, float)) and value > 0 else 3)
                    elif trigger_name == "sell_fabric" and character_key == "azzedine":
                        await self._sell_fabric()
                    elif trigger_name == "fix_mongolfière" and character_key == "claude":
                        await self._repair_balloon()
                except Exception as e:
                    print(f"❌ Erreur lors de l'exécution du trigger {trigger_name}: {e}")
        
        # Enregistrement de l'événement
        try:
            event = {
                "timestamp": time.time(),
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
    
    async def _give_gold(self, amount: int = 10):
        """Donne de l'or au joueur."""
        PLAYER_DATA["player_stats"]["gold"] += amount
        game_state["player_gold"] = PLAYER_DATA["player_stats"]["gold"]
        print(f"💰 +{amount} or! Total: {PLAYER_DATA['player_stats']['gold']}")
        # Découverte de la sous-quête si pas encore découverte
        if not PLAYER_DATA["quests"]["sub_quests"]["find_gold_for_azzedine"]["discovered"]:
            PLAYER_DATA["quests"]["sub_quests"]["find_gold_for_azzedine"]["discovered"] = True
    
    async def _give_cookies(self, amount: int = 3):
        """Donne des cookies au joueur."""
        PLAYER_DATA["player_stats"]["cookies"] += amount
        game_state["player_cookies"] = PLAYER_DATA["player_stats"]["cookies"]
        print(f"🍪 +{amount} cookies! Total: {PLAYER_DATA['player_stats']['cookies']}")
        # Découverte de la sous-quête si pas encore découverte
        if not PLAYER_DATA["quests"]["sub_quests"]["find_cookies_for_claude"]["discovered"]:
            PLAYER_DATA["quests"]["sub_quests"]["find_cookies_for_claude"]["discovered"] = True
    
    async def _sell_fabric(self, cost: int = 15):
        """Vend du tissu au joueur."""
        if PLAYER_DATA["player_stats"]["gold"] >= cost:
            PLAYER_DATA["player_stats"]["gold"] -= cost
            PLAYER_DATA["player_stats"]["fabric"] += 1
            game_state["player_gold"] = PLAYER_DATA["player_stats"]["gold"]
            game_state["player_fabric"] = PLAYER_DATA["player_stats"]["fabric"]
            print(f"🧶 Tissu acheté pour {cost} or! Or: {PLAYER_DATA['player_stats']['gold']}, Tissu: {PLAYER_DATA['player_stats']['fabric']}")
    
    async def _repair_balloon(self):
        """Répare la montgolfière."""
        cookies_needed = 5
        fabric_needed = 1
        
        if (PLAYER_DATA["player_stats"]["cookies"] >= cookies_needed and 
            PLAYER_DATA["player_stats"]["fabric"] >= fabric_needed):
            
            PLAYER_DATA["player_stats"]["cookies"] -= cookies_needed
            PLAYER_DATA["player_stats"]["fabric"] -= fabric_needed
            PLAYER_DATA["montgolfiere_status"]["motor_repaired"] = True
            PLAYER_DATA["montgolfiere_status"]["fabric_sewn"] = True
            PLAYER_DATA["montgolfiere_status"]["fully_operational"] = True
            
            # Mise à jour des quêtes
            PLAYER_DATA["quests"]["main_quests"]["repair_montgolfiere"]["completed"] = True
            PLAYER_DATA["quests"]["sub_quests"]["find_cookies_for_claude"]["completed"] = True
            PLAYER_DATA["quests"]["sub_quests"]["find_gold_for_azzedine"]["completed"] = True
            
            # Synchronisation
            sync_game_state_with_player_data()
            
            print(f"🎈 Montgolfière complètement réparée! Vous pouvez repartir!")


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
    balloon_emoji = "🎈" if not game_state["montgolfiere_repaired"] else "✨"
    player_avatar = create_character_avatar(balloon_emoji, 60)
    player_pos = game_state["player_position"]
    map_img.paste(player_avatar, (player_pos["x"]-30, player_pos["y"]-30), player_avatar)
    
    return map_img


def handle_map_click(evt: gr.SelectData) -> Tuple[str, bool, str, Image.Image, bool]:
    """Gère les clics sur la carte."""
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
        game_state["chat_locked"] = False  # Reset du verrouillage
        game_state["conversation_ending"] = False
        game_state["last_bye_score"] = 0.0
        
        char_data = CHARACTERS[clicked_character]
        welcome_message = f"Vous approchez de {char_data['emoji']} {char_data['name']} ({char_data['role']})"
        
        # Générer la carte avec le personnage actif mis en évidence
        updated_map = generate_interactive_map(clicked_character)
        
        return welcome_message, True, clicked_character, updated_map, False  # map_visible=False
    else:
        return "Cliquez sur un personnage pour lui parler!", False, "", generate_interactive_map(), True


def chat_interface(message: str, history: List[Tuple[str, str]], character_id: str) -> Tuple[List[Tuple[str, str]], str, bool]:
    """Interface de chat avec un personnage."""
    
    # Vérification du verrouillage du chat
    if game_state.get("chat_locked", False):
        return history, "", True  # Chat verrouillé
    
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
    
    # Met à jour l'historique d'affichage
    history.append((message, character_response))
    
    # Retourne l'état du verrouillage
    return history, "", game_state.get("chat_locked", False)


def close_chat() -> Tuple[bool, List, str, Image.Image, bool, bool]:
    """Ferme la fenêtre de chat."""
    game_state["chat_open"] = False
    game_state["current_character"] = None
    game_state["chat_locked"] = False
    game_state["conversation_ending"] = False
    # Sauvegarde lors de la fermeture manuelle
    save_player_data(PLAYER_DATA)
    return False, [], "", generate_interactive_map(), True, False  # chat_visible, map_visible, chat_locked


def get_game_status() -> str:
    """Retourne l'état actuel du jeu."""
    repair_status = "✅ Réparée" if PLAYER_DATA["montgolfiere_status"]["fully_operational"] else "❌ Endommagée"
    
    # Calcul du temps de jeu
    play_time = int(time.time() - game_state["start_time"])
    play_time_str = f"{play_time // 60}m {play_time % 60}s"
    
    # Système de dialogue actif
    dialogue_system = "🤖 Agents + Mémoire" if AGENTS_AVAILABLE else ("📚 RAG Basique" if RAG_AVAILABLE else "❌ Aucun")
    
    # Informations de mémoire
    memory_info = ""
    if game_state["memory_stats"]:
        total_messages = sum(stats.get('total_messages', 0) for stats in game_state["memory_stats"].values())
        total_summaries = sum(stats.get('summaries', 0) for stats in game_state["memory_stats"].values())
        memory_info = f"\n\n**Mémoire:**\n- 💬 Messages: {total_messages}\n- 📝 Résumés: {total_summaries}"
    
    status = f"""## 🎮 État du Jeu
    
**Ressources:**
- 💰 Or: {PLAYER_DATA['player_stats']['gold']}
- 🍪 Cookies: {PLAYER_DATA['player_stats']['cookies']}
- 🧶 Tissu: {PLAYER_DATA['player_stats']['fabric']}

**Montgolfière:** {repair_status}

**Temps de jeu:** {play_time_str}

**Système:** {dialogue_system}{memory_info}

**Objectif:** Réparer votre montgolfière pour quitter l'île !
"""
    return status


def get_quests_info() -> str:
    """Retourne les informations sur les quêtes."""
    quests_text = "## 🎯 Quêtes\n\n"
    
    # Quêtes principales
    quests_text += "**Quêtes principales:**\n"
    for quest_id, quest in PLAYER_DATA["quests"]["main_quests"].items():
        if quest.get("discovered", False):
            status = "✅" if quest.get("completed", False) else "🔄" if quest.get("active", False) else "⏸️"
            quests_text += f"{status} {quest.get('title', quest_id)}\n"
    
    # Sous-quêtes
    sub_quests_discovered = [q for q in PLAYER_DATA["quests"]["sub_quests"].values() if q.get("discovered", False)]
    if sub_quests_discovered:
        quests_text += "\n**Sous-quêtes:**\n"
        for quest in sub_quests_discovered:
            status = "✅" if quest.get("completed", False) else "🔄" if quest.get("active", False) else "⏸️"
            quests_text += f"{status} {quest.get('title', 'Quête inconnue')}\n"
    
    # Quêtes annexes
    side_quests_discovered = [q for q in PLAYER_DATA["quests"]["side_quests"].values() if q.get("discovered", False)]
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
    
    return debug_text


def get_debug_info() -> str:
    """Retourne les informations de debug."""
    if not game_state["game_events"]:
        return "## 🐛 Debug\n\nAucun événement enregistré."
    
    recent_events = game_state["game_events"][-5:]  # 5 derniers événements
    
    debug_text = "## 🐛 Debug - Derniers Événements\n\n"
    
    # Statut du système
    if graph_manager and hasattr(graph_manager, 'graph_manager'):
        manager_status = graph_manager.graph_manager.get_status() if graph_manager.graph_manager else {}
        debug_text += f"**Statut système:**\n"
        debug_text += f"- Base de données: {'✅' if manager_status.get('database_available', False) else '❌'}\n"
        debug_text += f"- Checkpointer: {'✅' if manager_status.get('checkpointer_enabled', False) else '❌'}\n"
        debug_text += f"- Mode fallback: {'⚠️ Oui' if manager_status.get('fallback_mode', False) else '✅ Non'}\n"
        debug_text += f"- Graphes créés: {manager_status.get('graphs_created', 0)}\n\n"
    
    debug_text += "**Événements récents:**\n"
    
    for i, event in enumerate(recent_events, 1):
        timestamp = datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S")
        debug_text += f"**{i}. {timestamp}** - {event['character'].title()}\n"
        debug_text += f"   - Message: {event['user_message'][:50]}...\n"
        debug_text += f"   - Complexité: {event['complexity']}\n"
        debug_text += f"   - RAG: {'✅' if event['rag_used'] else '❌'}\n"
        debug_text += f"   - Résumé mémoire: {'✅' if event.get('memory_summarized', False) else '❌'}\n"
        
        # Informations sur les modes de fallback
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


def reset_game() -> Tuple[List, str, Image.Image, str, str, str, bool, bool]:
    """Remet à zéro le jeu."""
    global PLAYER_DATA
    
    # Reset des données de personnages
    for char_data in CHARACTERS.values():
        char_data['conversation_history'] = []
        char_data['current_emotion'] = 'neutral'
    
    # Reset des données joueur
    PLAYER_DATA = create_default_player_data()
    save_player_data(PLAYER_DATA)
    
    # Reset game_state
    game_state.update({
        "current_character": None,
        "player_position": BALLOON_POSITION.copy(),
        "chat_open": False,
        "chat_locked": False,
        "conversation_ending": False,
        "game_events": [],
        "start_time": time.time(),
        "memory_stats": {},
        "last_bye_score": 0.0
    })
    
    sync_game_state_with_player_data()
    
    return ([], get_game_status(), generate_interactive_map(), 
            get_debug_info(), get_memory_debug_info(), get_quests_info(),
            True, False)  # map_visible=True, chat_locked=False


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


def auto_close_chat():
    """Ferme automatiquement le chat après 2 secondes si conversation_ending=True"""
    if game_state.get("conversation_ending", False):
        time.sleep(2)
        return close_chat()
    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def create_interface():
    """Crée l'interface Gradio avec agents et mémoire avancée."""
    
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald",
        neutral_hue="slate",
    )
    
    with gr.Blocks(theme=theme, title="🎈 EchoForge - Système de Quêtes") as demo:
        
        # Variables d'état pour l'interface
        chat_visible = gr.State(False)
        current_char = gr.State("")
        map_visible = gr.State(True)
        chat_locked = gr.State(False)
        
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
            <h1>🎈 EchoForge - Agents Intelligents avec Système de Quêtes</h1>
            <h3>Système: {system_info}{system_status} | Cliquez sur les personnages pour leur parler</h3>
            <p><em>Réparez votre montgolfière pour quitter l'île mystérieuse !</em></p>
        </div>
        """)
        
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
                    
                    chatbot = gr.Chatbot(
                        label="Conversation avec IA avancée et mémoire",
                        height=300,
                        show_label=True,
                        container=True,
                        bubble_full_width=False
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
                        personality_info = f"""
**👑 Martine** - Maire  
*Donne de l'or, connaît les secrets*

**🔨 Claude** - Forgeron  
*Répare la montgolfière contre des cookies*

**✂️ Azzedine** - Styliste  
*Vend du tissu contre de l'or*

**👩‍🍳 Roberte** - Cuisinière  
*Donne des cookies pendant ses pauses*

💡 **IA Avancée:** Les personnages gardent en mémoire vos interactions et détectent automatiquement vos intentions !
"""
                        gr.Markdown(personality_info)
                    
                    with gr.TabItem("🧠 Mémoire"):
                        memory_info = gr.Markdown(get_memory_debug_info())
                    
                    with gr.TabItem("🐛 Debug"):
                        debug_info = gr.Markdown(get_debug_info())
                
                # Boutons d'action
                with gr.Column():
                    refresh_btn = gr.Button("🔄 Actualiser État", variant="secondary")
                    reset_btn = gr.Button("🆕 Nouveau Jeu", variant="stop")
                    save_btn = gr.Button("💾 Sauvegarder", variant="primary")
                    
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
                    map_image: gr.update(visible=not visible),  # Cache la carte en conversation
                    msg: gr.update(interactive=msg_interactive),
                    send_btn: gr.update(interactive=send_interactive)
                }
            else:
                return {
                    chat_container: gr.update(visible=False),
                    character_title: gr.update(visible=False),
                    instruction_msg: gr.update(visible=True),
                    map_image: gr.update(visible=True),  # Montre la carte
                    msg: gr.update(interactive=True),
                    send_btn: gr.update(interactive=True)
                }
        
        def refresh_all_stats():
            """Actualise toutes les statistiques."""
            return get_game_status(), get_memory_debug_info(), get_debug_info(), get_quests_info()
        
        def manual_save():
            """Sauvegarde manuelle."""
            save_player_data(PLAYER_DATA)
            status = get_game_status()
            status += "\n\n💾 **Jeu sauvegardé manuellement !**"
            return status
        
        def check_conversation_end():
            """Vérifie si la conversation doit se terminer automatiquement."""
            if game_state.get("conversation_ending", False):
                # Déclenche la fermeture automatique après 2 secondes
                import threading
                def delayed_close():
                    time.sleep(2)
                    # Ici on pourrait déclencher un événement pour fermer le chat
                    # mais c'est plus complexe avec Gradio, on laisse l'utilisateur fermer
                
                threading.Thread(target=delayed_close).start()
                return gr.update(visible=True)  # Montre le message de fin
            return gr.update(visible=False)
        
        # Connexions des événements
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
            outputs=[chatbot, game_status, map_image, debug_info, memory_info, quests_info, map_visible, chat_locked]
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
        
        # Instructions
        gr.HTML(f"""
        <div style="text-align: center; padding: 20px; margin-top: 20px; background-color: #f0f0f0; border-radius: 10px;">
            <h4>🎯 Comment jouer avec l'IA avancée et système de quêtes</h4>
            <p><strong>1.</strong> Votre montgolfière 🎈 est endommagée au centre de la carte</p>
            <p><strong>2.</strong> Cliquez sur les personnages pour dialoguer (la carte se cache pendant les conversations)</p>
            <p><strong>3.</strong> L'IA détecte automatiquement vos intentions et les actions des personnages</p>
            <p><strong>4.</strong> Suivez les quêtes dans l'onglet "🎯 Quêtes" pour progresser</p>
            <p><strong>5.</strong> Les conversations se terminent automatiquement quand vous dites au revoir</p>
            <p><strong>6.</strong> Collectez des ressources et réparez votre montgolfière pour gagner !</p>
            <hr>
            <p>🎯 <strong>Quêtes automatiques:</strong> Les quêtes se découvrent et se valident automatiquement selon vos actions</p>
            <p>💾 <strong>Sauvegarde automatique:</strong> Le jeu se sauvegarde automatiquement à chaque fin de conversation</p>
            <p>🤖 <strong>IA Avancée:</strong> Les personnages comprennent vos demandes et réagissent de façon cohérente</p>
        </div>
        """)
    
    return demo


def main():
    """Lance l'application avec le système d'agents et mémoire avancée."""
    
    print("🎈 Démarrage d'EchoForge avec Système de Quêtes...")
    print("=" * 60)
    print(config.debug_info())
    print("=" * 60)
    
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
    print(f"💾 Données joueur chargées: {PLAYER_DATA['player_stats']}")
    print("🎮 Lancement de l'interface avec système de quêtes...")
    
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