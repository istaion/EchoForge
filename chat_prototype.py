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
import json

config = get_config()
# Import du système d'agents LangGraph
try:
    from echoforge.agents.graphs.character_graph import CharacterGraphManager
    from echoforge.agents.state.character_state import CharacterState
    AGENTS_AVAILABLE = True
    print("✅ Système d'agents LangGraph chargé avec succès!")
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

# Données des personnages avec leurs positions sur la carte
with open("data/game_data/characters.json", "r") as f:
    CHARACTERS = json.load(f)

# Position de la montgolfière
BALLOON_POSITION = {"x": 120, "y": 120}

# État global du jeu
game_state = {
    "player_gold": 0,
    "player_cookies": 0,
    "player_fabric": 0,
    "montgolfiere_repaired": False,
    "conversation_history": {},
    "current_character": None,
    "player_position": BALLOON_POSITION.copy(),
    "chat_open": False,
    "game_events": [],
    "start_time": time.time()
}

# Instances globales
graph_manager = None
rag_system = None

class EchoForgeAgentWrapper:
    """Wrapper pour intégrer les agents LangGraph avec le système existant."""
    
    def __init__(self):
        self.graph_manager = CharacterGraphManager() if AGENTS_AVAILABLE else None
        self.initialized = AGENTS_AVAILABLE
        
    async def get_character_response(self, character_key: str, user_message: str) -> str:
        """Obtient une réponse du personnage via le système d'agents."""
        
        if not self.initialized:
            return "❌ Système d'agents non disponible. Veuillez vérifier l'installation."
        
        try:
            character_data = CHARACTERS[character_key].copy()
            
            # Traitement du message avec l'agent
            result = await self.graph_manager.process_message(
                user_message=user_message,
                character_data=character_data,
                thread_id=f"game_conversation_{character_key}"
            )
            
            # Mise à jour des données du personnage
            CHARACTERS[character_key]['conversation_history'] = result.get('conversation_history', [])
            if 'current_emotion' in result:
                CHARACTERS[character_key]['current_emotion'] = result['current_emotion']
            
            # Traitement des actions spéciales basées sur la réponse
            await self._process_agent_actions(character_key, result, user_message)
            
            # Ajout d'informations de debug en mode développement
            response = result['response']
            if os.getenv('ECHOFORGE_DEBUG', 'false').lower() == 'true':
                debug_info = result.get('debug_info', {})
                complexity = result.get('complexity_level', 'unknown')
                input_prob = result.get('input_trigger_probs')
                output_prob = result.get('output_trigger_probs')
                rag_used = bool(result.get('rag_results', []))
                processing_time = debug_info.get('final_stats', {}).get('total_processing_time', 0)
                
                response += f"\n\n🐛 Debug: {complexity} | RAG: {rag_used} | {processing_time:.3f}s \n input_probs : {input_prob} \n output_probs : {output_prob}"
            
            return response
            
        except Exception as e:
            print(f"❌ Erreur dans l'agent pour {character_key}: {str(e)}")
            return f"❌ Erreur de traitement: {str(e)}"
    
    async def _process_agent_actions(self, character_key: str, result: dict, user_message: str):
        """Traite les actions spéciales basées sur la réponse de l'agent."""
        
        character_data = CHARACTERS[character_key]
        response_lower = result['response'].lower()
        
        # Actions spécifiques par personnage
        if character_key == "martine" and character_data.get("can_give_gold"):
            if self._detect_give_action(response_lower, "or"):
                await self._give_gold()
        
        elif character_key == "roberte" and character_data.get("gives_cookies"):
            if self._detect_give_action(response_lower, "cookie"):
                await self._give_cookies()
        
        elif character_key == "azzedine" and character_data.get("sells_fabric"):
            if self._detect_sell_action(response_lower, "tissu"):
                await self._sell_fabric()
        
        elif character_key == "claude" and character_data.get("can_repair"):
            if self._detect_repair_action(response_lower):
                await self._repair_balloon()
        
        # Enregistrement de l'événement
        event = {
            "timestamp": time.time(),
            "character": character_key,
            "user_message": user_message,
            "response_summary": result['response'][:100] + "...",
            "complexity": result.get('complexity_level', 'unknown'),
            "rag_used": bool(result.get('rag_results', []))
        }
        game_state["game_events"].append(event)
        
        # Limite l'historique des événements
        if len(game_state["game_events"]) > 50:
            game_state["game_events"] = game_state["game_events"][-50:]
    
    def _detect_give_action(self, response: str, item: str) -> bool:
        """Détecte les actions de don dans la réponse."""
        give_keywords = ["donne", "offre", "voici", "prends", "cadeau", "pour toi"]
        return any(keyword in response for keyword in give_keywords) and item in response
    
    def _detect_sell_action(self, response: str, item: str) -> bool:
        """Détecte les actions de vente."""
        sell_keywords = ["vends", "achète", "prends", "voici", "commerce", "échange"]
        return any(keyword in response for keyword in sell_keywords) and item in response and game_state["player_gold"] >= 15
    
    def _detect_repair_action(self, response: str) -> bool:
        """Détecte les actions de réparation."""
        repair_keywords = ["répare", "réparer", "fini", "terminé", "réparation"]
        has_repair = any(keyword in response for keyword in repair_keywords)
        has_resources = game_state["player_cookies"] >= 5 and game_state["player_fabric"] >= 1
        return has_repair and has_resources
    
    async def _give_gold(self):
        """Donne de l'or au joueur."""
        if game_state["player_gold"] < 100:
            game_state["player_gold"] += 10
            print(f"💰 martine vous a donné 10 pièces d'or! Total: {game_state['player_gold']}")
    
    async def _give_cookies(self):
        """Donne des cookies au joueur."""
        if game_state["player_cookies"] < 20:
            game_state["player_cookies"] += 3
            print(f"🍪 Roberte vous a donné 3 cookies! Total: {game_state['player_cookies']}")
    
    async def _sell_fabric(self):
        """Vend du tissu au joueur."""
        if game_state["player_gold"] >= 15:
            game_state["player_gold"] -= 15
            game_state["player_fabric"] += 1
            print(f"🧶 Azzedine vous a vendu du tissu pour 15 or! Or: {game_state['player_gold']}, Tissu: {game_state['player_fabric']}")
    
    async def _repair_balloon(self):
        """Répare la montgolfière."""
        if game_state["player_cookies"] >= 5 and game_state["player_fabric"] >= 1:
            game_state["player_cookies"] -= 5
            game_state["player_fabric"] -= 1
            game_state["montgolfiere_repaired"] = True
            print(f"🎈 Claude a réparé votre montgolfière! Vous pouvez repartir!")


class RAGSystemWrapper:
    """Wrapper pour le système RAG de base comme fallback."""
    
    def __init__(self):
        if RAG_AVAILABLE:
            try:
                print("🚀 Initialisation du système RAG...")
                self.rag_system = EchoForgeRAG(
                    data_path="./data",
                    vector_store_path="./vector_stores"
                )
                self._build_vector_stores()
                self.initialized = True
                print("✅ Système RAG initialisé!")
            except Exception as e:
                print(f"❌ Erreur RAG: {e}")
                self.initialized = False
        else:
            self.initialized = False
    
    def _build_vector_stores(self):
        """Construit les vector stores si nécessaire."""
        world_store_path = Path("./vector_stores/world_lore")
        if not world_store_path.exists():
            print("🌍 Construction du vector store du monde...")
            self.rag_system.build_world_vectorstore()
        
        for character_id in CHARACTERS.keys():
            char_store_path = Path(f"./vector_stores/character_{character_id}")
            if not char_store_path.exists():
                print(f"👤 Construction du vector store pour {character_id}...")
                self.rag_system.build_character_vectorstore(character_id)

    @traceable
    async def get_character_response(self, character_key: str, user_message: str) -> str:
        """Obtient une réponse via le système RAG."""
        if not self.initialized:
            return "❌ Système RAG non disponible."
        
        try:
            character_data = CHARACTERS[character_key]
            parsed_input = self.rag_system.parse_actions(user_message)
            
            world_context = self.rag_system.retrieve_world_context(parsed_input.text, top_k=3)
            character_context = self.rag_system.retrieve_character_context(
                parsed_input.text, character_key, top_k=5
            )
            
            conversation_history = self._get_conversation_history_string(character_key)
            
            prompt = self.rag_system.create_character_prompt(
                character_data=character_data,
                world_context=world_context,
                character_context=character_context,
                parsed_input=parsed_input,
                conversation_history=conversation_history
            )
            
            response = self.rag_system.llm.invoke(prompt)
            
            # Sauvegarde dans l'historique
            if character_key not in game_state["conversation_history"]:
                game_state["conversation_history"][character_key] = []
            
            game_state["conversation_history"][character_key].extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response}
            ])
            
            # Traitement des actions
            self._process_character_actions(character_key, response, parsed_input)
            
            return response
            
        except Exception as e:
            return f"❌ Erreur RAG: {str(e)}"
    
    def _get_conversation_history_string(self, character_key: str, max_messages: int = 6) -> str:
        """Récupère l'historique formaté."""
        if character_key not in game_state["conversation_history"]:
            return ""
        
        history = game_state["conversation_history"][character_key]
        recent_history = history[-max_messages:] if len(history) > max_messages else history
        
        formatted_history = []
        for msg in recent_history:
            role = "Joueur" if msg["role"] == "user" else CHARACTERS[character_key]["name"]
            formatted_history.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_history)
    
    def _process_character_actions(self, character_key: str, character_response: str, user_input):
        """Traite les actions du personnage (version RAG)."""
        character_data = CHARACTERS[character_key]
        response_lower = character_response.lower()
        
        # Même logique que le système d'agents mais synchrone
        if character_key == "martine" and character_data.get("can_give_gold"):
            if any(word in response_lower for word in ["donne", "offre", "voici", "prends"]) and "or" in response_lower:
                if game_state["player_gold"] < 100:
                    game_state["player_gold"] += 10
                    print(f"💰 martine vous a donné 10 pièces d'or!")
        
        # Autres actions similaires...


def initialize_dialogue_system():
    """Initialise le système de dialogue (agents ou RAG)."""
    global graph_manager, rag_system
    
    if AGENTS_AVAILABLE:
        print("🤖 Initialisation du système d'agents LangGraph...")
        graph_manager = EchoForgeAgentWrapper()
        return True
    elif RAG_AVAILABLE:
        print("📚 Initialisation du système RAG de base...")
        rag_system = RAGSystemWrapper()
        return rag_system.initialized
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
    
    # Ajouter l'avatar du joueur
    player_avatar = create_character_avatar("🧑", 40)
    player_pos = game_state["player_position"]
    map_img.paste(player_avatar, (player_pos["x"]-20, player_pos["y"]-20), player_avatar)
    
    return map_img


def handle_map_click(evt: gr.SelectData) -> Tuple[str, bool, str, Image.Image]:
    """Gère les clics sur la carte."""
    if not evt.index:
        return "Cliquez sur un personnage pour lui parler!", False, "", generate_interactive_map()
    
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
        
        char_data = CHARACTERS[clicked_character]
        welcome_message = f"Vous approchez de {char_data['emoji']} {char_data['name']} ({char_data['role']})"
        
        # Générer la carte avec le personnage actif mis en évidence
        updated_map = generate_interactive_map(clicked_character)
        
        return welcome_message, True, clicked_character, updated_map
    else:
        return "Cliquez sur un personnage pour lui parler!", False, "", generate_interactive_map()


def chat_interface(message: str, history: List[Tuple[str, str]], character_id: str) -> Tuple[List[Tuple[str, str]], str]:
    """Interface de chat avec un personnage."""
    
    if not message.strip() or not character_id:
        return history, ""
    
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
    
    return history, ""


def close_chat() -> Tuple[bool, List, str, Image.Image]:
    """Ferme la fenêtre de chat."""
    game_state["chat_open"] = False
    game_state["current_character"] = None
    return False, [], "", generate_interactive_map()


def get_game_status() -> str:
    """Retourne l'état actuel du jeu."""
    repair_status = "✅ Réparée" if game_state["montgolfiere_repaired"] else "❌ Endommagée"
    
    # Calcul du temps de jeu
    play_time = int(time.time() - game_state["start_time"])
    play_time_str = f"{play_time // 60}m {play_time % 60}s"
    
    # Système de dialogue actif
    dialogue_system = "🤖 Agents LangGraph" if AGENTS_AVAILABLE else ("📚 RAG Basique" if RAG_AVAILABLE else "❌ Aucun")
    
    status = f"""## 🎮 État du Jeu
    
**Ressources:**
- 💰 Or: {game_state['player_gold']}
- 🍪 Cookies: {game_state['player_cookies']}
- 🧶 Tissu: {game_state['player_fabric']}

**Montgolfière:** {repair_status}

**Temps de jeu:** {play_time_str}

**Système:** {dialogue_system}

**Objectif:** Réparer votre montgolfière pour quitter l'île !
"""
    return status


def get_debug_info() -> str:
    """Retourne les informations de debug."""
    if not game_state["game_events"]:
        return "## 🐛 Debug\n\nAucun événement enregistré."
    
    recent_events = game_state["game_events"][-5:]  # 5 derniers événements
    
    debug_text = "## 🐛 Debug - Derniers Événements\n\n"
    
    for i, event in enumerate(recent_events, 1):
        timestamp = datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S")
        debug_text += f"**{i}. {timestamp}** - {event['character'].title()}\n"
        debug_text += f"   - Message: {event['user_message'][:50]}...\n"
        debug_text += f"   - Complexité: {event['complexity']}\n"
        debug_text += f"   - RAG: {'✅' if event['rag_used'] else '❌'}\n\n"
    
    return debug_text


def reset_game() -> Tuple[List, str, Image.Image, str]:
    """Remet à zéro le jeu."""
    global game_state
    
    # Reset des données de personnages
    for char_data in CHARACTERS.values():
        char_data['conversation_history'] = []
        char_data['current_emotion'] = 'neutral'
    
    game_state = {
        "player_gold": 0,
        "player_cookies": 0,
        "player_fabric": 0,
        "montgolfiere_repaired": False,
        "conversation_history": {},
        "current_character": None,
        "player_position": BALLOON_POSITION.copy(),
        "chat_open": False,
        "game_events": [],
        "start_time": time.time()
    }
    
    return [], get_game_status(), generate_interactive_map(), get_debug_info()


def create_interface():
    """Crée l'interface Gradio avec agents intégrés."""
    
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald",
        neutral_hue="slate",
    )
    
    with gr.Blocks(theme=theme, title="🎈 EchoForge - Agents LangGraph") as demo:
        
        # Variables d'état pour l'interface
        chat_visible = gr.State(False)
        current_char = gr.State("")
        
        # En-tête
        system_info = "🤖 LangGraph" if AGENTS_AVAILABLE else ("📚 RAG" if RAG_AVAILABLE else "❌ Aucun")
        
        gr.HTML(f"""
        <div style="text-align: center; padding: 20px;">
            <h1>🎈 EchoForge - Agents Intelligents</h1>
            <h3>Système: {system_info} | Cliquez sur les personnages pour leur parler</h3>
            <p><em>Personnages avec IA avancée et routage conditionnel RAG</em></p>
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
                    label="Carte de l'île - Cliquez sur les personnages",
                    show_label=True,
                    height=480
                )
                
                # Interface de chat (initialement masquée)
                with gr.Column(visible=False) as chat_container:
                    
                    character_title = gr.Markdown("## Conversation", visible=False)
                    
                    chatbot = gr.Chatbot(
                        label="Conversation avec IA avancée",
                        height=300,
                        show_label=True,
                        container=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Votre message",
                            placeholder="Tapez votre message... (L'IA décidera automatiquement si elle a besoin du RAG)",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("📤 Envoyer", scale=1, variant="primary")
                    
                    with gr.Row():
                        close_chat_btn = gr.Button("🚪 Retour à la carte", variant="secondary")
                        clear_chat_btn = gr.Button("🗑️ Effacer chat", variant="secondary")
            
            # Colonne latérale - État du jeu et infos
            with gr.Column(scale=1):
                
                # État du jeu
                game_status = gr.Markdown(get_game_status())
                
                gr.Markdown("---")
                
                # Guide des personnages
                personality_info = f"""
                ## 👥 Personnages IA
                
                **👑 martine** - Maire  
                *Donne de l'or, connaît les secrets*  
                Traits: Leadership {CHARACTERS['martine']['personality']['traits']['leadership']}, Curiosité {CHARACTERS['martine']['personality']['traits']['curiosity']}
                
                **🔨 Claude** - Forgeron  
                *Répare la montgolfière contre des cookies*  
                Traits: Pragmatisme {CHARACTERS['claude']['personality']['traits']['pragmatism']}, Artisanat {CHARACTERS['claude']['personality']['traits']['craftsmanship']}
                
                **✂️ Azzedine** - Styliste  
                *Vend du tissu contre de l'or*  
                Traits: Créativité {CHARACTERS['azzedine']['personality']['traits']['creativity']}, Perfectionnisme {CHARACTERS['azzedine']['personality']['traits']['perfectionism']}
                
                **👩‍🍳 Roberte** - Cuisinière  
                *Donne des cookies pendant ses pauses*  
                Traits: Générosité {CHARACTERS['roberte']['personality']['traits']['generosity']}, Chaleur {CHARACTERS['roberte']['personality']['traits']['warmth']}
                
                💡 **IA Avancée:** Chaque personnage utilise des agents intelligents qui décident automatiquement quand chercher des informations !
                """
                
                gr.Markdown(personality_info)
                
                gr.Markdown("---")
                
                # Informations de debug (si activées)
                if os.getenv('ECHOFORGE_DEBUG', 'false').lower() == 'true':
                    debug_info = gr.Markdown(get_debug_info(), label="Debug Info")
                
                # Boutons d'action
                with gr.Column():
                    refresh_btn = gr.Button("🔄 Actualiser État", variant="secondary")
                    reset_btn = gr.Button("🆕 Nouveau Jeu", variant="stop")
                    
                    if AGENTS_AVAILABLE:
                        toggle_debug_btn = gr.Button("🐛 Toggle Debug", variant="secondary")
        
        # Fonctions de gestion des événements
        def update_chat_visibility(visible: bool, char_id: str):
            """Met à jour la visibilité du chat."""
            if visible and char_id:
                char_data = CHARACTERS[char_id]
                system_type = "🤖 Agent LangGraph" if AGENTS_AVAILABLE else "📚 RAG"
                title = f"## 💬 {system_type} - {char_data['emoji']} {char_data['name']}"
                return {
                    chat_container: gr.update(visible=True),
                    character_title: gr.update(value=title, visible=True),
                    instruction_msg: gr.update(visible=False)
                }
            else:
                return {
                    chat_container: gr.update(visible=False),
                    character_title: gr.update(visible=False),
                    instruction_msg: gr.update(visible=True)
                }
        
        def reinitialize_system():
            """Réinitialise le système de dialogue."""
            success = initialize_dialogue_system()
            status = get_game_status()
            
            if success:
                status += "\n\n✅ Système réinitialisé avec succès !"
            else:
                status += "\n\n❌ Échec de la réinitialisation."
            
            return status
        
        def toggle_debug_mode():
            """Active/désactive le mode debug."""
            current = os.getenv('ECHOFORGE_DEBUG', 'false').lower()
            new_value = 'false' if current == 'true' else 'true'
            os.environ['ECHOFORGE_DEBUG'] = new_value
            
            status = get_game_status()
            status += f"\n\n🐛 Mode debug: {'✅ Activé' if new_value == 'true' else '❌ Désactivé'}"
            
            return status, get_debug_info() if new_value == 'true' else ""
        
        # Connexions des événements
        map_image.select(
            handle_map_click,
            outputs=[instruction_msg, chat_visible, current_char, map_image]
        ).then(
            update_chat_visibility,
            inputs=[chat_visible, current_char],
            outputs=[chat_container, character_title, instruction_msg]
        )
        
        msg.submit(
            chat_interface,
            inputs=[msg, chatbot, current_char],
            outputs=[chatbot, msg]
        ).then(
            lambda: get_game_status(),
            outputs=[game_status]
        )
        
        send_btn.click(
            chat_interface,
            inputs=[msg, chatbot, current_char],
            outputs=[chatbot, msg]
        ).then(
            lambda: get_game_status(),
            outputs=[game_status]
        )
        
        close_chat_btn.click(
            close_chat,
            outputs=[chat_visible, chatbot, current_char, map_image]
        ).then(
            update_chat_visibility,
            inputs=[chat_visible, current_char],
            outputs=[chat_container, character_title, instruction_msg]
        )
        
        clear_chat_btn.click(lambda: [], outputs=[chatbot])
        
        refresh_btn.click(
            lambda: get_game_status(),
            outputs=[game_status]
        )
        
        # Reset avec mise à jour du debug si activé
        if os.getenv('ECHOFORGE_DEBUG', 'false').lower() == 'true':
            reset_btn.click(
                reset_game,
                outputs=[chatbot, game_status, map_image, debug_info]
            ).then(
                lambda: (False, "", []),
                outputs=[chat_visible, current_char, chatbot]
            ).then(
                update_chat_visibility,
                inputs=[chat_visible, current_char],
                outputs=[chat_container, character_title, instruction_msg]
            )
        else:
            reset_btn.click(
                lambda: reset_game()[:3],  # Ignore debug output
                outputs=[chatbot, game_status, map_image]
            ).then(
                lambda: (False, "", []),
                outputs=[chat_visible, current_char, chatbot]
            ).then(
                update_chat_visibility,
                inputs=[chat_visible, current_char],
                outputs=[chat_container, character_title, instruction_msg]
            )
        
        # Toggle debug button (seulement si agents disponibles)
        if AGENTS_AVAILABLE:
            if os.getenv('ECHOFORGE_DEBUG', 'false').lower() == 'true':
                toggle_debug_btn.click(
                    toggle_debug_mode,
                    outputs=[game_status, debug_info]
                )
            else:
                toggle_debug_btn.click(
                    lambda: toggle_debug_mode()[0],  # Juste le status
                    outputs=[game_status]
                )
        
        # Instructions
        system_description = """
        🤖 **Système d'Agents LangGraph** - Chaque personnage utilise un agent intelligent qui :
        - 🧠 Analyse automatiquement la complexité de votre message
        - 🔍 Décide s'il a besoin de chercher des informations (RAG)
        - ⚡ Optimise la vitesse de réponse selon le contexte
        - 💾 Maintient une mémoire persistante des conversations
        - 🎭 Adapte sa personnalité et ses émotions dynamiquement
        """ if AGENTS_AVAILABLE else """
        📚 **Système RAG de Base** - Utilise le système de recherche d'informations classique
        """
        
        gr.HTML(f"""
        <div style="text-align: center; padding: 20px; margin-top: 20px; background-color: #f0f0f0; border-radius: 10px;">
            <h4>🎯 Comment jouer avec l'IA avancée</h4>
            <p><strong>1.</strong> Cliquez sur les avatars des personnages (emojis) sur la carte</p>
            <p><strong>2.</strong> Dialoguez naturellement - l'IA s'adapte automatiquement</p>
            <p><strong>3.</strong> Posez des questions complexes pour déclencher le RAG</p>
            <p><strong>4.</strong> Négociez intelligemment pour obtenir les ressources</p>
            <p><strong>5.</strong> Réparez votre montgolfière pour quitter l'île !</p>
            <hr>
            {system_description}
            <p><em>💡 Activez le mode debug pour voir le fonctionnement interne de l'IA !</em></p>
        </div>
        """)
    
    return demo


def main():
    """Lance l'application avec le système d'agents intégré."""
    
    print("🎈 Démarrage d'EchoForge avec Agents LangGraph...")
    print("=" * 60)
    
    # Vérification des systèmes disponibles
    if AGENTS_AVAILABLE:
        print("✅ Système d'agents LangGraph disponible")
    elif RAG_AVAILABLE:
        print("⚠️ Fallback vers le système RAG de base")
    else:
        print("❌ Aucun système de dialogue disponible")
        print("Vérifiez l'installation des dépendances:")
        print("  pip install langgraph==0.5.1 langchain>=0.3.0")
        return
    
    # Initialisation du système de dialogue
    if not initialize_dialogue_system():
        print("❌ Impossible d'initialiser le système de dialogue.")
        print("Vérifiez que:")
        print("  - Ollama est installé et en cours d'exécution")
        print("  - Les modèles sont disponibles")
        print("  - Le dossier ./data contient les données")
        return
    
    print("✅ Système de dialogue initialisé avec succès !")
    print("🎮 Lancement de l'interface...")
    
    # Création et lancement de l'interface
    demo = create_interface()
    
    # Configuration du lancement
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=AGENTS_AVAILABLE,  # Debug activé seulement avec les agents
        show_error=True,
        favicon_path=None,
        auth=None
    )


if __name__ == "__main__":
    main()