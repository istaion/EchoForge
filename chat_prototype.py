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

# Import du système RAG
try:
    from main import EchoForgeRAG, ActionParsed
except ImportError:
    print("⚠️ Erreur: Impossible d'importer EchoForgeRAG depuis main.py")
    EchoForgeRAG = None
    ActionParsed = None

# Données des personnages avec leurs positions sur la carte
CHARACTERS = {
    "fathira": {
        "name": "Fathira",
        "role": "Maire de l'île",
        "personality": "Diplomatique, curieuse et protectrice de sa communauté",
        "speech_style": "Formel mais chaleureux, utilise 'citoyen' et 'notre communauté'",
        "backstory": "Maire élue depuis 10 ans, garde les secrets de l'île et possède un trésor ancestral",
        "current_mood": "neutre",
        "can_give_gold": True,
        "special_knowledge": ["Histoire de l'île", "Localisation du trésor", "Relations entre habitants"],
        "emoji": "👑",
        "building": "Mairie (grande maison en haut)",
        "position": {"x": 598, "y": 190}
    },
    "claude": {
        "name": "Claude",
        "role": "Forgeron de l'île", 
        "personality": "Pragmatique, direct, passionné par son métier",
        "speech_style": "Franc, utilise du vocabulaire technique, parle de métal et d'outils",
        "backstory": "Forgeron depuis 20 ans, peut réparer n'importe quoi mais aime négocier",
        "current_mood": "neutre",
        "wants_cookies": True,
        "can_repair": True,
        "special_knowledge": ["Métallurgie", "Réparation d'objets complexes", "Histoire des outils de l'île"],
        "emoji": "🔨",
        "building": "Forge (maison à gauche)",
        "position": {"x": 300, "y": 400}
    },
    "azzedine": {
        "name": "Azzedine", 
        "role": "Styliste de l'île",
        "personality": "Créatif, perfectionniste, parfois capricieux",
        "speech_style": "Artistique, utilise des métaphores, parle de beauté et d'esthétique",
        "backstory": "Styliste talentueux, vend des tissus rares mais exigeant sur la qualité",
        "current_mood": "neutre",
        "sells_fabric": True,
        "special_knowledge": ["Tissus et matériaux", "Tendances artistiques", "Secrets de confection"],
        "emoji": "✂️",
        "building": "Atelier de couture (maison colorée à droite)",
        "position": {"x": 820, "y": 580}
    },
    "roberte": {
        "name": "Roberte",
        "role": "Cuisinière de l'île",
        "personality": "Généreuse mais territoriale, perfectionniste en cuisine",
        "speech_style": "Maternel, parle de recettes et d'ingrédients, utilise des expressions culinaires",
        "backstory": "Cuisinière réputée, déteste être dérangée pendant son travail mais offre volontiers des cookies en pause",
        "current_mood": "neutre", 
        "gives_cookies": True,
        "cooking_schedule": "Cuisine le matin (8h-12h), pause l'après-midi (14h-16h)",
        "special_knowledge": ["Recettes ancestrales", "Ingrédients de l'île", "Habitudes alimentaires des habitants"],
        "emoji": "👩‍🍳",
        "building": "Auberge (maison avec terrasse au centre)",
        "position": {"x": 590, "y": 480}
    }
}

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
    "chat_open": False
}

# Instance globale du système RAG
rag_system = None

def initialize_rag_system():
    """Initialise le système RAG"""
    global rag_system
    
    if not EchoForgeRAG:
        print("❌ EchoForgeRAG non disponible")
        return False
    
    try:
        print("🚀 Initialisation du système RAG...")
        rag_system = EchoForgeRAG(
            data_path="./data",
            vector_store_path="./vector_stores",
            model_name="llama3.1:8b"
        )
        
        print("📚 Vérification des vector stores...")
        
        # Vector store du monde
        world_store_path = Path("./vector_stores/world_lore")
        if not world_store_path.exists():
            print("🌍 Construction du vector store du monde...")
            rag_system.build_world_vectorstore()
        
        # Vector stores des personnages
        for character_id in CHARACTERS.keys():
            char_store_path = Path(f"./vector_stores/character_{character_id}")
            if not char_store_path.exists():
                print(f"👤 Construction du vector store pour {character_id}...")
                rag_system.build_character_vectorstore(character_id)
        
        print("✅ Système RAG initialisé avec succès!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation du RAG: {str(e)}")
        return False

def create_character_avatar(emoji: str, size: int = 60, active: bool = False) -> Image.Image:
    """Crée un avatar circulaire pour un personnage"""
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
        # Calculer la position pour centrer l'emoji
        text_bbox = draw.textbbox((0, 0), emoji)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (size - text_width) // 2
        y = (size - text_height) // 2
        
        draw.text((x, y), emoji, fill=(0, 0, 0, 255))
    except:
        # Fallback
        draw.text((size//4, size//4), "?", fill=(0, 0, 0, 255))
    
    return img

def load_map_image(map_path: str = "data/img/board.png") -> Image.Image:
    """Charge l'image de la carte ou crée une carte placeholder"""
    
    try:
        print(f"chemin de l'image : {Path(map_path)}")
        return Image.open(map_path)
        
    except Exception as e:
        return f"❌ Erreur lors de la récupération de l'image du board: {str(e)}"
        

def generate_interactive_map(active_character: str = None) -> Image.Image:
    """Génère la carte avec les personnages positionnés"""
    
    # Charger l'image de base
    map_img = load_map_image("data/img/board.png")  
    
    # Ajouter les avatars des personnages
    for char_id, char_data in CHARACTERS.items():
        pos = char_data["position"]
        is_active = (char_id == active_character)
        avatar = create_character_avatar(char_data["emoji"], 50, is_active)
        
        # Superposer l'avatar sur la carte
        map_img.paste(avatar, (pos["x"]-25, pos["y"]-25), avatar)
    
    # Ajouter l'avatar du joueur
    player_avatar = create_character_avatar("🧑", 40)
    player_pos = game_state["player_position"]
    map_img.paste(player_avatar, (player_pos["x"]-20, player_pos["y"]-20), player_avatar)
    
    return map_img

def get_conversation_history_string(character_key: str, max_messages: int = 6) -> str:
    """Récupère l'historique de conversation formaté pour un personnage"""
    if character_key not in game_state["conversation_history"]:
        return ""
    
    history = game_state["conversation_history"][character_key]
    recent_history = history[-max_messages:] if len(history) > max_messages else history
    
    formatted_history = []
    for msg in recent_history:
        role = "Joueur" if msg["role"] == "user" else CHARACTERS[character_key]["name"]
        formatted_history.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted_history)

def get_character_response(character_key: str, user_message: str) -> str:
    """Obtient une réponse du personnage via le système RAG"""
    
    if not rag_system:
        return "❌ Système RAG non initialisé. Redémarrez l'application."
    
    try:
        character_data = CHARACTERS[character_key]
        
        # Parse le message utilisateur pour extraire les actions
        parsed_input = rag_system.parse_actions(user_message)
        
        # Récupère les contextes pertinents
        world_context = rag_system.retrieve_world_context(parsed_input.text, top_k=3)
        character_context = rag_system.retrieve_character_context(
            parsed_input.text, character_key, top_k=5
        )
        
        # Récupère l'historique de conversation
        conversation_history = get_conversation_history_string(character_key)
        
        # Crée le prompt complet
        prompt = rag_system.create_character_prompt(
            character_data=character_data,
            world_context=world_context,
            character_context=character_context,
            parsed_input=parsed_input,
            conversation_history=conversation_history
        )
        
        # Génère la réponse via le LLM local
        response = rag_system.llm.invoke(prompt)
        
        # Sauvegarde dans l'historique
        if character_key not in game_state["conversation_history"]:
            game_state["conversation_history"][character_key] = []
        
        game_state["conversation_history"][character_key].extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ])
        
        # Parse la réponse pour détecter des actions spéciales
        process_character_actions(character_key, response, parsed_input)
        
        return response
        
    except Exception as e:
        return f"❌ Erreur lors de la génération de réponse: {str(e)}"

def process_character_actions(character_key: str, character_response: str, user_input):
    """Traite les actions spéciales du personnage et met à jour l'état du jeu"""
    
    character_data = CHARACTERS[character_key]
    response_lower = character_response.lower()
    
    # Détection des actions basées sur le contenu de la réponse
    if character_key == "fathira" and character_data.get("can_give_gold"):
        if any(word in response_lower for word in ["donne", "offre", "voici", "prends"]) and "or" in response_lower:
            if game_state["player_gold"] < 100:
                game_state["player_gold"] += 10
                print(f"💰 Fathira vous a donné 10 pièces d'or!")
    
    elif character_key == "roberte" and character_data.get("gives_cookies"):
        if any(word in response_lower for word in ["donne", "offre", "voici", "prends"]) and "cookie" in response_lower:
            if game_state["player_cookies"] < 20:
                game_state["player_cookies"] += 3
                print(f"🍪 Roberte vous a donné 3 cookies!")
    
    elif character_key == "azzedine" and character_data.get("sells_fabric"):
        if "tissu" in response_lower and game_state["player_gold"] >= 15:
            if any(word in response_lower for word in ["vends", "achète", "prends", "voici"]):
                game_state["player_gold"] -= 15
                game_state["player_fabric"] += 1
                print(f"🧶 Azzedine vous a vendu du tissu pour 15 or!")
    
    elif character_key == "claude" and character_data.get("can_repair"):
        if "répare" in response_lower and game_state["player_cookies"] >= 5 and game_state["player_fabric"] >= 1:
            if any(word in response_lower for word in ["répare", "réparer", "fini", "terminé"]):
                game_state["player_cookies"] -= 5
                game_state["player_fabric"] -= 1
                game_state["montgolfiere_repaired"] = True
                print(f"🎈 Claude a réparé votre montgolfière! Vous pouvez repartir!")

def handle_map_click(evt: gr.SelectData) -> Tuple[str, bool, str, Image.Image]:
    """Gère les clics sur la carte"""
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
    """Interface de chat avec un personnage"""
    
    if not message.strip() or not character_id:
        return history, ""
    
    if not rag_system:
        error_msg = "❌ Système RAG non initialisé. Redémarrez l'application."
        history.append((message, error_msg))
        return history, ""
    
    # Obtient la réponse du personnage
    character_response = get_character_response(character_id, message)
    
    # Met à jour l'historique d'affichage
    history.append((message, character_response))
    
    return history, ""

def close_chat() -> Tuple[bool, List, str, Image.Image]:
    """Ferme la fenêtre de chat"""
    game_state["chat_open"] = False
    game_state["current_character"] = None
    return False, [], "", generate_interactive_map()

def get_game_status() -> str:
    """Retourne l'état actuel du jeu"""
    repair_status = "✅ Réparée" if game_state["montgolfiere_repaired"] else "❌ Endommagée"
    
    status = f"""## 🎮 État du Jeu
    
**Ressources:**
- 💰 Or: {game_state['player_gold']}
- 🍪 Cookies: {game_state['player_cookies']}
- 🧶 Tissu: {game_state['player_fabric']}

**Montgolfière:** {repair_status}

**Objectif:** Réparer votre montgolfière pour quitter l'île !

**Système RAG:** {"✅ Actif" if rag_system else "❌ Non initialisé"}
"""
    return status

def reset_game() -> Tuple[List, str, Image.Image]:
    """Remet à zéro le jeu"""
    global game_state
    game_state = {
        "player_gold": 0,
        "player_cookies": 0,
        "player_fabric": 0,
        "montgolfiere_repaired": False,
        "conversation_history": {},
        "current_character": None,
        "player_position": BALLOON_POSITION.copy(),
        "chat_open": False
    }
    return [], get_game_status(), generate_interactive_map()

def create_interface():
    """Crée l'interface Gradio avec carte interactive"""
    
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald",
        neutral_hue="slate",
    )
    
    with gr.Blocks(theme=theme, title="🎈 EchoForge RAG Interactive") as demo:
        
        # Variables d'état pour l'interface
        chat_visible = gr.State(False)
        current_char = gr.State("")
        
        # En-tête
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎈 EchoForge RAG - Île Interactive</h1>
            <h3>Cliquez sur les personnages pour leur parler</h3>
            <p><em>Donnez une âme à vos personnages avec l'IA</em></p>
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
                        label="Conversation",
                        height=300,
                        show_label=True,
                        container=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Votre message",
                            placeholder="Tapez votre message ici... (utilisez *action* pour les actions physiques)",
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
                gr.Markdown("""
                ## 👥 Personnages de l'île
                
                **👑 Fathira** - Maire  
                *Donne de l'or, connaît les secrets*
                
                **🔨 Claude** - Forgeron  
                *Répare la montgolfière contre des cookies*
                
                **✂️ Azzedine** - Styliste  
                *Vend du tissu contre de l'or*
                
                **👩‍🍳 Roberte** - Cuisinière  
                *Donne des cookies pendant ses pauses*
                
                💡 **Astuce:** Chaque personnage a sa personnalité et ses motivations propres !
                """)
                
                gr.Markdown("---")
                
                # Boutons d'action
                with gr.Column():
                    refresh_btn = gr.Button("🔄 Actualiser État", variant="secondary")
                    reset_btn = gr.Button("🆕 Nouveau Jeu", variant="stop")
                    init_rag_btn = gr.Button("🤖 Réinitialiser RAG", variant="secondary")
        
        # Fonctions de gestion des événements
        def update_chat_visibility(visible: bool, char_id: str):
            """Met à jour la visibilité du chat"""
            if visible and char_id:
                char_data = CHARACTERS[char_id]
                title = f"## 💬 Conversation avec {char_data['emoji']} {char_data['name']}"
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
        
        def reinitialize_rag():
            """Réinitialise le système RAG"""
            success = initialize_rag_system()
            return get_game_status()
        
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
        
        reset_btn.click(
            reset_game,
            outputs=[chatbot, game_status, map_image]
        ).then(
            lambda: (False, "", []),
            outputs=[chat_visible, current_char, chatbot]
        ).then(
            update_chat_visibility,
            inputs=[chat_visible, current_char],
            outputs=[chat_container, character_title, instruction_msg]
        )
        
        init_rag_btn.click(
            reinitialize_rag,
            outputs=[game_status]
        )
        
        # Instructions
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 20px; background-color: #f0f0f0; border-radius: 10px;">
            <h4>🎯 Comment jouer</h4>
            <p><strong>1.</strong> Cliquez sur les avatars des personnages (emojis) sur la carte</p>
            <p><strong>2.</strong> Dialoguez intelligemment avec eux grâce au système RAG</p>
            <p><strong>3.</strong> Négociez pour obtenir les ressources nécessaires</p>
            <p><strong>4.</strong> Réparez votre montgolfière pour quitter l'île !</p>
            <hr>
            <p><em>💡 Le système RAG utilise la mémoire et la personnalité de chaque personnage pour des conversations plus riches !</em></p>
        </div>
        """)
    
    return demo

def main():
    """Lance l'application"""
    
    print("🎈 Démarrage d'EchoForge RAG Interactive...")
    
    # Initialisation du système RAG
    if not initialize_rag_system():
        print("❌ Impossible d'initialiser le système RAG.")
        print("Vérifiez que:")
        print("  - Ollama est installé et en cours d'exécution")
        print("  - Les modèles llama3.1:8b et paraphrase-multilingual:278m-mpnet-base-v2-fp16 sont disponibles")
        print("  - Le dossier ./data contient les données des personnages")
        print("  - Le fichier main.py avec EchoForgeRAG est présent")
        return
    
    # Création et lancement de l'interface
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()