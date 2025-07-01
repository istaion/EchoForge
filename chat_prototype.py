import gradio as gr
import json
from datetime import datetime
import os
from typing import List, Tuple, Dict
from main import EchoForgeRAG, ActionParsed
from pathlib import Path

# Données des personnages
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
        "emoji": "👑"
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
        "emoji": "🔨"
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
        "emoji": "✂️"
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
        "emoji": "👩‍🍳"
    }
}

# Contexte général du jeu
GAME_CONTEXT = """
CONTEXTE: Tu es sur une île mystérieuse après que ta montgolfière ait été prise dans une tempête. 
Ta montgolfière est endommagée et tu as besoin de la réparer pour repartir.

RESSOURCES NÉCESSAIRES:
- Tissu pour réparer l'enveloppe de la montgolfière
- Réparation mécanique pour le panier et les systèmes

HABITANTS DE L'ÎLE:
- Fathira (Maire): Peut donner de l'or, connaît tous les secrets de l'île
- Claude (Forgeron): Peut réparer ta montgolfière mais veut des cookies en échange
- Azzedine (Styliste): Vend du tissu de qualité pour de l'or
- Roberte (Cuisinière): Donne des cookies pendant ses pauses (14h-16h), n'aime pas être dérangée pendant qu'elle cuisine

OBJECTIF: Obtenir les ressources nécessaires en interagissant intelligemment avec les habitants.
"""

# État global du jeu
game_state = {
    "player_gold": 0,
    "player_cookies": 0,
    "player_fabric": 0,
    "montgolfiere_repaired": False,
    "conversation_history": {},
    "current_character": "fathira"
}

# Instance globale du système RAG
rag_system = None

def initialize_rag_system():
    """Initialise le système RAG"""
    global rag_system
    
    try:
        print("🚀 Initialisation du système RAG...")
        rag_system = EchoForgeRAG(
            data_path="./data",
            vector_store_path="./vector_stores",
            model_name="llama3.1:8b"
        )
        
        # Vérifie si les vector stores existent, sinon les crée
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

def process_character_actions(character_key: str, character_response: str, user_input: ActionParsed):
    """Traite les actions spéciales du personnage et met à jour l'état du jeu"""
    
    character_data = CHARACTERS[character_key]
    response_lower = character_response.lower()
    
    # Détection des actions basées sur le contenu de la réponse
    if character_key == "fathira" and character_data.get("can_give_gold"):
        if any(word in response_lower for word in ["donne", "offre", "voici", "prends"]) and "or" in response_lower:
            if game_state["player_gold"] < 100:  # Limite pour éviter l'abus
                game_state["player_gold"] += 10
                print(f"💰 Fathira vous a donné 10 pièces d'or!")
    
    elif character_key == "roberte" and character_data.get("gives_cookies"):
        if any(word in response_lower for word in ["donne", "offre", "voici", "prends"]) and "cookie" in response_lower:
            if game_state["player_cookies"] < 20:  # Limite
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

def chat_interface(message: str, history: List[Tuple[str, str]], character_dropdown: str) -> Tuple[List[Tuple[str, str]], str]:
    """Interface de chat principal"""
    
    if not message.strip():
        return history, ""
    
    if not rag_system:
        error_msg = "❌ Système RAG non initialisé. Redémarrez l'application."
        history.append((message, error_msg))
        return history, ""
    
    # Met à jour le personnage actuel
    game_state["current_character"] = character_dropdown
    
    # Obtient la réponse du personnage
    character_response = get_character_response(character_dropdown, message)
    
    # Met à jour l'historique d'affichage
    history.append((message, character_response))
    
    return history, ""

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

def reset_game() -> Tuple[List, str]:
    """Remet à zéro le jeu"""
    global game_state
    game_state = {
        "player_gold": 0,
        "player_cookies": 0,
        "player_fabric": 0,
        "montgolfiere_repaired": False,
        "conversation_history": {},
        "current_character": "fathira"
    }
    return [], get_game_status()

def create_interface():
    """Crée l'interface Gradio"""
    
    # Thème personnalisé
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald",
        neutral_hue="slate",
    )
    
    with gr.Blocks(theme=theme, title="🎈 EchoForge RAG") as demo:
        
        # En-tête
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎈 EchoForge RAG</h1>
            <h3>L'Île des Personnages Intelligents</h3>
            <p><em>Donnez une âme à vos personnages avec RAG</em></p>
        </div>
        """)
        
        with gr.Row():
            # Colonne principale - Chat
            with gr.Column(scale=2):
                
                # Sélection du personnage
                character_choices = [(f"{data['emoji']} {data['name']} - {data['role']}", key) 
                                   for key, data in CHARACTERS.items()]
                
                character_dropdown = gr.Dropdown(
                    choices=character_choices,
                    value="fathira",
                    label="💬 Parler avec:",
                    interactive=True
                )
                
                # Interface de chat
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )
                
                msg = gr.Textbox(
                    label="Votre message",
                    placeholder="Tapez votre message ici... (utilisez *action* pour les actions physiques)",
                    lines=2,
                    max_lines=4
                )
                
                with gr.Row():
                    send_btn = gr.Button("📤 Envoyer", variant="primary")
                    clear_btn = gr.Button("🗑️ Effacer Chat", variant="secondary")
            
            # Colonne latérale - État du jeu et infos
            with gr.Column(scale=1):
                
                # État du jeu
                game_status = gr.Markdown(get_game_status())
                
                gr.Markdown("---")
                
                # Informations sur le personnage sélectionné
                character_info = gr.Markdown(
                    f"""## 👑 Fathira
                    **Rôle:** Maire de l'île
                    
                    **Personnalité:** {CHARACTERS['fathira']['personality']}
                    
                    **Peut aider avec:** Donner de l'or, informations sur l'île
                    """,
                    label="Personnage actuel"
                )
                
                gr.Markdown("---")
                
                # Boutons d'action
                with gr.Column():
                    refresh_btn = gr.Button("🔄 Actualiser État", variant="secondary")
                    reset_btn = gr.Button("🆕 Nouveau Jeu", variant="stop")
                    init_btn = gr.Button("🚀 Réinitialiser RAG", variant="secondary")
        
        # Logique des événements
        def update_character_info(character_key):
            """Met à jour les infos du personnage sélectionné"""
            char_data = CHARACTERS[character_key]
            
            # Détermine ce que le personnage peut faire
            abilities = []
            if char_data.get("can_give_gold"):
                abilities.append("Donner de l'or")
            if char_data.get("can_repair"):
                abilities.append("Réparer la montgolfière")
            if char_data.get("sells_fabric"):
                abilities.append("Vendre du tissu")
            if char_data.get("gives_cookies"):
                abilities.append("Donner des cookies")
            
            abilities_text = ", ".join(abilities) if abilities else "Conversation générale"
            
            info_text = f"""## {char_data['emoji']} {char_data['name']}
**Rôle:** {char_data['role']}

**Personnalité:** {char_data['personality']}

**Peut aider avec:** {abilities_text}

**Style:** {char_data['speech_style']}
"""
            return info_text
        
        def reinitialize_rag():
            """Réinitialise le système RAG"""
            success = initialize_rag_system()
            if success:
                return get_game_status()
            else:
                return "❌ Échec de la réinitialisation du RAG"
        
        # Connexions des événements
        msg.submit(chat_interface, [msg, chatbot, character_dropdown], [chatbot, msg])
        send_btn.click(chat_interface, [msg, chatbot, character_dropdown], [chatbot, msg])
        clear_btn.click(lambda: [], None, chatbot)
        refresh_btn.click(get_game_status, None, game_status)
        reset_btn.click(reset_game, None, [chatbot, game_status])
        character_dropdown.change(update_character_info, character_dropdown, character_info)
        init_btn.click(reinitialize_rag, None, game_status)
        
        # Instructions en bas
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 20px; background-color: #f0f0f0; border-radius: 10px;">
            <h4>🎯 Instructions</h4>
            <p>Vous êtes échoué sur une île après un crash de montgolfière. Interagissez avec les habitants pour obtenir les ressources nécessaires à votre réparation !</p>
            <p><strong>Astuce:</strong> Chaque personnage a ses propres motivations et conditions d'échange.</p>
            <p><strong>Actions:</strong> Utilisez *votre action* pour décrire des actions physiques.</p>
            <p><strong>RAG:</strong> Le système utilise la mémoire et le contexte des personnages pour des réponses plus intelligentes.</p>
        </div>
        """)
    
    return demo

def main():
    """Lance l'application"""
    
    print("🎈 Démarrage d'EchoForge RAG...")
    
    # Initialisation du système RAG
    if not initialize_rag_system():
        print("❌ Impossible d'initialiser le système RAG.")
        print("Vérifiez que:")
        print("  - Ollama est installé et en cours d'exécution")
        print("  - Les modèles llama3.1:8b et paraphrase-multilingual:278m-mpnet-base-v2-fp16 sont disponibles")
        print("  - Le dossier ./data contient les données des personnages")
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