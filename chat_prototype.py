import gradio as gr
from groq import Groq
import json
from datetime import datetime
import os
from typing import List, Tuple, Dict
from main import EchoForgeRAG
from dotenv import load_dotenv

load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")

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

def create_character_prompt(character_data: Dict, conversation_history: List, game_state: Dict, rag_system: EchoForgeRAG) -> str:
    """Crée le prompt système pour un personnage"""
    
    rag_system.create_character_prompt(self, character_data: Dict,
                              world_context: List[str],
                              character_context: List[str],
                              parsed_input: ActionParsed,
                              conversation_history: str = "")

    return prompt

def get_character_response(character_key: str, user_message: str) -> str:
    """Obtient une réponse du personnage via Groq"""
    
    try:
        client = initialize_groq_client()
        character_data = CHARACTERS[character_key]
        
        # Récupère l'historique pour ce personnage
        if character_key not in game_state["conversation_history"]:
            game_state["conversation_history"][character_key] = []
        
        conversation_history = game_state["conversation_history"][character_key]
        system_prompt = create_character_prompt(character_data, conversation_history, game_state)
        
        # Prépare les messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Ajoute l'historique récent (derniers 6 messages)
        for msg in conversation_history[-6:]:
            messages.append(msg)
        
        # Ajoute le message actuel
        messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        character_response = response.choices[0].message.content
        
        # Sauvegarde dans l'historique
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": character_response})
        
        return character_response
        
    except Exception as e:
        return f"❌ Erreur de communication: {str(e)}"

def chat_interface(message: str, history: List[Tuple[str, str]], character_dropdown: str) -> Tuple[List[Tuple[str, str]], str]:
    """Interface de chat principal"""
    
    if not message.strip():
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
    
    with gr.Blocks(theme=theme, title="🎈 EchoForge") as demo:
        
        # En-tête
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎈 EchoForge</h1>
            <h3>L'Île des Personnages Intelligents</h3>
            <p><em>Donnez une âme à vos personnages</em></p>
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
                    placeholder="Tapez votre message ici...",
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
        
        # Connexions des événements
        msg.submit(chat_interface, [msg, chatbot, character_dropdown], [chatbot, msg])
        send_btn.click(chat_interface, [msg, chatbot, character_dropdown], [chatbot, msg])
        clear_btn.click(lambda: [], None, chatbot)
        refresh_btn.click(get_game_status, None, game_status)
        reset_btn.click(reset_game, None, [chatbot, game_status])
        character_dropdown.change(update_character_info, character_dropdown, character_info)
        
        # Instructions en bas
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 20px; background-color: #f0f0f0; border-radius: 10px;">
            <h4>🎯 Instructions</h4>
            <p>Vous êtes échoué sur une île après un crash de montgolfière. Interagissez avec les habitants pour obtenir les ressources nécessaires à votre réparation !</p>
            <p><strong>Astuce:</strong> Chaque personnage a ses propres motivations et conditions d'échange.</p>
        </div>
        """)
    
    return demo

def main():
    """Lance l'application"""
    
    # Vérification de la clé API
    if GROQ_API_KEY == "your-groq-api-key-here":
        print("⚠️  ATTENTION: Configurez votre clé API Groq !")
        print("   export GROQ_API_KEY='votre_cle_ici'")
        print("   ou créez un fichier .env avec GROQ_API_KEY=votre_cle_ici")
        return
    # Initialisation
    rag_system = EchoForgeRAG()

    # Construction des vector stores (à faire une fois)
    rag_system.build_world_vectorstore()
    rag_system.build_character_vectorstore("fathira")
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