"""Nœuds de génération de réponses."""

import random
from typing import Dict, Any, List
from ..state.character_state import CharacterState


def generate_simple_response(state: CharacterState) -> CharacterState:
    """
    Génère une réponse simple basée sur la personnalité, sans RAG.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État mis à jour avec la réponse générée
    """
    state["processing_steps"].append("simple_response_generation")
    
    character_name = state["character_name"]
    personality = state["personality_traits"]
    intent = state["message_intent"]
    emotion = state["current_emotion"]
    
    # Génération de réponse simple selon l'intention
    response = _generate_personality_response(
        character_name, personality, intent, emotion, state["user_message"]
    )
    
    state["response"] = response
    
    # Debug info
    state["debug_info"]["response_generation"] = {
        "type": "simple",
        "intent_used": intent,
        "emotion_used": emotion,
        "personality_traits_count": len(personality)
    }
    
    return state


def generate_response(state: CharacterState) -> CharacterState:
    """
    Génère une réponse complexe avec ou sans contexte RAG.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État mis à jour avec la réponse générée
    """
    state["processing_steps"].append("complex_response_generation")
    
    character_name = state["character_name"]
    personality = state["personality_traits"]
    user_message = state["user_message"]
    rag_results = state["rag_results"]
    
    # Construction du contexte pour la génération
    context = _build_response_context(state)
    
    # Génération de la réponse
    if rag_results:
        response = _generate_rag_enhanced_response(
            character_name, personality, user_message, context, rag_results
        )
        response_type = "rag_enhanced"
    else:
        response = _generate_contextual_response(
            character_name, personality, user_message, context
        )
        response_type = "contextual"
    
    state["response"] = response
    
    # Debug info
    state["debug_info"]["response_generation"] = {
        "type": response_type,
        "rag_results_used": len(rag_results),
        "context_elements": len(context),
        "response_length": len(response)
    }
    
    return state


def _generate_personality_response(
    character_name: str, 
    personality: Dict[str, Any], 
    intent: str, 
    emotion: str, 
    user_message: str
) -> str:
    """Génère une réponse simple basée uniquement sur la personnalité."""
    
    # Templates de réponses par personnage et intention
    response_templates = {
        "fathira": {
            "greeting": [
                "Bonjour, citoyen ! Comment puis-je vous aider aujourd'hui ?",
                "Salutations ! En tant que maire, je suis à votre service.",
                "Bienvenue ! Que puis-je faire pour notre communauté ?"
            ],
            "farewell": [
                "Au revoir ! Revenez quand vous aurez faim !",
                "À bientôt ! Mes cookies vous attendront !",
                "Prenez soin de vous et mangez bien !"
            ],
            "general": [
                "En tant que maire, je veille sur tous les habitants de notre île.",
                "Notre communauté est ma priorité absolue, citoyen.",
                "Parlons de ce qui vous préoccupe."
            ]
        },
        "claude": {
            "greeting": [
                "Salut ! Mon marteau et moi sommes à votre service.",
                "Bonjour ! Besoin de quelque chose de réparé ?",
                "Hey ! L'atelier est ouvert, que puis-je forger pour vous ?"
            ],
            "farewell": [
                "À plus ! Si ça casse, vous savez où me trouver !",
                "Prenez soin de vos outils !",
                "Au revoir ! Que le métal soit avec vous !"
            ],
            "general": [
                "Je peux réparer à peu près n'importe quoi, mais ça se négocie.",
                "Mon atelier a vu passer bien des objets étranges...",
                "Le métal ne ment jamais, contrairement aux gens."
            ]
        },
        "azzedine": {
            "greeting": [
                "Bonjour ! Admirez-vous ma dernière création ?",
                "Salut ! L'art textile vous intéresse-t-il ?",
                "Bienvenue dans mon atelier de beauté !"
            ],
            "farewell": [
                "Au revoir ! Que la beauté vous accompagne !",
                "Partez avec style !",
                "À bientôt ! Revenez quand vous aurez développé votre goût esthétique."
            ],
            "general": [
                "La beauté est dans les détails, et je suis un perfectionniste.",
                "Mes tissus sont d'une qualité incomparable sur cette île.",
                "L'art véritable demande de la patience et de l'exigence."
            ]
        },
        "roberte": {
            "greeting": [
                "Bonjour ! J'espère que vous avez faim !",
                "Salut ! La cuisine sent bon aujourd'hui, n'est-ce pas ?",
                "Bienvenue ! Venez-vous pour mes fameux cookies ?"
            ],
            "farewell": [
                "Au revoir, citoyen. Que votre journée soit prospère !",
                "Prenez soin de vous et de notre belle île !",
                "À bientôt ! N'hésitez pas à revenir me voir."
            ],
            "general": [
                "Ma cuisine est ma fierté, ne me dérangez pas pendant que je travaille !",
                "Un bon repas réchauffe le cœur et l'âme.",
                "Mes cookies sont les meilleurs de l'île, mais il faut les mériter !"
            ]
        }
    }
    
    # Sélection du template approprié
    character_templates = response_templates.get(character_name.lower(), response_templates["fathira"])
    intent_templates = character_templates.get(intent, character_templates["general"])
    
    # Sélection aléatoire d'une réponse
    response = random.choice(intent_templates)
    
    # Modification selon l'émotion si nécessaire
    if emotion == "angry":
        response = response.replace("!", ".")
        response += " (Je suis un peu énervé en ce moment...)"
    elif emotion == "happy":
        response += " 😊"
    elif emotion == "sad":
        response = response.lower()
        response += " *soupir*"
    
    return response


def _build_response_context(state: CharacterState) -> Dict[str, Any]:
    """Construit le contexte pour la génération de réponse."""
    
    context = {
        "character_name": state["character_name"],
        "current_emotion": state["current_emotion"],
        "message_intent": state["message_intent"],
        "complexity_level": state["complexity_level"],
        "conversation_turns": len(state["conversation_history"]),
        "has_rag_data": bool(state["rag_results"])
    }
    
    # Ajout du résumé de conversation si disponible
    if state.get("context_summary"):
        context["conversation_summary"] = state["context_summary"]
    
    # Ajout des actions planifiées
    if state["planned_actions"]:
        context["planned_actions"] = state["planned_actions"]
    
    return context


def _generate_rag_enhanced_response(
    character_name: str,
    personality: Dict[str, Any], 
    user_message: str,
    context: Dict[str, Any],
    rag_results: List[Dict[str, Any]]
) -> str:
    """Génère une réponse enrichie par les données RAG."""
    
    # Extraction des informations les plus pertinentes
    relevant_info = []
    for result in rag_results:
        if result["relevance"] > 0.6:
            relevant_info.append(result["content"])
    
    # Templates de réponse par personnage avec intégration RAG
    rag_templates = {
        "fathira": [
            "En tant que maire, je peux vous dire que {knowledge}. Notre communauté a une riche histoire.",
            "D'après mes connaissances de l'île, {knowledge}. C'est important pour notre communauté.",
            "Laissez-moi vous expliquer, citoyen : {knowledge}. Voilà ce que je sais sur le sujet."
        ],
        "claude": [
            "D'après mon expérience de forgeron, {knowledge}. J'ai vu beaucoup de choses dans mon atelier.",
            "Je peux vous dire que {knowledge}. Le métal garde la mémoire des événements.",
            "Écoutez, {knowledge}. C'est ce que j'ai appris au fil des années."
        ],
        "azzedine": [
            "Avec mon œil artistique, je peux vous révéler que {knowledge}. La beauté révèle la vérité.",
            "D'un point de vue esthétique, {knowledge}. Chaque détail a son importance.",
            "Permettez-moi de vous éclairer : {knowledge}. L'art révèle bien des secrets."
        ],
        "roberte": [
            "En cuisinant pour tout le monde, j'ai appris que {knowledge}. On entend beaucoup de choses en cuisine !",
            "Entre deux plats, je peux vous dire que {knowledge}. Les habitants parlent beaucoup pendant les repas.",
            "D'après ce que j'ai entendu, {knowledge}. La cuisine est le cœur de l'information sur l'île !"
        ]
    }
    
    # Sélection du template
    character_templates = rag_templates.get(character_name.lower(), rag_templates["fathira"])
    template = random.choice(character_templates)
    
    # Intégration des connaissances
    if relevant_info:
        knowledge = relevant_info[0]  # Utilise la plus pertinente
        response = template.format(knowledge=knowledge)
    else:
        # Fallback si pas d'info pertinente
        response = _generate_contextual_response(character_name, personality, user_message, context)
    
    return response


def _generate_contextual_response(
    character_name: str,
    personality: Dict[str, Any],
    user_message: str, 
    context: Dict[str, Any]
) -> str:
    """Génère une réponse contextuelle sans RAG."""
    
    # Templates contextuels par personnage
    contextual_templates = {
        "fathira": [
            "C'est une question intéressante, citoyen. En tant que maire, je dois réfléchir à l'impact sur notre communauté.",
            "Hmm, votre demande mérite considération. Laissez-moi consulter mes connaissances de l'île.",
            "Voilà une préoccupation légitime. Notre île recèle encore bien des mystères."
        ],
        "claude": [
            "Intéressant... Laissez-moi réfléchir à ça. Mes outils me disent que c'est plus complexe qu'il n'y paraît.",
            "Votre question me rappelle quelque chose. J'ai déjà travaillé sur des cas similaires.",
            "Ah, ça c'est du travail de précision ! Donnez-moi un moment pour bien vous expliquer."
        ],
        "azzedine": [
            "Quelle question fascinante ! L'esthétique de votre demande me plaît.",
            "Votre curiosité témoigne d'un certain raffinement. Laissez-moi vous répondre avec style.",
            "Ah ! Enfin quelqu'un qui s'intéresse aux détails ! C'est tout un art de bien répondre."
        ],
        "roberte": [
            "Ça me rappelle une recette... Attendez, je vais vous expliquer ça comme il faut !",
            "Votre question me met l'eau à la bouche ! Enfin, façon de parler...",
            "Tiens, c'est comme quand je prépare mes cookies : il faut du temps et de la patience !"
        ]
    }
    
    # Sélection et personnalisation
    character_templates = contextual_templates.get(character_name.lower(), contextual_templates["fathira"])
    response = random.choice(character_templates)
    
    # Ajout d'éléments contextuels
    if context["complexity_level"] == "complex":
        response += " Cette question demande réflexion..."
    
    if context["conversation_turns"] > 3:
        response += " Nous avons déjà bien discuté ensemble !"
    
    return response