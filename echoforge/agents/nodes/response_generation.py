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
    Génère une réponse complexe avec ou sans contexte RAG via LLM.
    
    Args:
        state: État actuel du personnage
        
    Returns:
        État mis à jour avec la réponse générée
    """
    state["processing_steps"].append("complex_response_generation")
    
    # Génération via LLM avec contexte
    response = _generate_llm_response(state)
    
    state["response"] = response["content"]
    
    # Debug info
    state["debug_info"]["response_generation"] = {
        "type": "llm_generated",
        "rag_results_used": len(state["rag_results"]),
        "context_elements": len(state["relevant_knowledge"]),
        "response_length": len(response["content"]),
        "method": response.get("method", "llm_call"),
        "success": response.get("success", True)
    }
    
    return state


def _generate_llm_response(state: CharacterState) -> Dict[str, Any]:
    """
    Génère une réponse via LLM avec tout le contexte disponible.
    
    Args:
        state: État du personnage avec contexte
        
    Returns:
        Dict avec la réponse et métadonnées
    """
    try:
        # Import du système LLM
        from echoforge.core import EchoForgeRAG
        from echoforge.utils.config import get_config
        
        # Récupération de la configuration et du LLM
        config = get_config()
        rag_system = EchoForgeRAG(
            data_path=str(config.data_path),
            vector_store_path=str(config.vector_store_path),
            embedding_model=config.embedding_model,
            llm_model=config.llm_model
        )
        
        # Construction du prompt complet
        response_prompt = _build_comprehensive_prompt(state)
        
        # Appel au LLM
        llm_response = rag_system.llm.invoke(response_prompt)
        
        return {
            "content": llm_response.strip(),
            "method": "llm_call",
            "success": True,
            "prompt_length": len(response_prompt)
        }
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la génération LLM: {e}")
        # Fallback sur génération simple
        fallback_response = _generate_fallback_response(state)
        return {
            "content": fallback_response,
            "method": "fallback",
            "success": False,
            "error": str(e)
        }


def _build_comprehensive_prompt(state: CharacterState) -> str:
    """Construit un prompt complet avec tout le contexte disponible."""
    
    # Récupération des données
    character_name = state["character_name"]
    personality = state.get("personality_traits", {})
    emotion = state["current_emotion"]
    user_message = state["user_message"]
    rag_results = state["rag_results"]
    conversation_history = state.get("conversation_history", [])
    
    # Construction des sections du prompt
    
    # 1. Identité du personnage
    identity_section = f"""Tu es {character_name}.
PERSONNALITÉ: {_format_personality(personality)}
ÉMOTION ACTUELLE: {emotion}"""

    # 2. Contexte RAG si disponible
    context_section = ""
    if rag_results:
        context_items = []
        for result in rag_results[:3]:  # Limite à 3 résultats les plus pertinents
            context_items.append(f"- {result['content']} (pertinence: {result['relevance']:.2f})")
        
        context_section = f"""
CONNAISSANCES PERTINENTES:
{chr(10).join(context_items)}"""

    # 3. Historique de conversation récent
    history_section = ""
    if conversation_history:
        recent_history = conversation_history[-3:]  # 3 derniers échanges
        history_items = []
        for exchange in recent_history:
            if isinstance(exchange, dict):
                if "user" in exchange and "assistant" in exchange:
                    history_items.append(f"Utilisateur: {exchange['user']}")
                    history_items.append(f"Toi: {exchange['assistant']}")
        
        if history_items:
            history_section = f"""
HISTORIQUE RÉCENT:
{chr(10).join(history_items)}"""

    # 4. Instructions spécifiques
    instructions_section = f"""
MESSAGE ACTUEL DE L'UTILISATEUR: "{user_message}"

INSTRUCTIONS:
1. Reste parfaitement en personnage comme {character_name}
2. Utilise ta personnalité et ton émotion actuelle ({emotion})
3. Si tu as des connaissances pertinentes ci-dessus, intègre-les naturellement
4. Tiens compte de l'historique de conversation pour la cohérence
5. Garde ta réponse courte et naturelle (2-3 phrases maximum)
6. Si tu fais des actions physiques, mets-les entre *astérisques*
7. Réponds en français de manière authentique à ton personnage

RÉPONSE:"""

    # Assemblage final
    full_prompt = f"""{identity_section}{context_section}{history_section}

{instructions_section}"""
    
    return full_prompt


def _format_personality(personality: Dict[str, Any]) -> str:
    """Formate les traits de personnalité pour le prompt."""
    if not personality:
        return "Personnalité standard"
    
    if isinstance(personality, dict):
        traits = []
        for key, value in personality.items():
            if isinstance(value, (int, float)):
                if value > 0.7:
                    traits.append(f"{key} élevé")
                elif value < 0.3:
                    traits.append(f"{key} faible")
            else:
                traits.append(f"{key}: {value}")
        return ", ".join(traits) if traits else "Équilibré"
    
    return str(personality)


def _generate_fallback_response(state: CharacterState) -> str:
    """Génère une réponse de fallback si le LLM échoue."""
    
    character_name = state["character_name"]
    intent = state["message_intent"]
    emotion = state["current_emotion"]
    
    # Templates de fallback par intention
    fallback_templates = {
        "question": [
            f"C'est une question intéressante... Laissez-moi réfléchir.",
            f"Hmm, votre question me fait réfléchir.",
            f"Interessant que vous me demandiez cela."
        ],
        "greeting": [
            f"Bonjour ! Comment puis-je vous aider ?",
            f"Salutations ! Que puis-je faire pour vous ?",
            f"Bienvenue ! En quoi puis-je vous être utile ?"
        ],
        "request": [
            f"Je vais voir ce que je peux faire pour vous.",
            f"Votre demande mérite considération.",
            f"Laissez-moi réfléchir à votre requête."
        ],
        "general": [
            f"Je vous écoute attentivement.",
            f"Continuez, vous avez mon attention.",
            f"Intéressant, dites-moi en plus."
        ]
    }
    
    # Sélection d'un template approprié
    templates = fallback_templates.get(intent, fallback_templates["general"])
    response = random.choice(templates)
    
    # Ajustement selon l'émotion
    if emotion == "sad":
        response += " *soupire*"
    elif emotion == "happy":
        response += " 😊"
    elif emotion == "angry":
        response = response.replace("!", ".")
    
    return response


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
                "Au revoir, citoyen. Que votre journée soit prospère !",
                "Prenez soin de vous et de notre belle île !",
                "À bientôt ! N'hésitez pas à revenir me voir."
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
                "Au revoir ! Revenez quand vous aurez faim !",
                "À bientôt ! Mes cookies vous attendront !",
                "Prenez soin de vous et mangez bien !"
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

# import random
# from typing import Dict, Any, List
# from ..state.character_state import CharacterState
# from langsmith import traceable
# from echoforge.utils.config import get_config

# config = get_config()
# @traceable
# def generate_simple_response(state: CharacterState) -> CharacterState:
#     """
#     Génère une réponse simple basée sur la personnalité, sans RAG.
    
#     Args:
#         state: État actuel du personnage
        
#     Returns:
#         État mis à jour avec la réponse générée
#     """
#     state["processing_steps"].append("simple_response_generation")
    
#     character_name = state["character_name"]
#     personality = state["personality_traits"]
#     intent = state["message_intent"]
#     emotion = state["current_emotion"]
    
#     # Génération de réponse simple selon l'intention
#     response = _generate_personality_response(
#         character_name, personality, intent, emotion, state["user_message"]
#     )
    
#     state["response"] = response
    
#     # Debug info
#     state["debug_info"]["response_generation"] = {
#         "type": "simple",
#         "intent_used": intent,
#         "emotion_used": emotion,
#         "personality_traits_count": len(personality)
#     }
    
#     return state

# @traceable
# def generate_response(state: CharacterState) -> CharacterState:
#     """
#     Génère une réponse complexe avec ou sans contexte RAG.
    
#     Args:
#         state: État actuel du personnage
        
#     Returns:
#         État mis à jour avec la réponse générée
#     """
#     state["processing_steps"].append("complex_response_generation")
    
#     character_name = state["character_name"]
#     personality = state["personality_traits"]
#     user_message = state["user_message"]
#     rag_results = state["rag_results"]
    
#     # Construction du contexte pour la génération
#     context = _build_response_context(state)
    
#     # Génération de la réponse
#     if rag_results:
#         response = _generate_rag_enhanced_response(
#             character_name, personality, user_message, context, rag_results
#         )
#         response_type = "rag_enhanced"
#     else:
#         response = _generate_contextual_response(
#             character_name, personality, user_message, context
#         )
#         response_type = "contextual"
    
#     state["response"] = response
    
#     # Debug info
#     state["debug_info"]["response_generation"] = {
#         "type": response_type,
#         "rag_results_used": len(rag_results),
#         "context_elements": len(context),
#         "response_length": len(response)
#     }
    
#     return state

# @traceable
# def _generate_personality_response(
#     character_name: str, 
#     personality: Dict[str, Any], 
#     intent: str, 
#     emotion: str, 
#     user_message: str
# ) -> str:
#     """Génère une réponse simple basée uniquement sur la personnalité."""
    
#     # Templates de réponses par personnage et intention
#     response_templates = {
#         "fathira": {
#             "greeting": [
#                 "Bonjour, citoyen ! Comment puis-je vous aider aujourd'hui ?",
#                 "Salutations ! En tant que maire, je suis à votre service.",
#                 "Bienvenue ! Que puis-je faire pour notre communauté ?"
#             ],
#             "farewell": [
#                 "Au revoir ! Revenez quand vous aurez faim !",
#                 "À bientôt ! Mes cookies vous attendront !",
#                 "Prenez soin de vous et mangez bien !"
#             ],
#             "general": [
#                 "En tant que maire, je veille sur tous les habitants de notre île.",
#                 "Notre communauté est ma priorité absolue, citoyen.",
#                 "Parlons de ce qui vous préoccupe."
#             ]
#         },
#         "claude": {
#             "greeting": [
#                 "Salut ! Mon marteau et moi sommes à votre service.",
#                 "Bonjour ! Besoin de quelque chose de réparé ?",
#                 "Hey ! L'atelier est ouvert, que puis-je forger pour vous ?"
#             ],
#             "farewell": [
#                 "À plus ! Si ça casse, vous savez où me trouver !",
#                 "Prenez soin de vos outils !",
#                 "Au revoir ! Que le métal soit avec vous !"
#             ],
#             "general": [
#                 "Je peux réparer à peu près n'importe quoi, mais ça se négocie.",
#                 "Mon atelier a vu passer bien des objets étranges...",
#                 "Le métal ne ment jamais, contrairement aux gens."
#             ]
#         },
#         "azzedine": {
#             "greeting": [
#                 "Bonjour ! Admirez-vous ma dernière création ?",
#                 "Salut ! L'art textile vous intéresse-t-il ?",
#                 "Bienvenue dans mon atelier de beauté !"
#             ],
#             "farewell": [
#                 "Au revoir ! Que la beauté vous accompagne !",
#                 "Partez avec style !",
#                 "À bientôt ! Revenez quand vous aurez développé votre goût esthétique."
#             ],
#             "general": [
#                 "La beauté est dans les détails, et je suis un perfectionniste.",
#                 "Mes tissus sont d'une qualité incomparable sur cette île.",
#                 "L'art véritable demande de la patience et de l'exigence."
#             ]
#         },
#         "roberte": {
#             "greeting": [
#                 "Bonjour ! J'espère que vous avez faim !",
#                 "Salut ! La cuisine sent bon aujourd'hui, n'est-ce pas ?",
#                 "Bienvenue ! Venez-vous pour mes fameux cookies ?"
#             ],
#             "farewell": [
#                 "Au revoir, citoyen. Que votre journée soit prospère !",
#                 "Prenez soin de vous et de notre belle île !",
#                 "À bientôt ! N'hésitez pas à revenir me voir."
#             ],
#             "general": [
#                 "Ma cuisine est ma fierté, ne me dérangez pas pendant que je travaille !",
#                 "Un bon repas réchauffe le cœur et l'âme.",
#                 "Mes cookies sont les meilleurs de l'île, mais il faut les mériter !"
#             ]
#         }
#     }
    
#     # Sélection du template approprié
#     character_templates = response_templates.get(character_name.lower(), response_templates["fathira"])
#     intent_templates = character_templates.get(intent, character_templates["general"])
    
#     # Sélection aléatoire d'une réponse
#     response = random.choice(intent_templates)
    
#     # Modification selon l'émotion si nécessaire
#     if emotion == "angry":
#         response = response.replace("!", ".")
#         response += " (Je suis un peu énervé en ce moment...)"
#     elif emotion == "happy":
#         response += " 😊"
#     elif emotion == "sad":
#         response = response.lower()
#         response += " *soupir*"
    
#     return response

# @traceable
# def _build_response_context(state: CharacterState) -> Dict[str, Any]:
#     """Construit le contexte pour la génération de réponse."""
    
#     context = {
#         "character_name": state["character_name"],
#         "current_emotion": state["current_emotion"],
#         "message_intent": state["message_intent"],
#         "complexity_level": state["complexity_level"],
#         "conversation_turns": len(state["conversation_history"]),
#         "has_rag_data": bool(state["rag_results"])
#     }
    
#     # Ajout du résumé de conversation si disponible
#     if state.get("context_summary"):
#         context["conversation_summary"] = state["context_summary"]
    
#     # Ajout des actions planifiées
#     if state["planned_actions"]:
#         context["planned_actions"] = state["planned_actions"]
    
#     return context

# @traceable
# def _generate_rag_enhanced_response(
#     character_name: str,
#     personality: Dict[str, Any], 
#     user_message: str,
#     context: Dict[str, Any],
#     rag_results: List[Dict[str, Any]]
# ) -> str:
#     """Génère une réponse enrichie par les données RAG."""
    
#     # Extraction des informations les plus pertinentes
#     relevant_info = []
#     for result in rag_results:
#         if result["relevance"] > 0.6:
#             relevant_info.append(result["content"])
    
#     # Templates de réponse par personnage avec intégration RAG
#     rag_templates = {
#         "fathira": [
#             "En tant que maire, je peux vous dire que {knowledge}. Notre communauté a une riche histoire.",
#             "D'après mes connaissances de l'île, {knowledge}. C'est important pour notre communauté.",
#             "Laissez-moi vous expliquer, citoyen : {knowledge}. Voilà ce que je sais sur le sujet."
#         ],
#         "claude": [
#             "D'après mon expérience de forgeron, {knowledge}. J'ai vu beaucoup de choses dans mon atelier.",
#             "Je peux vous dire que {knowledge}. Le métal garde la mémoire des événements.",
#             "Écoutez, {knowledge}. C'est ce que j'ai appris au fil des années."
#         ],
#         "azzedine": [
#             "Avec mon œil artistique, je peux vous révéler que {knowledge}. La beauté révèle la vérité.",
#             "D'un point de vue esthétique, {knowledge}. Chaque détail a son importance.",
#             "Permettez-moi de vous éclairer : {knowledge}. L'art révèle bien des secrets."
#         ],
#         "roberte": [
#             "En cuisinant pour tout le monde, j'ai appris que {knowledge}. On entend beaucoup de choses en cuisine !",
#             "Entre deux plats, je peux vous dire que {knowledge}. Les habitants parlent beaucoup pendant les repas.",
#             "D'après ce que j'ai entendu, {knowledge}. La cuisine est le cœur de l'information sur l'île !"
#         ]
#     }
    
#     # Sélection du template
#     character_templates = rag_templates.get(character_name.lower(), rag_templates["fathira"])
#     template = random.choice(character_templates)
    
#     # Intégration des connaissances
#     if relevant_info:
#         knowledge = relevant_info[0]  # Utilise la plus pertinente
#         response = template.format(knowledge=knowledge)
#     else:
#         # Fallback si pas d'info pertinente
#         response = _generate_contextual_response(character_name, personality, user_message, context)
    
#     return response

# @traceable
# def _generate_contextual_response(
#     character_name: str,
#     personality: Dict[str, Any],
#     user_message: str, 
#     context: Dict[str, Any]
# ) -> str:
#     """Génère une réponse contextuelle sans RAG."""
    
#     # Templates contextuels par personnage
#     contextual_templates = {
#         "fathira": [
#             "C'est une question intéressante, citoyen. En tant que maire, je dois réfléchir à l'impact sur notre communauté.",
#             "Hmm, votre demande mérite considération. Laissez-moi consulter mes connaissances de l'île.",
#             "Voilà une préoccupation légitime. Notre île recèle encore bien des mystères."
#         ],
#         "claude": [
#             "Intéressant... Laissez-moi réfléchir à ça. Mes outils me disent que c'est plus complexe qu'il n'y paraît.",
#             "Votre question me rappelle quelque chose. J'ai déjà travaillé sur des cas similaires.",
#             "Ah, ça c'est du travail de précision ! Donnez-moi un moment pour bien vous expliquer."
#         ],
#         "azzedine": [
#             "Quelle question fascinante ! L'esthétique de votre demande me plaît.",
#             "Votre curiosité témoigne d'un certain raffinement. Laissez-moi vous répondre avec style.",
#             "Ah ! Enfin quelqu'un qui s'intéresse aux détails ! C'est tout un art de bien répondre."
#         ],
#         "roberte": [
#             "Ça me rappelle une recette... Attendez, je vais vous expliquer ça comme il faut !",
#             "Votre question me met l'eau à la bouche ! Enfin, façon de parler...",
#             "Tiens, c'est comme quand je prépare mes cookies : il faut du temps et de la patience !"
#         ]
#     }
    
#     # Sélection et personnalisation
#     character_templates = contextual_templates.get(character_name.lower(), contextual_templates["fathira"])
#     response = random.choice(character_templates)
    
#     # Ajout d'éléments contextuels
#     if context["complexity_level"] == "complex":
#         response += " Cette question demande réflexion..."
    
#     if context["conversation_turns"] > 3:
#         response += " Nous avons déjà bien discuté ensemble !"
    
#     return response