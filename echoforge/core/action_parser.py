"""
Parsing et gestion des actions physiques dans les messages
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class ActionParsed:
    """Structure pour les actions physiques parsées"""
    text: str
    actions: List[str]
    raw_message: str


class ActionParser:
    """Parser pour extraire les actions physiques des messages"""
    
    def __init__(self):
        # Pattern pour capturer le texte entre *
        self.action_pattern = r'\*([^*]+)\*'
    
    def parse(self, message: str) -> ActionParsed:
        """Parse les actions physiques dans un message"""
        
        # Extrait les actions entre *
        actions = re.findall(self.action_pattern, message)
        
        # Retire les actions du texte principal
        clean_text = re.sub(self.action_pattern, '', message).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)  # Nettoie les espaces multiples
        
        return ActionParsed(
            text=clean_text,
            actions=actions,
            raw_message=message
        )
    
    def format_actions(self, actions: List[str]) -> str:
        """Formate une liste d'actions pour l'affichage"""
        if not actions:
            return ""
        
        return " ".join([f"*{action}*" for action in actions])
    
    def has_actions(self, message: str) -> bool:
        """Vérifie si un message contient des actions"""
        return bool(re.search(self.action_pattern, message))