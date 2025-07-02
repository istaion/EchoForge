"""
Configuration centralisée pour EchoForge
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class EchoForgeConfig(BaseSettings):
    """Configuration principale d'EchoForge"""
    
    # Chemins
    data_path: Path = Field(default=Path("./data"), description="Chemin vers les données")
    vector_store_path: Path = Field(default=Path("./vector_stores"), description="Chemin vers les vector stores")
    
    # Modèles
    embedding_model: str = Field(
        default="paraphrase-multilingual:278m-mpnet-base-v2-fp16",
        description="Modèle d'embeddings"
    )
    llm_model: str = Field(default="llama3.1:8b", description="Modèle LLM")
    llm_provider: str = Field(default="ollama", description="Provider LLM (ollama, groq)")
    
    # Configuration RAG
    chunk_size: int = Field(default=300, description="Taille des chunks généraux")
    chunk_overlap: int = Field(default=50, description="Chevauchement des chunks")
    character_chunk_size: int = Field(default=150, description="Taille des chunks pour personnages")
    character_chunk_overlap: int = Field(default=25, description="Chevauchement chunks personnages")
    
    # Paramètres de récupération
    top_k_world: int = Field(default=3, description="Nombre de docs monde à récupérer")
    top_k_character: int = Field(default=5, description="Nombre de docs personnage à récupérer")
    
    # Conversation
    max_conversation_history: int = Field(default=10, description="Historique max de conversation")
    llm_temperature: float = Field(default=0.7, description="Température du LLM")
    
    # API Keys (optionnel)
    groq_api_key: Optional[str] = Field(default=None, description="Clé API Groq")
    openai_api_key: Optional[str] = Field(default=None, description="Clé API OpenAI")
    
    # Interface
    gradio_server_name: str = Field(default="0.0.0.0", description="Nom du serveur Gradio")
    gradio_server_port: int = Field(default=7860, description="Port du serveur Gradio")
    gradio_share: bool = Field(default=False, description="Partage public Gradio")
    
    class Config:
        env_file = ".env"
        env_prefix = "ECHOFORGE_"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Assure-toi que les chemins existent
        self.data_path.mkdir(exist_ok=True)
        self.vector_store_path.mkdir(exist_ok=True)
    
    @classmethod
    def from_env_file(cls, env_file: str = ".env") -> "EchoForgeConfig":
        """Charge la configuration depuis un fichier .env"""
        return cls(_env_file=env_file)
    
    def to_dict(self) -> dict:
        """Convertit la configuration en dictionnaire"""
        return self.dict()


# Instance globale de configuration
_config: Optional[EchoForgeConfig] = None


def get_config() -> EchoForgeConfig:
    """Récupère la configuration globale"""
    global _config
    if _config is None:
        _config = EchoForgeConfig()
    return _config


def set_config(config: EchoForgeConfig):
    """Définit la configuration globale"""
    global _config
    _config = config


def reset_config():
    """Remet à zéro la configuration"""
    global _config
    _config = None