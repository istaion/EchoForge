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
    llm_model: str = Field(default="llama-3.1-8b-instant", description="Modèle LLM, (ollama : llama3.1:8b, groq : llama-3.1-8b-instant)")
    llm_provider: str = Field(default="groq", description="Provider LLM (ollama, groq)")
    
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
    
    # API Keys
    groq_api_key: Optional[str] = Field(default=None, description="Clé API Groq")
    openai_api_key: Optional[str] = Field(default=None, description="Clé API OpenAI")
    
    # 🆕 LangSmith Configuration
    langsmith_tracing: bool = Field(default=True, description="Activer le tracing LangSmith")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", description="Endpoint LangSmith")
    langsmith_api_key: Optional[str] = Field(default=None, description="Clé API LangSmith")
    langsmith_project: str = Field(default="echoforge-dev", description="Nom du projet LangSmith")
    
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
        # 🆕 Configure les API keys depuis l'environnement si pas définies
        self._setup_api_keys()
        # 🆕 Configure LangSmith automatiquement
        self._setup_langsmith()
    
    def _setup_langsmith(self):
        """Configure les variables d'environnement LangSmith"""
        if self.langsmith_tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ["LANGCHAIN_ENDPOINT"] = self.langsmith_endpoint
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
            
            if self.langsmith_api_key:
                os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key

    def _setup_api_keys(self):
        """Configure les clés API depuis les variables d'environnement"""
        # Groq API Key
        if not self.groq_api_key:
            self.groq_api_key = (
                os.getenv("GROQ_API_KEY") or 
                os.getenv("ECHOFORGE_GROQ_API_KEY")
            )
        
        # OpenAI API Key  
        if not self.openai_api_key:
            self.openai_api_key = (
                os.getenv("OPENAI_API_KEY") or 
                os.getenv("ECHOFORGE_OPENAI_API_KEY")
            )
        
        # LangSmith API Key
        if not self.langsmith_api_key:
            self.langsmith_api_key = (
                os.getenv("LANGSMITH_API_KEY") or 
                os.getenv("LANGCHAIN_API_KEY") or
                os.getenv("ECHOFORGE_LANGSMITH_API_KEY")
            )

    @classmethod
    def from_env_file(cls, env_file: str = ".env") -> "EchoForgeConfig":
        """Charge la configuration depuis un fichier .env"""
        return cls(_env_file=env_file)
    
    def to_dict(self) -> dict:
        """Convertit la configuration en dictionnaire"""
        return self.dict()
    
    def debug_info(self) -> str:
        """Retourne des informations de debug sur la configuration"""
        return f"""
🔧 Configuration EchoForge:
  - LLM Provider: {self.llm_provider}
  - LLM Model: {self.llm_model}
  - Temperature: {self.llm_temperature}
  - Groq API Key: {'✅ Configurée' if self.groq_api_key else '❌ Manquante'}
  - LangSmith: {'✅ Activé' if self.langsmith_tracing and self.langsmith_api_key else '❌ Désactivé'}
  - Data Path: {self.data_path}
  - Vector Store Path: {self.vector_store_path}
        """.strip()


# Instance globale de configuration
_config: Optional[EchoForgeConfig] = None


def get_config() -> EchoForgeConfig:
    """Récupère la configuration globale"""
    global _config
    if _config is None:
        _config = EchoForgeConfig.from_env_file()
    return _config


def set_config(config: EchoForgeConfig):
    """Définit la configuration globale"""
    global _config
    _config = config


def reset_config():
    """Remet à zéro la configuration"""
    global _config
    _config = None
