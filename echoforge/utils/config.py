"""
Configuration centralisée pour EchoForge avec support mémoire avancée
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
    ollama_model: str = Field(default="llama3.1:8b", description="Modèle Ollama")
    groq_model: str = Field(default="llama-3.1-8b-instant", description="Modèle Groq")
    openai_model: str = Field(default="gpt-3.5-turbo", description="Modèle OpenAI")
    llm_model: str = Field(default="llama-3.1-8b-instant", description="Modèle par défaut")
    llm_provider: str = Field(default="groq", description="Provider LLM (ollama, groq, openai)")
    llm_temperature: float = Field(default=0.7, description="Température du LLM")
    
    # Configuration RAG
    chunk_size: int = Field(default=300, description="Taille des chunks généraux")
    chunk_overlap: int = Field(default=50, description="Chevauchement des chunks")
    character_chunk_size: int = Field(default=150, description="Taille des chunks pour personnages")
    character_chunk_overlap: int = Field(default=25, description="Chevauchement chunks personnages")
    
    # Paramètres de récupération
    top_k_world: int = Field(default=3, description="Nombre de docs monde à récupérer")
    top_k_character: int = Field(default=5, description="Nombre de docs personnage à récupérer")
    
    # 🆕 Configuration avancée de la mémoire
    max_messages_without_summary: int = Field(
        default=60, 
        description="Nombre max de messages avant résumé automatique"
    )
    keep_recent_messages: int = Field(
        default=20, 
        description="Nombre de messages récents à garder après résumé"
    )
    max_history_size: int = Field(
        default=100, 
        description="Taille max de l'historique avant nettoyage forcé"
    )
    summary_max_token_limit: int = Field(
        default=2000,
        description="Limite de tokens pour les résumés ConversationSummaryMemory"
    )
    memory_search_limit: int = Field(
        default=5,
        description="Nombre max de résultats lors de recherches mémoire"
    )
    auto_backup_messages: bool = Field(
        default=True,
        description="Sauvegarde automatique des messages en DB"
    )
    
    # 🆕 Configuration des déclencheurs de résumé
    bye_trigger_threshold: float = Field(
        default=0.7,
        description="Seuil de confiance pour déclencher un résumé sur 'bye'"
    )
    conversation_cleanup_interval: int = Field(
        default=24,
        description="Intervalle en heures pour le nettoyage automatique"
    )
    max_summaries_per_thread: int = Field(
        default=50,
        description="Nombre max de résumés par thread avant archivage"
    )
    
    # 🆕 Configuration LangGraph Checkpointer
    enable_checkpoints: bool = Field(
        default=True,
        description="Activer les checkpoints LangGraph"
    )
    checkpoint_save_frequency: int = Field(
        default=10,
        description="Fréquence de sauvegarde des checkpoints (en messages)"
    )
    max_checkpoints_per_thread: int = Field(
        default=100,
        description="Nombre max de checkpoints par thread"
    )
    
    # Base de données
    database_url: str = Field(
        default="postgresql+psycopg2://echoforge:devpass@localhost/echoforge_db",
        description="URL de connexion à la base de données"
    )
    database_echo: bool = Field(default=False, description="Activer les logs SQL")
    database_pool_size: int = Field(default=5, description="Taille du pool de connexions")
    database_max_overflow: int = Field(default=10, description="Débordement max du pool")
    
    # API Keys
    groq_api_key: Optional[str] = Field(default=None, description="Clé API Groq")
    openai_api_key: Optional[str] = Field(default=None, description="Clé API OpenAI")
    
    # LangSmith Configuration
    langsmith_tracing: bool = Field(default=True, description="Activer le tracing LangSmith")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", description="Endpoint LangSmith")
    langsmith_api_key: Optional[str] = Field(default=None, description="Clé API LangSmith")
    langsmith_project: str = Field(default="echoforge-dev", description="Nom du projet LangSmith")

    # Debug et monitoring
    debug: bool = Field(default=False, description="Activer le debug")
    enable_performance_monitoring: bool = Field(default=True, description="Monitoring des performances")
    log_conversation_analytics: bool = Field(default=True, description="Analytics des conversations")
    
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
        # Configure les API keys depuis l'environnement
        self._setup_api_keys()
        # Configure LangSmith automatiquement
        self._setup_langsmith()
        # 🆕 Validation des paramètres mémoire
        self._validate_memory_config()
        if self.llm_provider == "groq":
            self.llm_model = self.groq_model
        elif self.llm_provider == "openai":
            self.llm_model = self.openai_model
        else:
            self.llm_model = self.ollama_model
    
    def _validate_memory_config(self):
        """Valide la cohérence des paramètres de mémoire."""
        if self.keep_recent_messages >= self.max_messages_without_summary:
            print("⚠️ ATTENTION: keep_recent_messages >= max_messages_without_summary")
            print(f"   Ajustement: keep_recent_messages={self.max_messages_without_summary // 3}")
            self.keep_recent_messages = self.max_messages_without_summary // 3
        
        if self.max_history_size < self.max_messages_without_summary:
            print("⚠️ ATTENTION: max_history_size < max_messages_without_summary")
            print(f"   Ajustement: max_history_size={self.max_messages_without_summary * 2}")
            self.max_history_size = self.max_messages_without_summary * 2
    
    def _setup_langsmith(self):
        """Configure les variables d'environnement LangSmith"""
        if self.langsmith_tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = str(self.langsmith_tracing).lower()
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
    
    def get_memory_config(self) -> dict:
        """Retourne la configuration spécifique à la mémoire."""
        return {
            "max_messages_without_summary": self.max_messages_without_summary,
            "keep_recent_messages": self.keep_recent_messages,
            "max_history_size": self.max_history_size,
            "summary_max_token_limit": self.summary_max_token_limit,
            "memory_search_limit": self.memory_search_limit,
            "auto_backup_messages": self.auto_backup_messages,
            "bye_trigger_threshold": self.bye_trigger_threshold,
            "conversation_cleanup_interval": self.conversation_cleanup_interval,
            "max_summaries_per_thread": self.max_summaries_per_thread
        }
    
    def get_checkpoint_config(self) -> dict:
        """Retourne la configuration des checkpoints."""
        return {
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_save_frequency": self.checkpoint_save_frequency,
            "max_checkpoints_per_thread": self.max_checkpoints_per_thread
        }
    
    def debug_info(self) -> str:
        """Retourne des informations de debug sur la configuration"""
        memory_config = self.get_memory_config()
        
        return f"""
🔧 Configuration EchoForge:
  - LLM Provider: {self.llm_provider}
  - LLM Model: {self.groq_model}
  - Temperature: {self.llm_temperature}
  - Groq API Key: {'✅ Configurée' if self.groq_api_key else '❌ Manquante'}
  - LangSmith: {'✅ Activé' if self.langsmith_tracing and self.langsmith_api_key else '❌ Désactivé'}
  - Data Path: {self.data_path}
  - Vector Store Path: {self.vector_store_path}

🧠 Configuration Mémoire:
  - Messages max avant résumé: {memory_config['max_messages_without_summary']}
  - Messages gardés après résumé: {memory_config['keep_recent_messages']}
  - Sauvegarde auto: {'✅' if memory_config['auto_backup_messages'] else '❌'}
  - Checkpoints: {'✅' if self.enable_checkpoints else '❌'}

💾 Base de données:
  - URL: {self.database_url.split('@')[1] if '@' in self.database_url else 'localhost'}
  - Pool size: {self.database_pool_size}
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
