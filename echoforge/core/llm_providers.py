from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain.llms import Ollama as OllamaLLM


class LLMProvider(ABC):
    """Interface abstraite pour les providers LLM"""
    
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Génère une réponse à partir d'un prompt"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle"""
        pass


class OllamaProvider(LLMProvider):
    """Provider LLM utilisant Ollama"""
    
    def __init__(self, 
                 model: str = "llama3.1:8b", 
                 temperature: float = 0.7,
                 **kwargs):
        self.model_name = model
        self.temperature = temperature
        self.llm = OllamaLLM(
            model=model, 
            temperature=temperature,
            **kwargs
        )
    
    def invoke(self, prompt: str) -> str:
        """Génère une réponse à partir d'un prompt"""
        try:
            response = self.llm.invoke(prompt)
            return response.strip() if isinstance(response, str) else str(response).strip()
        except Exception as e:
            return f"❌ Erreur LLM: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle"""
        return {
            "provider": "Ollama",
            "model": self.model_name,
            "temperature": self.temperature
        }


class GroqProvider(LLMProvider):
    """Provider LLM utilisant Groq (pour intégration future)"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model_name = model
        # TODO: Implémenter avec groq
    
    def invoke(self, prompt: str) -> str:
        """Génère une réponse à partir d'un prompt"""
        # TODO: Implémenter l'appel Groq
        raise NotImplementedError("Groq provider pas encore implémenté")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle"""
        return {
            "provider": "Groq",
            "model": self.model_name
        }


class LLMManager:
    """Gestionnaire centralisé des LLM"""
    
    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or OllamaProvider()
    
    def get_llm(self) -> LLMProvider:
        """Retourne le provider LLM actuel"""
        return self.provider
    
    def set_provider(self, provider: LLMProvider):
        """Change le provider LLM"""
        self.provider = provider
    
    def invoke(self, prompt: str) -> str:
        """Génère une réponse à partir d'un prompt"""
        return self.provider.invoke(prompt)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle actuel"""
        return self.provider.get_model_info()