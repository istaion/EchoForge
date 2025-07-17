from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
import re
import json
try:
    from langchain_ollama import OllamaLLM, ChatOllama
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
    
    @abstractmethod
    def get_langchain_llm(self) -> BaseLanguageModel:
        """Retourne l'objet LLM LangChain natif"""
        pass


class GroqToolsWrapper:
    """Wrapper pour gérer les tool calls spécifiques à Groq"""
    
    def __init__(self, llm: ChatGroq, tools: List[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        
    def invoke(self, messages):
        """Invoque le LLM et gère les tool calls Groq"""
        # Ajoute les descriptions des tools au prompt système
        if isinstance(messages, list) and len(messages) > 0:
            # Trouve le message système
            system_message = None
            for i, msg in enumerate(messages):
                if isinstance(msg, tuple) and msg[0] == "system":
                    system_message = msg[1]
                    break
            
            if system_message:
                # Ajoute la description des tools
                tools_desc = "\n\nTOOLS DISPONIBLES:\n"
                for tool_name, tool in self.tools.items():
                    tools_desc += f"- {tool_name}(): {tool.description}\n"
                
                tools_desc += "\nPour utiliser un tool, écris: <function={tool_name}></function>"
                tools_desc += "\nAprès avoir utilisé les tools, donne ta réponse finale en JSON."
                
                # Remplace le message système
                messages[i] = ("system", system_message + tools_desc)
        
        # Invoque le LLM
        response = self.llm.invoke(messages)
        
        # Parse et exécute les tool calls Groq
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        # Détecte les tool calls dans le format Groq
        tool_calls = re.findall(r'<function=([^>]+)></function>', content)
        
        if tool_calls:
            print(f"🔧 Tool calls Groq détectés: {tool_calls}")
            
            # Construit les messages avec les résultats des tools
            messages_with_tools = list(messages)
            messages_with_tools.append(("assistant", content))
            
            # Exécute chaque tool call
            for tool_name in tool_calls:
                if tool_name in self.tools:
                    try:
                        tool_result = self.tools[tool_name].func()
                        print(f"✅ Tool Groq {tool_name} exécuté: {tool_result}")
                        
                        # Ajoute le résultat
                        messages_with_tools.append(("user", f"Résultat de {tool_name}: {tool_result}"))
                    except Exception as e:
                        print(f"❌ Erreur tool Groq {tool_name}: {e}")
                        messages_with_tools.append(("user", f"Erreur {tool_name}: {e}"))
            
            # Demande la réponse finale
            messages_with_tools.append(("user", "Maintenant, donne ta réponse finale en JSON."))
            
            # Nouvelle invocation pour obtenir la réponse finale
            final_response = self.llm.invoke(messages_with_tools)
            return final_response
        
        return response


class GroqProvider(LLMProvider):
    """Provider LLM utilisant Groq avec support tool calls amélioré"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant",
                 temperature: float = 0.7):
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.llm = ChatGroq(
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature
        )
    
    def invoke(self, prompt: str) -> str:
        """Génère une réponse à partir d'un prompt"""
        try:
            response = self.llm.invoke(prompt)
            if isinstance(response, str):
                return response.strip()
            elif isinstance(response, AIMessage):
                return response.content.strip()
            else:
                return str(response).strip()
        except Exception as e:
            return f"❌ Erreur LLM: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle"""
        return {
            "provider": "Groq",
            "model": self.model_name,
            "temperature": self.temperature,
            "tools_support": "custom"  # Indique le support custom
        }
    
    def get_langchain_llm(self) -> BaseLanguageModel:
        """Retourne l'objet LLM LangChain natif"""
        return self.llm
    
    def bind_tools_groq(self, tools: List[Tool]) -> GroqToolsWrapper:
        """Binding spécifique pour Groq"""
        return GroqToolsWrapper(self.llm, tools)


class OpenaiProvider(LLMProvider):
    """Provider LLM utilisant OpenAI"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo",
                 temperature: float = 0.7):
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature
        )
    
    def invoke(self, prompt: str) -> str:
        """Génère une réponse à partir d'un prompt"""
        try:
            response = self.llm.invoke(prompt)
            if isinstance(response, str):
                return response.strip()
            elif isinstance(response, AIMessage):
                return response.content.strip()
            else:
                return str(response).strip()
        except Exception as e:
            return f"❌ Erreur LLM: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle"""
        return {
            "provider": "OpenAI",
            "model": self.model_name,
            "temperature": self.temperature,
            "tools_support": "native"  # Support natif
        }
    
    def get_langchain_llm(self) -> BaseLanguageModel:
        """Retourne l'objet LLM LangChain natif"""
        return self.llm


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
            "temperature": self.temperature,
            "tools_support": "none"  # Pas de support tools
        }
    
    def get_langchain_llm(self) -> BaseLanguageModel:
        """Retourne l'objet LLM LangChain natif"""
        return self.llm


class LLMManager:
    """Gestionnaire centralisé des LLM avec support tools adaptatif"""
    
    def __init__(self, provider: Optional[str] = None):
        from echoforge.utils.config import get_config, reset_config
        reset_config()
        config = get_config()
        self.config = config
        provider = self.config.llm_provider
        
        if provider == "groq":
            self.provider = GroqProvider(
                api_key=self.config.groq_api_key,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature
            )
        elif provider == "openai":
            self.provider = OpenaiProvider(
                api_key=self.config.openai_api_key,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature
            )
        else:
            self.provider = OllamaProvider(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature
            )

    def bind_tools(self, tools: List[Tool]) -> Runnable:
        """
        Lie dynamiquement des tools à l'objet LLM - adaptatif selon le provider
        """
        provider_info = self.provider.get_model_info()
        
        # Support natif (OpenAI)
        if provider_info.get("tools_support") == "native":
            llm = self.get_llm()
            return llm.bind_tools(tools)
        
        # Support custom (Groq)
        elif provider_info.get("tools_support") == "custom":
            if isinstance(self.provider, GroqProvider):
                return self.provider.bind_tools_groq(tools)
        
        # Pas de support (Ollama) - retourne le LLM normal
        else:
            return self.get_llm()

    def get_llm(self) -> BaseLanguageModel:
        """Retourne le LLM LangChain pour compatibilité avec LangChain"""
        return self.provider.get_langchain_llm()
    
    def get_provider(self) -> LLMProvider:
        """Retourne le provider LLM complet"""
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
    
    def supports_tools(self) -> bool:
        """Indique si le provider supporte les tools"""
        return self.provider.get_model_info().get("tools_support") != "none"
    
    def get_tools_support_type(self) -> str:
        """Retourne le type de support des tools"""
        return self.provider.get_model_info().get("tools_support", "none")