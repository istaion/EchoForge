"""
Gestion des embeddings pour le système RAG
"""

from typing import List, Optional
from abc import abstractmethod
from langchain.embeddings.base import Embeddings
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain.embeddings import OllamaEmbeddings


class EmbeddingInterface(Embeddings):
    """Interface abstraite pour les embeddings"""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Génère les embeddings pour une liste de documents"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Génère l'embedding pour une requête"""
        pass

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


class OllamaEmbeddingProvider(EmbeddingInterface):
    """Provider d'embeddings utilisant Ollama"""
    
    def __init__(self, model: str = "paraphrase-multilingual:278m-mpnet-base-v2-fp16"):
        self.model = model
        self._embeddings = OllamaEmbeddings(model=model)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Génère les embeddings pour une liste de documents"""
        return self._embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Génère l'embedding pour une requête"""
        return self._embeddings.embed_query(text)


class EmbeddingManager:
    """Gestionnaire centralisé des embeddings"""
    
    def __init__(self, provider: Optional[EmbeddingInterface] = None):
        self.provider = provider or OllamaEmbeddingProvider()
    
    def get_embeddings(self) -> EmbeddingInterface:
        """Retourne le provider d'embeddings"""
        return self.provider
    
    def set_provider(self, provider: EmbeddingInterface):
        """Change le provider d'embeddings"""
        self.provider = provider
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Génère les embeddings pour une liste de documents"""
        return self.provider.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Génère l'embedding pour une requête"""
        return self.provider.embed_query(text)