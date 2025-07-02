"""
Gestion des vector stores FAISS
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .embeddings import EmbeddingInterface


class VectorStoreInterface(ABC):
    """Interface abstraite pour les vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]):
        """Ajoute des documents au vector store"""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Recherche de documents similaires"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Sauvegarde le vector store"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Charge un vector store"""
        pass


class FAISSVectorStore(VectorStoreInterface):
    """Implementation FAISS du vector store"""
    
    def __init__(self, embeddings: EmbeddingInterface):
        self.embeddings = embeddings
        self._store: Optional[FAISS] = None
    
    def add_documents(self, documents: List[Document]):
        """Ajoute des documents au vector store"""
        if not documents:
            return
        
        if self._store is None:
            self._store = FAISS.from_documents(documents, self.embeddings)
        else:
            self._store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Recherche de documents similaires"""
        if self._store is None:
            return []
        
        try:
            return self._store.similarity_search(query, k=k)
        except Exception as e:
            print(f"⚠️ Erreur lors de la recherche: {e}")
            return []
    
    def save(self, path: str):
        """Sauvegarde le vector store"""
        if self._store is None:
            raise ValueError("Aucun vector store à sauvegarder")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._store.save_local(path)
    
    def load(self, path: str):
        """Charge un vector store"""
        if not Path(path).exists():
            raise FileNotFoundError(f"Vector store non trouvé: {path}")
        
        self._store = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def exists(self) -> bool:
        """Vérifie si le vector store existe"""
        return self._store is not None


class VectorStoreManager:
    """Gestionnaire centralisé des vector stores"""
    
    def __init__(self, 
                 embeddings: EmbeddingInterface,
                 store_path: Path = Path("./vector_stores")):
        self.embeddings = embeddings
        self.store_path = store_path
        self.stores: Dict[str, VectorStoreInterface] = {}
        
        # Configuration du text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "]
        )
        
        # Splitter spécialisé pour les données de personnages
        self.character_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=25
        )
    
    def create_store(self, store_id: str) -> VectorStoreInterface:
        """Crée un nouveau vector store"""
        store = FAISSVectorStore(self.embeddings)
        self.stores[store_id] = store
        return store
    
    def get_store(self, store_id: str) -> Optional[VectorStoreInterface]:
        """Récupère un vector store existant"""
        if store_id in self.stores:
            return self.stores[store_id]
        
        # Tente de charger depuis le disque
        store_path = self.store_path / store_id
        if store_path.exists():
            store = FAISSVectorStore(self.embeddings)
            store.load(str(store_path))
            self.stores[store_id] = store
            return store
        
        return None
    
    def get_or_create_store(self, store_id: str) -> VectorStoreInterface:
        """Récupère ou crée un vector store"""
        store = self.get_store(store_id)
        if store is None:
            store = self.create_store(store_id)
        return store
    
    def save_store(self, store_id: str):
        """Sauvegarde un vector store"""
        if store_id not in self.stores:
            raise ValueError(f"Store {store_id} non trouvé")
        
        store_path = self.store_path / store_id
        self.stores[store_id].save(str(store_path))
    
    def save_all_stores(self):
        """Sauvegarde tous les vector stores"""
        for store_id in self.stores:
            try:
                self.save_store(store_id)
            except Exception as e:
                print(f"⚠️ Erreur sauvegarde {store_id}: {e}")
    
    def chunk_documents(self, documents: List[Document], 
                       document_type: str = "general") -> List[Document]:
        """Découpe les documents en chunks"""
        
        if document_type == "character":
            return self.character_splitter.split_documents(documents)
        else:
            return self.text_splitter.split_documents(documents)
    
    def store_exists(self, store_id: str) -> bool:
        """Vérifie si un vector store existe"""
        if store_id in self.stores:
            return self.stores[store_id].exists()
        
        store_path = self.store_path / store_id
        return store_path.exists()
    
    def list_stores(self) -> List[str]:
        """Liste tous les vector stores disponibles"""
        stores = set(self.stores.keys())
        
        # Ajoute les stores sur disque
        if self.store_path.exists():
            for path in self.store_path.iterdir():
                if path.is_dir():
                    stores.add(path.name)
        
        return sorted(list(stores))