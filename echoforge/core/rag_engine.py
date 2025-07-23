"""
Moteur RAG principal d'EchoForge
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from langchain.schema import Document

from .embeddings import EmbeddingManager, EmbeddingInterface
from .vector_stores import VectorStoreManager, VectorStoreInterface  
from .llm_providers import LLMManager, LLMProvider
from .action_parser import ActionParser, ActionParsed


class EchoForgeRAG:
    """Système RAG principal pour EchoForge"""
    
    def __init__(self, 
                 data_path: str = "./data",
                 vector_store_path: str = "./vector_stores",
                 embedding_model: str = "paraphrase-multilingual:278m-mpnet-base-v2-fp16",
                 llm_model: str = "llama3.1:8b",
                 llm_provider: Optional[str] = None,
                 embedding_provider: Optional[EmbeddingInterface] = None):
        
        # Configuration des chemins
        self.data_path = Path(data_path)
        self.vector_store_path = Path(vector_store_path)
        
        # Initialisation des composants
        self.embedding_manager = EmbeddingManager(embedding_provider)
        self.llm_manager = LLMManager(llm_provider)
        self.vector_store_manager = VectorStoreManager(
            self.embedding_manager.get_embeddings(),
            self.vector_store_path
        )
        self.action_parser = ActionParser()
        
        # Configuration des modèles par défaut si aucun provider fourni
        if llm_provider is None:
            from .llm_providers import OllamaProvider
            self.llm_manager.set_provider(OllamaProvider(llm_model))
        
        if embedding_provider is None:
            from .embeddings import OllamaEmbeddingProvider
            self.embedding_manager.set_provider(OllamaEmbeddingProvider(embedding_model))
    
    @property
    def llm(self) -> LLMProvider:
        """Accès direct au LLM pour compatibilité"""
        return self.llm_manager.get_llm()
    
    def parse_actions(self, message: str) -> ActionParsed:
        """Parse les actions physiques dans un message"""
        return self.action_parser.parse(message)
    
    def build_world_vectorstore(self, world_data_path: Optional[str] = None):
        """Construit le vector store du lore du monde"""
        if world_data_path is None:
            world_data_path = self.data_path / "world_lore"
        
        world_data_path = Path(world_data_path)
        documents = self._load_world_documents(world_data_path)
        
        # Chunking des documents
        chunks = self.vector_store_manager.chunk_documents(documents, "general")
        
        # Création/mise à jour du vector store
        store = self.vector_store_manager.get_or_create_store("world_lore")
        store.add_documents(chunks)
        
        # Sauvegarde
        self.vector_store_manager.save_store("world_lore")
        
        print(f"✅ Vector store du monde créé avec {len(chunks)} chunks")
    
    def build_character_vectorstore(self, character_id: str):
        """Construit le vector store d'un personnage"""
        char_data_path = self.data_path / "characters" / character_id
        documents = self._load_character_documents(char_data_path, character_id)
        
        # Chunking spécialisé pour les personnages
        chunks = self.vector_store_manager.chunk_documents(documents, "character")
        
        # Création/mise à jour du vector store
        store_id = f"character_{character_id}"
        store = self.vector_store_manager.get_or_create_store(store_id)
        store.add_documents(chunks)
        
        # Sauvegarde
        self.vector_store_manager.save_store(store_id)
        
        print(f"✅ Vector store de {character_id} créé avec {len(chunks)} chunks")
    
    def retrieve_world_context(self, query: str, top_k: int = 3) -> List[str]:
        """Récupère le contexte du monde pour une requête"""
        store = self.vector_store_manager.get_store("world_lore")
        if not store:
            return []
        
        try:
            docs = store.similarity_search(query, k=top_k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération du contexte monde: {e}")
            return []
    
    def retrieve_character_context(self, query: str, character_id: str, top_k: int = 5) -> List[str]:
        """Récupère le contexte d'un personnage pour une requête"""
        store_id = f"character_{character_id}"
        store = self.vector_store_manager.get_store(store_id)
        
        print(f"🔍 Recherche dans {store_id} avec query: '{query}'")
        
        if not store:
            print(f"❌ Store {store_id} non trouvé!")
            return []
        
        try:
            docs = store.similarity_search(query, k=top_k)
            print(f"📄 {len(docs)} documents trouvés")
            for doc in docs:
                print(f"  - {doc.page_content[:100]}...")
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération du contexte de {character_id}: {e}")
            return []
        
        try:
            docs = store.similarity_search(query, k=top_k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération du contexte de {character_id}: {e}")
            return []
    
    def create_character_prompt(self, 
                              character_data: Dict,
                              world_context: List[str],
                              character_context: List[str],
                              parsed_input: ActionParsed,
                              conversation_history: str = "") -> str:
        """Crée le prompt complet pour le personnage"""
        
        prompt_template = """Tu es {character_name}, {character_role}.

PERSONNALITÉ: {personality}
STYLE DE PAROLE: {speech_style}
HISTOIRE: {backstory}

CONTEXTE DU MONDE:
{world_context}

TES SOUVENIRS ET RELATIONS:
{character_context}

HISTORIQUE RÉCENT:
{conversation_history}

MESSAGE DU JOUEUR: {user_message}
ACTIONS PHYSIQUES DU JOUEUR: {user_actions}

INSTRUCTIONS:
1. Réponds en restant fidèle à ton personnage
2. Utilise ton style de parole unique
3. Intègre les éléments pertinents du contexte
4. Si tu fais des actions physiques, mets-les entre *
5. Garde tes réponses courtes (2-3 phrases max)
6. Réagis aux actions physiques du joueur si approprié

RÉPONSE:"""

        return prompt_template.format(
            character_name=character_data.get("name", "Inconnu"),
            character_role=character_data.get("role", "Habitant"),
            personality=character_data.get("personality", "Amical"),
            speech_style=character_data.get("speech_style", "Normal"),
            backstory=character_data.get("backstory", "Aucune histoire connue"),
            world_context="\n".join(world_context) if world_context else "Aucun contexte disponible",
            character_context="\n".join(character_context) if character_context else "Aucun souvenir spécifique",
            conversation_history=conversation_history,
            user_message=parsed_input.text,
            user_actions=", ".join(parsed_input.actions) if parsed_input.actions else "Aucune"
        )
    
    def _load_world_documents(self, world_data_path: Path) -> List[Document]:
        """Charge les documents du lore du monde"""
        if not world_data_path.exists():
            print(f"❌ Dossier {world_data_path} introuvable")
            return [Document(
                page_content="Île mystérieuse avec des habitants accueillants.",
                metadata={"source": "default", "type": "world_lore"}
            )]
        
        documents = []
        
        for json_file in world_data_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        doc_text = self._format_world_item(item, json_file.stem)
                        documents.append(Document(
                            page_content=doc_text,
                            metadata={
                                "source": json_file.stem,
                                "type": "world_lore",
                                "item_id": item.get("id", item.get("name", "unknown"))
                            }
                        ))
                elif isinstance(data, dict):
                    doc_text = self._format_world_item(data, json_file.stem)
                    documents.append(Document(
                        page_content=doc_text,
                        metadata={
                            "source": json_file.stem,
                            "type": "world_lore"
                        }
                    ))
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de {json_file}: {e}")
        
        if not documents:
            print("❌ Aucun document trouvé pour le lore du monde, création d'un document par défaut")
            documents = [Document(
                page_content="Île mystérieuse avec des habitants accueillants.",
                metadata={"source": "default", "type": "world_lore"}
            )]
        
        return documents
    
    def _load_character_documents(self, char_data_path: Path, character_id: str) -> List[Document]:
        """Charge les documents d'un personnage"""
        if not char_data_path.exists():
            print(f"❌ Dossier personnage {char_data_path} introuvable")
            return [Document(
                page_content=f"Je suis {character_id}, un habitant de l'île.",
                metadata={
                    "character_id": character_id,
                    "type": "default",
                    "source": "default"
                }
            )]
        
        documents = []
        
        for json_file in char_data_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Traite selon le type de fichier
                if json_file.stem == "memories":
                    documents.extend(self._process_memories(data, character_id))
                elif json_file.stem == "relationships":
                    documents.extend(self._process_relationships(data, character_id))
                elif json_file.stem == "secrets":
                    documents.extend(self._process_secrets(data, character_id))
                
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de {json_file}: {e}")
        
        if not documents:
            print(f"❌ Aucun document trouvé pour {character_id}, création d'un document par défaut")
            documents = [Document(
                page_content=f"Je suis {character_id}, un habitant de l'île.",
                metadata={
                    "character_id": character_id,
                    "type": "default",
                    "source": "default"
                }
            )]
        
        return documents
    
    def _format_world_item(self, item: Any, category: str) -> str:
        """Formate un élément du lore en texte"""
        if isinstance(item, str):
            return item
        
        if not isinstance(item, dict):
            return str(item)
        
        text_parts = [f"Catégorie: {category}"]
        
        if "name" in item:
            text_parts.append(f"Nom: {item['name']}")
        
        if "description" in item:
            text_parts.append(f"Description: {item['description']}")
        
        for key, value in item.items():
            if key not in ["name", "description", "id"] and isinstance(value, (str, int, float)):
                text_parts.append(f"{key.title()}: {value}")
            elif key not in ["name", "description", "id"] and isinstance(value, list):
                text_parts.append(f"{key.title()}: {', '.join(map(str, value))}")
        
        return "\n".join(text_parts)
    
    def _process_memories(self, data: Any, character_id: str) -> List[Document]:
        """Traite les mémoires d'un personnage"""
        documents = []
        memories = data.get("memories", []) if isinstance(data, dict) else data
        
        if isinstance(memories, list):
            for memory in memories:
                if isinstance(memory, str):
                    documents.append(Document(
                        page_content=memory,
                        metadata={
                            "character_id": character_id,
                            "type": "memory",
                            "source": "memories"
                        }
                    ))
                elif isinstance(memory, dict):
                    memory_text = self._format_world_item(memory, "memory")
                    documents.append(Document(
                        page_content=memory_text,
                        metadata={
                            "character_id": character_id,
                            "type": "memory",
                            "source": "memories"
                        }
                    ))
        return documents
    
    def _process_relationships(self, data: Any, character_id: str) -> List[Document]:
        """Traite les relations d'un personnage"""
        documents = []
        relationships = data.get("relationships", {}) if isinstance(data, dict) else {}
        
        for person, relation in relationships.items():
            doc_text = f"Relation avec {person}: {relation}"
            documents.append(Document(
                page_content=doc_text,
                metadata={
                    "character_id": character_id,
                    "type": "relationship",
                    "person": person,
                    "source": "relationships"
                }
            ))
        
        return documents
    
    def _process_secrets(self, data: Any, character_id: str) -> List[Document]:
        """Traite les secrets d'un personnage"""
        documents = []
        secrets = data.get("secrets", []) if isinstance(data, dict) else data
        
        if isinstance(secrets, list):
            for secret in secrets:
                if isinstance(secret, str):
                    documents.append(Document(
                        page_content=secret,
                        metadata={
                            "character_id": character_id,
                            "type": "secret",
                            "source": "secrets"
                        }
                    ))
                elif isinstance(secret, dict):
                    secret_text = self._format_world_item(secret, "secret")
                    documents.append(Document(
                        page_content=secret_text,
                        metadata={
                            "character_id": character_id,
                            "type": "secret",
                            "source": "secrets"
                        }
                    ))
        
        return documents
    
    def get_system_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le système RAG"""
        return {
            "llm": self.llm_manager.get_model_info(),
            "embedding_model": "Configured via EmbeddingManager",
            "vector_stores": self.vector_store_manager.list_stores(),
            "data_path": str(self.data_path),
            "vector_store_path": str(self.vector_store_path)
        }
    
    def rebuild_all_vectorstores(self):
        """Reconstruit tous les vector stores"""
        print("🔄 Reconstruction de tous les vector stores...")
        
        # Reconstruit le monde
        self.build_world_vectorstore()
        
        # Reconstruit les personnages
        characters_path = self.data_path / "characters"
        if characters_path.exists():
            for char_dir in characters_path.iterdir():
                if char_dir.is_dir():
                    print(f"🔄 Reconstruction du vector store pour {char_dir.name}")
                    self.build_character_vectorstore(char_dir.name)
        
        print("✅ Reconstruction terminée!")
