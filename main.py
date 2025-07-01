import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import re
from pathlib import Path

# LangChain imports - versions mises à jour
try:
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
except ImportError:
    # Fallback vers les anciennes versions si langchain-ollama n'est pas installé
    from langchain.embeddings import OllamaEmbeddings
    from langchain.llms import Ollama as OllamaLLM

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory

@dataclass
class ActionParsed:
    """Structure pour les actions physiques parsées"""
    text: str
    actions: List[str]
    raw_message: str

@dataclass
class CharacterResponse:
    """Réponse complète d'un personnage"""
    text: str
    actions: List[str]
    triggers: Dict
    character_id: str
    timestamp: str

class EchoForgeRAG:
    """Système RAG principal pour EchoForge"""
    
    def __init__(self, 
                 data_path: str = "./data",
                 vector_store_path: str = "./vector_stores",
                 model_name: str = "llama3.1:8b"):
        
        self.data_path = Path(data_path)
        self.vector_store_path = Path(vector_store_path)
        self.model_name = model_name
        
        # Initialisation des composants
        self.embeddings = OllamaEmbeddings(
            model="paraphrase-multilingual:278m-mpnet-base-v2-fp16"
        )
        self.llm = OllamaLLM(model=model_name, temperature=0.7)
        
        # Stores vectoriels
        self.world_vectorstore = None
        self.character_vectorstores = {}
        
        # Text splitter pour chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,      # Chunks courts pour précision
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "]
        )
        
        self._initialize_vectorstores()
    
    def _initialize_vectorstores(self):
        """Initialise les vector stores"""
        self.vector_store_path.mkdir(exist_ok=True)
        
        # Charge ou crée le vector store du monde
        world_store_path = self.vector_store_path / "world_lore"
        if world_store_path.exists():
            self.world_vectorstore = FAISS.load_local(
                str(world_store_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("🌍 Vector store du monde non trouvé. Utilisez build_world_vectorstore()")
    
    def parse_actions(self, message: str) -> ActionParsed:
        """Parse les actions physiques dans un message"""
        # Regex pour capturer le texte entre *
        action_pattern = r'\*([^*]+)\*'
        actions = re.findall(action_pattern, message)
        
        # Retire les actions du texte principal
        clean_text = re.sub(action_pattern, '', message).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)  # Nettoie les espaces multiples
        
        return ActionParsed(
            text=clean_text,
            actions=actions,
            raw_message=message
        )
    
    def build_world_vectorstore(self, world_data_path: str = None):
        """Construit le vector store du lore du monde"""
        if world_data_path is None:
            world_data_path = self.data_path / "world_lore"
        
        world_data_path = Path(world_data_path)
        
        if not world_data_path.exists():
            print(f"❌ Dossier {world_data_path} introuvable")
            # Créer un document par défaut pour éviter l'erreur
            documents = [Document(
                page_content="Île mystérieuse avec des habitants accueillants.",
                metadata={"source": "default", "type": "world_lore"}
            )]
        else:
            documents = []
            
            # Charge tous les fichiers JSON du lore
            for json_file in world_data_path.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Convertit chaque entrée en document
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
        
        # Chunking des documents
        chunks = self.text_splitter.split_documents(documents)
        
        # Création du vector store
        self.world_vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Sauvegarde
        world_store_path = self.vector_store_path / "world_lore"
        self.world_vectorstore.save_local(str(world_store_path))
        
        print(f"✅ Vector store du monde créé avec {len(chunks)} chunks")
    
    def build_character_vectorstore(self, character_id: str):
        """Construit le vector store d'un personnage"""
        char_data_path = self.data_path / "characters" / character_id
        
        if not char_data_path.exists():
            print(f"❌ Dossier personnage {char_data_path} introuvable")
            # Créer des documents par défaut pour éviter l'erreur
            documents = [
                Document(
                    page_content=f"Je suis {character_id}, un habitant de l'île.",
                    metadata={
                        "character_id": character_id,
                        "type": "default",
                        "source": "default"
                    }
                )
            ]
        else:
            documents = []
            
            # Charge les données du personnage
            for json_file in char_data_path.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Traite selon le type de fichier
                    if json_file.stem == "memories":
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
                                    # Si la mémoire est un dictionnaire, convertir en texte
                                    memory_text = self._format_world_item(memory, "memory")
                                    documents.append(Document(
                                        page_content=memory_text,
                                        metadata={
                                            "character_id": character_id,
                                            "type": "memory",
                                            "source": "memories"
                                        }
                                    ))
                    
                    elif json_file.stem == "relationships":
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
                    
                    elif json_file.stem == "secrets":
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
                                    # Si le secret est un dictionnaire, convertir en texte
                                    secret_text = self._format_world_item(secret, "secret")
                                    documents.append(Document(
                                        page_content=secret_text,
                                        metadata={
                                            "character_id": character_id,
                                            "type": "secret",
                                            "source": "secrets"
                                        }
                                    ))
                    
                except Exception as e:
                    print(f"⚠️ Erreur lors du chargement de {json_file}: {e}")
            
            if not documents:
                print(f"❌ Aucun document trouvé pour {character_id}, création d'un document par défaut")
                documents = [
                    Document(
                        page_content=f"Je suis {character_id}, un habitant de l'île.",
                        metadata={
                            "character_id": character_id,
                            "type": "default",
                            "source": "default"
                        }
                    )
                ]
        
        # Chunking (plus fin pour les données personnelles)
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,  # Plus petit pour les souvenirs
            chunk_overlap=25
        )
        chunks = char_splitter.split_documents(documents)
        
        # Création du vector store
        char_vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Sauvegarde
        char_store_path = self.vector_store_path / f"character_{character_id}"
        char_vectorstore.save_local(str(char_store_path))
        
        # Cache en mémoire
        self.character_vectorstores[character_id] = char_vectorstore
        
        print(f"✅ Vector store de {character_id} créé avec {len(chunks)} chunks")
    
    def load_character_vectorstore(self, character_id: str):
        """Charge le vector store d'un personnage"""
        if character_id in self.character_vectorstores:
            return self.character_vectorstores[character_id]
        
        char_store_path = self.vector_store_path / f"character_{character_id}"
        
        if not char_store_path.exists():
            print(f"❌ Vector store de {character_id} introuvable. Construisez-le d'abord.")
            return None
        
        vectorstore = FAISS.load_local(
            str(char_store_path), 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.character_vectorstores[character_id] = vectorstore
        return vectorstore
    
    def _format_world_item(self, item: Dict, category: str) -> str:
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
        
        # Ajoute d'autres champs pertinents
        for key, value in item.items():
            if key not in ["name", "description", "id"] and isinstance(value, (str, int, float)):
                text_parts.append(f"{key.title()}: {value}")
            elif key not in ["name", "description", "id"] and isinstance(value, list):
                text_parts.append(f"{key.title()}: {', '.join(map(str, value))}")
        
        return "\n".join(text_parts)
    
    def retrieve_world_context(self, query: str, top_k: int = 3) -> List[str]:
        """Récupère le contexte du monde pour une requête"""
        if not self.world_vectorstore:
            return []
        
        try:
            docs = self.world_vectorstore.similarity_search(query, k=top_k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération du contexte monde: {e}")
            return []
    
    def retrieve_character_context(self, query: str, character_id: str, top_k: int = 5) -> List[str]:
        """Récupère le contexte d'un personnage pour une requête"""
        vectorstore = self.load_character_vectorstore(character_id)
        if not vectorstore:
            return []
        
        try:
            docs = vectorstore.similarity_search(query, k=top_k)
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
        
        # Template de prompt
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

# Exemple d'utilisation
def example_usage():
    """Exemple d'utilisation du système RAG"""
    
    # Initialisation
    rag_system = EchoForgeRAG()
    
    # Construction des vector stores (à faire une fois)
    rag_system.build_world_vectorstore()
    rag_system.build_character_vectorstore("fathira")
    
    # Simulation d'une interaction
    user_input = "Bonjour Fathira ! *salue respectueusement* Pouvez-vous m'aider ?"
    character_id = "fathira"
    
    # Données du personnage (normalement chargées depuis un fichier)
    character_data = {
        "name": "Fathira",
        "role": "Maire de l'île",
        "personality": "Diplomatique et protectrice",
        "speech_style": "Formel mais chaleureux",
        "backstory": "Maire élue depuis 10 ans, garde les secrets de l'île"
    }
    
    # Parse le message
    parsed_input = rag_system.parse_actions(user_input)
    print(f"Message parsé: {parsed_input.text}")
    print(f"Actions: {parsed_input.actions}")
    
    # Récupération des contextes
    world_context = rag_system.retrieve_world_context(parsed_input.text)
    character_context = rag_system.retrieve_character_context(parsed_input.text, character_id)
    
    # Création du prompt
    prompt = rag_system.create_character_prompt(
        character_data, world_context, character_context, parsed_input
    )
    
    print(f"\n📝 Prompt généré:\n{prompt}")

if __name__ == "__main__":
    example_usage()