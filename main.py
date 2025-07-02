"""
EchoForge - Point d'entrée principal (refactorisé)
"""

# Imports compatibilité
from echoforge.core import EchoForgeRAG, ActionParsed

# Réexport pour compatibilité avec l'ancien code
__all__ = ["EchoForgeRAG", "ActionParsed"]


def main():
    """Point d'entrée principal pour tests"""
    print("🎈 EchoForge Core - Framework de personnages IA")
    
    # Exemple d'utilisation du framework
    from echoforge.utils.config import get_config
    config = get_config()
    
    # Initialisation du système RAG
    rag_system = EchoForgeRAG(
        data_path=str(config.data_path),
        vector_store_path=str(config.vector_store_path),
        embedding_model=config.embedding_model,
        llm_model=config.llm_model
    )
    
    print("✅ Système RAG initialisé!")
    print(f"📊 Info système: {rag_system.get_system_info()}")
    
    # Test basique
    if not rag_system.vector_store_manager.store_exists("world_lore"):
        print("🌍 Construction du vector store du monde...")
        rag_system.build_world_vectorstore()
    
    print("🎯 Framework prêt à l'emploi!")


if __name__ == "__main__":
    main()