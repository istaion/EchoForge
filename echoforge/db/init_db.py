from sqlmodel import SQLModel
from echoforge.db.database import engine
from echoforge.db.models.memory import ConversationSummary, ConversationMessage, GameSession, SessionEvent

def init_db():
    print("📦 Création des tables SQLModel...")
    SQLModel.metadata.create_all(engine)
    print("✅ Tables créées.")

if __name__ == "__main__":
    init_db()
