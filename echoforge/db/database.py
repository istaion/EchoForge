from sqlmodel import SQLModel, create_engine, Session

DATABASE_URL = "postgresql://echoforge:unicornsoul@localhost:5432/echoforge_db"
engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    from .models import memory
    SQLModel.metadata.create_all(engine)

def get_session():
    return Session(engine)
