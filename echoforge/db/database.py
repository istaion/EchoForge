from sqlmodel import SQLModel, create_engine, Session
from echoforge.utils.config import get_config, reset_config
reset_config()
config = get_config()
DATABASE_URL = config.database_url
engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    from .models import memory
    SQLModel.metadata.create_all(engine)

def get_session():
    return Session(engine)
