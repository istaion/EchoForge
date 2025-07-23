from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from sqlmodel import Session, select, and_, desc
from echoforge.db.database import get_session
from echoforge.db.models.memory import GameSession, SessionEvent
import json


class DatabaseSessionService:
    """Service de gestion des sessions en base de données."""
    
    @staticmethod
    def create_session(session_id: str, session_name: str, 
                      player_data: dict, characters_data: dict) -> bool:
        """Crée une nouvelle session en base."""
        try:
            with get_session() as session:
                game_session = GameSession(
                    session_id=session_id,
                    session_name=session_name,
                    player_data=player_data,
                    characters_data=characters_data,
                    last_played_at=datetime.utcnow()
                )
                
                session.add(game_session)
                session.commit()
                session.refresh(game_session)
                
                print(f"✅ Session créée en DB: {session_id}")
                return True
                
        except Exception as e:
            print(f"❌ Erreur création session DB: {e}")
            return False
    
    @staticmethod
    def update_session(session_id: str, player_data: dict = None, 
                      characters_data: dict = None, game_state: dict = None) -> bool:
        """Met à jour une session existante."""
        try:
            with get_session() as session:
                stmt = select(GameSession).where(GameSession.session_id == session_id)
                game_session = session.exec(stmt).first()
                
                if not game_session:
                    print(f"⚠️ Session {session_id} non trouvée pour mise à jour")
                    return False
                
                # Mise à jour des données
                if player_data is not None:
                    game_session.player_data = player_data
                
                if characters_data is not None:
                    game_session.characters_data = characters_data
                
                if game_state is not None:
                    game_session.game_state = game_state
                
                # Mise à jour des métadonnées
                game_session.updated_at = datetime.utcnow()
                game_session.last_played_at = datetime.utcnow()
                
                # Calcul des statistiques
                if player_data:
                    # Vérification si le jeu est terminé
                    montgolfiere_status = player_data.get("montgolfiere_status", {})
                    game_session.is_completed = montgolfiere_status.get("fully_operational", False)
                
                session.add(game_session)
                session.commit()
                
                print(f"✅ Session mise à jour en DB: {session_id}")
                return True
                
        except Exception as e:
            print(f"❌ Erreur mise à jour session DB: {e}")
            return False
    
    @staticmethod
    def load_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Charge une session depuis la base."""
        try:
            with get_session() as session:
                stmt = select(GameSession).where(GameSession.session_id == session_id)
                game_session = session.exec(stmt).first()
                
                if not game_session:
                    return None
                
                return {
                    "session_id": game_session.session_id,
                    "session_name": game_session.session_name,
                    "player_data": game_session.player_data,
                    "characters_data": game_session.characters_data,
                    "game_state": game_session.game_state,
                    "created_at": game_session.created_at,
                    "last_played_at": game_session.last_played_at,
                    "is_completed": game_session.is_completed,
                    "total_playtime_seconds": game_session.total_playtime_seconds
                }
                
        except Exception as e:
            print(f"❌ Erreur chargement session DB: {e}")
            return None
    
    @staticmethod
    def list_sessions(limit: int = 50, only_active: bool = True) -> List[Dict[str, Any]]:
        """Liste les sessions disponibles."""
        try:
            with get_session() as session:
                stmt = select(GameSession)
                
                if only_active:
                    stmt = stmt.where(GameSession.is_active == True)
                
                stmt = stmt.order_by(desc(GameSession.last_played_at)).limit(limit)
                
                sessions = session.exec(stmt).all()
                
                result = []
                for game_session in sessions:
                    # Calcul du temps de jeu
                    playtime_minutes = game_session.total_playtime_seconds // 60
                    
                    # Statut du jeu
                    status = "🎯 En cours"
                    if game_session.is_completed:
                        status = "🏆 Terminé"
                    
                    display_name = f"{game_session.session_name} ({status} - {playtime_minutes}min)"
                    
                    result.append({
                        "session_id": game_session.session_id,
                        "display_name": display_name,
                        "session_name": game_session.session_name,
                        "created_at": game_session.created_at,
                        "last_played_at": game_session.last_played_at,
                        "is_completed": game_session.is_completed,
                        "has_player": bool(game_session.player_data),
                        "has_characters": bool(game_session.characters_data)
                    })
                
                return result
                
        except Exception as e:
            print(f"❌ Erreur listage sessions DB: {e}")
            return []
    
    @staticmethod
    def log_event(session_id: str, event_type: str, event_data: dict):
        """Enregistre un événement de session."""
        try:
            with get_session() as session:
                event = SessionEvent(
                    session_id=session_id,
                    event_type=event_type,
                    event_data=event_data
                )
                
                session.add(event)
                session.commit()
                
        except Exception as e:
            print(f"⚠️ Erreur enregistrement événement: {e}")
    
    @staticmethod
    def delete_session(session_id: str) -> bool:
        """Supprime une session (soft delete)."""
        try:
            with get_session() as session:
                stmt = select(GameSession).where(GameSession.session_id == session_id)
                game_session = session.exec(stmt).first()
                
                if game_session:
                    game_session.is_active = False
                    game_session.updated_at = datetime.utcnow()
                    
                    session.add(game_session)
                    session.commit()
                    
                    print(f"🗑️ Session supprimée: {session_id}")
                    return True
                
                return False
                
        except Exception as e:
            print(f"❌ Erreur suppression session: {e}")
            return False


# Instance globale
db_session_service = DatabaseSessionService()