"""
Conversation Manager - Foundation for agent conversations
AI QA Agent - Enhanced Sprint 1.4
"""
import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import redis

from src.core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class Message:
    """Individual conversation message"""
    id: str
    session_id: str
    role: str  # user, assistant, system
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            role=data["role"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

@dataclass
class ConversationSession:
    """Conversation session with context and metadata"""
    session_id: str
    user_id: Optional[str]
    title: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    message_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "message_count": self.message_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            title=data["title"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            message_count=data.get("message_count", 0)
        )

class ConversationManager:
    """Manage conversation sessions and context"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_sessions: Dict[str, ConversationSession] = {}
        self.memory_messages: Dict[str, List[Message]] = {}
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize Redis connection with fallback to memory"""
        try:
            import os
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info("Connected to Redis for conversation storage")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
            self.redis_client = None
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationSession:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            title=title or f"Conversation {session_id[:8]}",
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )
        
        await self._store_session(session)
        logger.info(f"Created conversation session: {session_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get conversation session by ID"""
        try:
            if self.redis_client:
                data = self.redis_client.get(f"session:{session_id}")
                if data:
                    return ConversationSession.from_dict(json.loads(data))
            else:
                return self.memory_sessions.get(session_id)
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
        return None
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add message to conversation"""
        message = Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.utcnow()
        )
        
        # Store message
        await self._store_message(message)
        
        # Update session
        session = await self.get_session(session_id)
        if session:
            session.message_count += 1
            session.updated_at = datetime.utcnow()
            await self._store_session(session)
        
        logger.debug(f"Added message to session {session_id}: {role}")
        return message
    
    async def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a session"""
        try:
            if self.redis_client:
                # Get message IDs from sorted set
                message_ids = self.redis_client.zrange(
                    f"session_messages:{session_id}",
                    offset,
                    offset + (limit or -1)
                )
                
                messages = []
                for msg_id in message_ids:
                    data = self.redis_client.get(f"message:{msg_id}")
                    if data:
                        messages.append(Message.from_dict(json.loads(data)))
                return messages
            else:
                # Use memory storage
                messages = self.memory_messages.get(session_id, [])
                if limit:
                    return messages[offset:offset + limit]
                return messages[offset:]
        except Exception as e:
            logger.error(f"Failed to get messages for session {session_id}: {e}")
        return []
    
    async def get_conversation_context(
        self,
        session_id: str,
        max_messages: int = 20
    ) -> Dict[str, Any]:
        """Get conversation context for AI processing"""
        session = await self.get_session(session_id)
        if not session:
            return {}
        
        messages = await self.get_messages(session_id, limit=max_messages)
        
        # Build context with analysis results if any
        context = {
            "session": session.to_dict(),
            "messages": [msg.to_dict() for msg in messages],
            "message_count": len(messages),
            "analysis_results": await self._get_session_analysis_results(session_id)
        }
        
        return context
    
    async def update_session_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update session metadata"""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.metadata.update(metadata)
        session.updated_at = datetime.utcnow()
        await self._store_session(session)
        return True
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete conversation session and all messages"""
        try:
            if self.redis_client:
                # Get all message IDs for this session
                message_ids = self.redis_client.zrange(f"session_messages:{session_id}", 0, -1)
                
                # Delete all messages
                for msg_id in message_ids:
                    self.redis_client.delete(f"message:{msg_id}")
                
                # Delete session and message list
                self.redis_client.delete(f"session:{session_id}")
                self.redis_client.delete(f"session_messages:{session_id}")
            else:
                # Delete from memory
                self.memory_sessions.pop(session_id, None)
                self.memory_messages.pop(session_id, None)
            
            logger.info(f"Deleted conversation session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def get_recent_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[ConversationSession]:
        """Get recent conversation sessions"""
        try:
            sessions = []
            
            if self.redis_client:
                # This would need a more sophisticated indexing system
                # For now, implement basic scanning
                keys = self.redis_client.keys("session:*")
                for key in keys:
                    data = self.redis_client.get(key)
                    if data:
                        session = ConversationSession.from_dict(json.loads(data))
                        if not user_id or session.user_id == user_id:
                            sessions.append(session)
            else:
                # Use memory storage
                for session in self.memory_sessions.values():
                    if not user_id or session.user_id == user_id:
                        sessions.append(session)
            
            # Sort by updated_at descending
            sessions.sort(key=lambda s: s.updated_at, reverse=True)
            return sessions[:limit]
        
        except Exception as e:
            logger.error(f"Failed to get recent sessions: {e}")
            return []
    
    # Private helper methods
    
    async def _store_session(self, session: ConversationSession):
        """Store session in Redis or memory"""
        try:
            if self.redis_client:
                self.redis_client.setex(
                    f"session:{session.session_id}",
                    86400 * 7,  # 7 days
                    json.dumps(session.to_dict())
                )
            else:
                self.memory_sessions[session.session_id] = session
        except Exception as e:
            logger.error(f"Failed to store session: {e}")
    
    async def _store_message(self, message: Message):
        """Store message in Redis or memory"""
        try:
            if self.redis_client:
                # Store message
                self.redis_client.setex(
                    f"message:{message.id}",
                    86400 * 7,  # 7 days
                    json.dumps(message.to_dict())
                )
                
                # Add to session message list (sorted by timestamp)
                self.redis_client.zadd(
                    f"session_messages:{message.session_id}",
                    {message.id: message.timestamp.timestamp()}
                )
            else:
                # Store in memory
                if message.session_id not in self.memory_messages:
                    self.memory_messages[message.session_id] = []
                self.memory_messages[message.session_id].append(message)
                
                # Keep messages sorted by timestamp
                self.memory_messages[message.session_id].sort(key=lambda m: m.timestamp)
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
    
    async def _get_session_analysis_results(self, session_id: str) -> List[Dict[str, Any]]:
        """Get analysis results associated with this session"""
        # This would integrate with the analysis task system
        # For now, return empty list
        return []
