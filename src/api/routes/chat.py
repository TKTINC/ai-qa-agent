"""
Chat API Routes - Conversational interface
AI QA Agent - Enhanced Sprint 1.4
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime

from src.core.logging import get_logger
from src.chat.conversation_manager import ConversationManager, Message, ConversationSession
from src.chat.llm_integration import LLMIntegration

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Initialize managers
conversation_manager = ConversationManager()
llm_integration = LLMIntegration()

# Request/Response Models
class ChatMessage(BaseModel):
    """Chat message request model"""
    session_id: Optional[str] = None
    message: str = Field(..., description="User message content")
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Chat response model"""
    session_id: str
    message_id: str
    response: str
    metadata: Dict[str, Any]
    timestamp: datetime

class SessionInfo(BaseModel):
    """Session information model"""
    session_id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

class ConversationHistory(BaseModel):
    """Conversation history model"""
    session: SessionInfo
    messages: List[Dict[str, Any]]
    total_messages: int

# WebSocket connection manager
class ChatConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for chat session: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for chat session: {session_id}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                self.disconnect(session_id)

chat_manager = ChatConnectionManager()

# Chat Endpoints

@router.post("/message", response_model=ChatResponse)
async def send_message(chat_message: ChatMessage) -> ChatResponse:
    """Send a message and get AI response (HTTP endpoint)"""
    try:
        # Create or get session
        if chat_message.session_id:
            session = await conversation_manager.get_session(chat_message.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            session = await conversation_manager.create_session(
                user_id=chat_message.user_id,
                title=f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            )
        
        # Add user message
        user_message = await conversation_manager.add_message(
            session.session_id,
            "user",
            chat_message.message,
            chat_message.metadata
        )
        
        # Get conversation context
        context = await conversation_manager.get_conversation_context(session.session_id)
        
        # Analyze user intent
        intent_analysis = await llm_integration.analyze_user_intent(chat_message.message)
        
        # Build messages for LLM
        recent_messages = await conversation_manager.get_messages(session.session_id, limit=10)
        llm_messages = []
        
        for msg in recent_messages:
            llm_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Generate AI response
        ai_response = await llm_integration.generate_response(
            llm_messages,
            context=context
        )
        
        # Add AI response to conversation
        ai_message = await conversation_manager.add_message(
            session.session_id,
            "assistant",
            ai_response,
            {
                "intent_analysis": intent_analysis,
                "provider": llm_integration.default_provider
            }
        )
        
        response = ChatResponse(
            session_id=session.session_id,
            message_id=ai_message.id,
            response=ai_response,
            metadata={
                "intent": intent_analysis,
                "message_count": session.message_count + 2  # +2 for user and AI messages
            },
            timestamp=ai_message.timestamp
        )
        
        logger.info(f"Chat response generated for session: {session.session_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat message processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/session/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await chat_manager.connect(websocket, session_id)
    
    try:
        # Get or create session
        session = await conversation_manager.get_session(session_id)
        if not session:
            session = await conversation_manager.create_session()
            session_id = session.session_id
        
        # Send welcome message
        welcome_msg = {
            "type": "system",
            "message": "Connected to AI QA Agent. I can help you with code analysis and testing!",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket.send_json(welcome_msg)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message_content = message_data.get("message", "")
            if not user_message_content:
                continue
            
            # Add user message
            user_message = await conversation_manager.add_message(
                session_id,
                "user", 
                user_message_content,
                message_data.get("metadata")
            )
            
            # Send typing indicator
            typing_msg = {
                "type": "typing",
                "message": "AI is thinking...",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            await websocket.send_json(typing_msg)
            
            # Get context and generate response
            context = await conversation_manager.get_conversation_context(session_id)
            intent_analysis = await llm_integration.analyze_user_intent(user_message_content)
            
            # Build LLM messages
            recent_messages = await conversation_manager.get_messages(session_id, limit=10)
            llm_messages = [{"role": msg.role, "content": msg.content} for msg in recent_messages]
            
            # Generate response
            ai_response = await llm_integration.generate_response(
                llm_messages,
                context=context
            )
            
            # Add AI message
            ai_message = await conversation_manager.add_message(
                session_id,
                "assistant",
                ai_response,
                {"intent_analysis": intent_analysis}
            )
            
            # Send AI response
            response_msg = {
                "type": "message",
                "message": ai_response,
                "message_id": ai_message.id,
                "session_id": session_id,
                "metadata": {"intent": intent_analysis},
                "timestamp": ai_message.timestamp.isoformat()
            }
            await websocket.send_json(response_msg)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        error_msg = {
            "type": "error",
            "message": "An error occurred during the conversation",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        try:
            await websocket.send_json(error_msg)
        except:
            pass
    finally:
        chat_manager.disconnect(session_id)

@router.get("/sessions", response_model=List[SessionInfo])
async def get_sessions(
    user_id: Optional[str] = None,
    limit: int = 20
) -> List[SessionInfo]:
    """Get recent conversation sessions"""
    try:
        sessions = await conversation_manager.get_recent_sessions(user_id=user_id, limit=limit)
        
        return [
            SessionInfo(
                session_id=session.session_id,
                title=session.title,
                message_count=session.message_count,
                created_at=session.created_at,
                updated_at=session.updated_at,
                metadata=session.metadata
            )
            for session in sessions
        ]
    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}", response_model=ConversationHistory)
async def get_conversation_history(
    session_id: str,
    limit: Optional[int] = 50,
    offset: int = 0
) -> ConversationHistory:
    """Get conversation history for a session"""
    try:
        session = await conversation_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = await conversation_manager.get_messages(
            session_id, 
            limit=limit, 
            offset=offset
        )
        
        return ConversationHistory(
            session=SessionInfo(
                session_id=session.session_id,
                title=session.title,
                message_count=session.message_count,
                created_at=session.created_at,
                updated_at=session.updated_at,
                metadata=session.metadata
            ),
            messages=[msg.to_dict() for msg in messages],
            total_messages=session.message_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions", response_model=SessionInfo)
async def create_session(
    user_id: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> SessionInfo:
    """Create a new conversation session"""
    try:
        session = await conversation_manager.create_session(
            user_id=user_id,
            title=title,
            metadata=metadata
        )
        
        return SessionInfo(
            session_id=session.session_id,
            title=session.title,
            message_count=session.message_count,
            created_at=session.created_at,
            updated_at=session.updated_at,
            metadata=session.metadata
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Delete a conversation session"""
    try:
        success = await conversation_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": f"Session {session_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/sessions/{session_id}/metadata")
async def update_session_metadata(
    session_id: str,
    metadata: Dict[str, Any]
) -> Dict[str, str]:
    """Update session metadata"""
    try:
        success = await conversation_manager.update_session_metadata(session_id, metadata)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": f"Session {session_id} metadata updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))
