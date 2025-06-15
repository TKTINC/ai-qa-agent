"""
Real-time learning data streaming manager.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import WebSocket
from dataclasses import dataclass, asdict

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StreamClient:
    """Represents a connected streaming client"""
    client_id: str
    websocket: WebSocket
    subscriptions: List[str]
    connected_at: datetime
    last_activity: datetime


class LearningStreamManager:
    """Manage real-time streaming of learning data"""
    
    def __init__(self):
        self.clients: Dict[str, StreamClient] = {}
        self.active_streams: Dict[str, bool] = {}
        
    async def register_client(self, client_id: str, websocket: WebSocket, subscriptions: List[str] = None) -> None:
        """Register a new streaming client"""
        
        client = StreamClient(
            client_id=client_id,
            websocket=websocket,
            subscriptions=subscriptions or ["learning_updates", "agent_performance"],
            connected_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.clients[client_id] = client
        logger.info(f"Registered streaming client {client_id} with subscriptions: {client.subscriptions}")
    
    async def unregister_client(self, client_id: str) -> None:
        """Unregister a streaming client"""
        
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Unregistered streaming client {client_id}")
    
    async def broadcast_learning_update(self, update_data: Dict[str, Any]) -> None:
        """Broadcast learning update to all subscribed clients"""
        
        message = {
            "type": "learning_update",
            "data": update_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_to_subscribers("learning_updates", message)
    
    async def broadcast_agent_performance(self, agent_name: str, performance_data: Dict[str, Any]) -> None:
        """Broadcast agent performance update"""
        
        message = {
            "type": "agent_performance_update",
            "agent": agent_name,
            "data": performance_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_to_subscribers("agent_performance", message)
    
    async def broadcast_user_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Broadcast user feedback event"""
        
        message = {
            "type": "user_feedback",
            "data": feedback_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_to_subscribers("user_feedback", message)
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific client"""
        
        if client_id not in self.clients:
            return False
        
        client = self.clients[client_id]
        
        try:
            await client.websocket.send_json(message)
            client.last_activity = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {str(e)}")
            # Remove disconnected client
            await self.unregister_client(client_id)
            return False
    
    async def _broadcast_to_subscribers(self, subscription: str, message: Dict[str, Any]) -> None:
        """Broadcast message to all clients subscribed to a specific topic"""
        
        subscribers = [
            client for client in self.clients.values()
            if subscription in client.subscriptions
        ]
        
        if not subscribers:
            return
        
        # Send to all subscribers concurrently
        tasks = []
        for client in subscribers:
            task = self.send_to_client(client.client_id, message)
            tasks.append(task)
        
        # Wait for all sends to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        successful_sends = sum(1 for result in results if result is True)
        logger.debug(f"Broadcasted {subscription} to {successful_sends}/{len(subscribers)} clients")
    
    def get_client_count(self) -> int:
        """Get number of connected clients"""
        return len(self.clients)
    
    def get_subscription_stats(self) -> Dict[str, int]:
        """Get subscription statistics"""
        
        stats = {}
        for client in self.clients.values():
            for subscription in client.subscriptions:
                stats[subscription] = stats.get(subscription, 0) + 1
        
        return stats
