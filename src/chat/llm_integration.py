"""
LLM Integration for Conversational AI
AI QA Agent - Enhanced Sprint 1.4
"""
import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp

from src.core.logging import get_logger
from src.core.config import get_settings

logger = get_logger(__name__)

class LLMProvider:
    """Base class for LLM providers"""
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion"""
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    """OpenAI GPT integration"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1"
        self.default_model = "gpt-3.5-turbo"
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate OpenAI chat completion"""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error_text}")

class AnthropicProvider(LLMProvider):
    """Anthropic Claude integration"""
    
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.base_url = "https://api.anthropic.com/v1"
        self.default_model = "claude-3-sonnet-20240229"
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate Anthropic chat completion"""
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Convert messages to Anthropic format
        system_message = ""
        formatted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                formatted_messages.append(msg)
        
        payload = {
            "model": model or self.default_model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        if system_message:
            payload["system"] = system_message
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # Convert to OpenAI-like format for consistency
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": result["content"][0]["text"]
                            }
                        }],
                        "usage": result.get("usage", {})
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error: {response.status} - {error_text}")

class MockProvider(LLMProvider):
    """Mock provider for testing"""
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate mock response"""
        last_message = messages[-1]["content"] if messages else ""
        
        # Simple response generation based on content
        if "analysis" in last_message.lower():
            content = "I can help you analyze your code! I have access to AST parsing, complexity analysis, and pattern detection capabilities. What specific aspect would you like me to examine?"
        elif "test" in last_message.lower():
            content = "I can help you with testing strategies! Based on my analysis capabilities, I can identify components that need testing and suggest appropriate test approaches."
        elif "quality" in last_message.lower():
            content = "I can assess code quality using metrics like cyclomatic complexity, testability scores, and pattern detection. Would you like me to analyze a specific file or repository?"
        else:
            content = f"I understand you said: '{last_message}'. As an AI QA Agent, I can help you with code analysis, testing strategies, and quality assessment. How can I assist you today?"
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content
                }
            }],
            "usage": {
                "prompt_tokens": len(last_message.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(last_message.split()) + len(content.split())
            }
        }

class LLMIntegration:
    """Main LLM integration class"""
    
    def __init__(self):
        self.providers = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "mock": MockProvider()
        }
        self.default_provider = self._get_default_provider()
    
    def _get_default_provider(self) -> str:
        """Determine default provider based on available API keys"""
        if os.getenv('OPENAI_API_KEY'):
            return "openai"
        elif os.getenv('ANTHROPIC_API_KEY'):
            return "anthropic"
        else:
            logger.warning("No LLM API keys configured, using mock provider")
            return "mock"
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Generate conversational response"""
        try:
            provider = provider or self.default_provider
            
            if provider not in self.providers:
                raise ValueError(f"Unknown provider: {provider}")
            
            # Add context to messages if provided
            if context:
                enhanced_messages = await self._enhance_messages_with_context(messages, context)
            else:
                enhanced_messages = messages
            
            # Generate response
            result = await self.providers[provider].chat_completion(
                enhanced_messages,
                model=model,
                **kwargs
            )
            
            response_content = result["choices"][0]["message"]["content"]
            
            # Log usage if available
            if "usage" in result:
                usage = result["usage"]
                logger.info(f"LLM usage - Provider: {provider}, Tokens: {usage.get('total_tokens', 'unknown')}")
            
            return response_content
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback to mock provider
            if provider != "mock":
                logger.info("Falling back to mock provider")
                return await self.generate_response(messages, provider="mock", context=context)
            else:
                return "I apologize, but I'm experiencing technical difficulties. Please try again later."
    
    async def _enhance_messages_with_context(
        self,
        messages: List[Dict[str, str]],
        context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Enhance messages with analysis context"""
        enhanced_messages = messages.copy()
        
        # Add system message with context if we have analysis results
        analysis_results = context.get("analysis_results", [])
        if analysis_results:
            system_context = self._build_analysis_context(analysis_results)
            
            # Insert or update system message
            system_msg = {
                "role": "system",
                "content": f"""You are an AI QA Agent that helps with code analysis and testing. You have access to analysis results for the user's code.

Current Analysis Context:
{system_context}

Use this context to provide informed responses about code quality, testing strategies, and improvement recommendations. Be specific and reference the analysis results when relevant."""
            }
            
            # Insert at beginning or replace existing system message
            if enhanced_messages and enhanced_messages[0]["role"] == "system":
                enhanced_messages[0] = system_msg
            else:
                enhanced_messages.insert(0, system_msg)
        
        return enhanced_messages
    
    def _build_analysis_context(self, analysis_results: List[Dict[str, Any]]) -> str:
        """Build context string from analysis results"""
        if not analysis_results:
            return "No analysis results available."
        
        context_parts = []
        
        for result in analysis_results[-3:]:  # Last 3 analyses
            if "components" in result:
                component_count = len(result["components"])
                avg_complexity = sum(
                    c.get("complexity", {}).get("cyclomatic", 0) 
                    for c in result["components"]
                ) / component_count if component_count > 0 else 0
                
                context_parts.append(
                    f"- Analysis: {component_count} components, "
                    f"avg complexity: {avg_complexity:.1f}, "
                    f"type: {result.get('analysis_type', 'unknown')}"
                )
        
        return "\n".join(context_parts) if context_parts else "Basic analysis completed."
    
    async def analyze_user_intent(self, message: str) -> Dict[str, Any]:
        """Analyze user intent for routing and context"""
        # Simple intent detection based on keywords
        intents = {
            "analysis_request": ["analyze", "analysis", "check", "examine", "review"],
            "test_generation": ["test", "testing", "generate tests", "create tests"],
            "quality_assessment": ["quality", "metrics", "complexity", "maintainability"],
            "help_request": ["help", "how to", "explain", "what is"],
            "general_conversation": []  # default
        }
        
        message_lower = message.lower()
        detected_intent = "general_conversation"
        confidence = 0.0
        
        for intent, keywords in intents.items():
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            if matches > 0:
                intent_confidence = matches / len(keywords) if keywords else 0
                if intent_confidence > confidence:
                    detected_intent = intent
                    confidence = intent_confidence
        
        return {
            "intent": detected_intent,
            "confidence": confidence,
            "entities": self._extract_entities(message)
        }
    
    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """Extract entities from user message"""
        # Simple entity extraction
        entities = {
            "file_paths": [],
            "programming_languages": [],
            "test_frameworks": []
        }
        
        # Look for file paths
        import re
        file_patterns = [
            r'\b\w+\.(py|js|ts|java|cpp|c|h)\b',
            r'\b[\w/]+/[\w/]+\.(py|js|ts|java|cpp|c|h)\b'
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, message)
            entities["file_paths"].extend(matches)
        
        # Look for programming languages
        languages = ["python", "javascript", "typescript", "java", "c++", "c"]
        for lang in languages:
            if lang in message.lower():
                entities["programming_languages"].append(lang)
        
        # Look for test frameworks
        frameworks = ["pytest", "unittest", "jest", "mocha", "junit"]
        for framework in frameworks:
            if framework in message.lower():
                entities["test_frameworks"].append(framework)
        
        return entities
