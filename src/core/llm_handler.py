# src/core/llm_handler.py
# HackRx 6.0 - LLM Handler with Gemini/Groq Failover

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

import google.generativeai as genai
from groq import AsyncGroq
import httpx

from ..utils.config import get_settings, LLMConfig
from ..utils.logger import get_logger, log_llm_usage
from ..utils.helpers import RateLimiter, TokenCounter, extract_json_from_text, create_prompt_template

logger = get_logger(__name__)


class LLMProvider(Enum):
    GEMINI = "gemini"
    GROQ = "groq"


@dataclass
class LLMResponse:
    """
    Response from LLM API call
    """
    content: str
    provider: LLMProvider
    model: str
    input_tokens: int
    output_tokens: int
    response_time: float
    success: bool
    error: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMHandler:
    """
    Handles LLM API calls with automatic failover between providers
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize API clients
        self.gemini_client = None
        self.groq_client = None
        
        # Rate limiters for each provider
        self.rate_limiters = {}
        
        # Provider status tracking
        self.provider_status = {
            LLMProvider.GEMINI: {"available": True, "last_error": None, "error_count": 0},
            LLMProvider.GROQ: {"available": True, "last_error": None, "error_count": 0}
        }
        
        # Token counters
        self.token_counter = TokenCounter()
    
    async def initialize(self):
        """
        Initialize LLM clients and rate limiters
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info("Initializing LLM handlers...")
            
            # Initialize Gemini
            if self.settings.gemini_api_key:
                genai.configure(api_key=self.settings.gemini_api_key)
                self.gemini_client = genai.GenerativeModel(
                    model_name=LLMConfig.MODELS["gemini"]["model_name"]
                )
                
                # Create rate limiter for Gemini
                gemini_limits = LLMConfig.GEMINI_LIMITS
                self.rate_limiters[LLMProvider.GEMINI] = RateLimiter(
                    max_calls=int(gemini_limits["requests_per_minute"] * self.settings.rate_limit_buffer),
                    time_window=60
                )
                
                logger.info("✅ Gemini client initialized")
            else:
                logger.warning("❌ Gemini API key not provided")
                self.provider_status[LLMProvider.GEMINI]["available"] = False
            
            # Initialize Groq
            if self.settings.groq_api_key:
                self.groq_client = AsyncGroq(api_key=self.settings.groq_api_key)
                
                # Create rate limiter for Groq
                groq_limits = LLMConfig.GROQ_LIMITS
                self.rate_limiters[LLMProvider.GROQ] = RateLimiter(
                    max_calls=int(groq_limits["requests_per_minute"] * self.settings.rate_limit_buffer),
                    time_window=60  
                )
                
                logger.info("✅ Groq client initialized")
            else:
                logger.warning("❌ Groq API key not provided")  
                self.provider_status[LLMProvider.GROQ]["available"] = False
            
            # Verify at least one provider is available
            available_providers = [p for p, status in self.provider_status.items() if status["available"]]
            if not available_providers:
                raise RuntimeError("No LLM providers available")
            
            logger.info(f"✅ LLM handler initialized with providers: {[p.value for p in available_providers]}")
            
        except Exception as e:
            logger.error(f"❌ LLM handler initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize LLM handler: {e}")
    
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None,
                              preferred_provider: Optional[LLMProvider] = None,
                              max_tokens: Optional[int] = None,
                              temperature: Optional[float] = None,
                              json_mode: bool = False) -> LLMResponse:
        """
        Generate response with automatic failover
        
        Args:
            prompt: User prompt
            system_prompt: System/instruction prompt
            preferred_provider: Preferred LLM provider
            max_tokens: Maximum output tokens
            temperature: Response randomness (0-1)
            json_mode: Whether to expect JSON response
        
        Returns:
            LLMResponse with generated content
        
        Raises:
            RuntimeError: If all providers fail
        """
        # Set defaults
        if max_tokens is None:
            max_tokens = self.settings.max_tokens
        if temperature is None:
            temperature = self.settings.temperature
        
        # Determine provider order
        if preferred_provider and self.provider_status[preferred_provider]["available"]:
            providers_to_try = [preferred_provider]
        else:
            providers_to_try = []
        
        # Add other available providers
        for provider in [LLMProvider.GEMINI, LLMProvider.GROQ]:
            if provider not in providers_to_try and self.provider_status[provider]["available"]:
                providers_to_try.append(provider)
        
        if not providers_to_try:
            raise RuntimeError("No LLM providers available")
        
        # Try each provider in order
        last_error = None
        for provider in providers_to_try:
            try:
                logger.debug(f"Attempting request with {provider.value}")
                
                response = await self._call_provider(
                    provider=provider,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_mode=json_mode
                )
                
                if response.success:
                    # Reset error count on success
                    self.provider_status[provider]["error_count"] = 0
                    return response
                else:
                    last_error = response.error
                    self._handle_provider_error(provider, response.error)
                    
            except Exception as e:
                last_error = str(e)
                self._handle_provider_error(provider, str(e))
                logger.warning(f"{provider.value} failed: {e}")
                continue
        
        # All providers failed
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
    
    async def _call_provider(self, provider: LLMProvider, prompt: str,
                           system_prompt: Optional[str] = None,
                           max_tokens: int = 1000,
                           temperature: float = 0.1,
                           json_mode: bool = False) -> LLMResponse:
        """
        Call specific LLM provider
        
        Args:
            provider: LLM provider to call
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum output tokens
            temperature: Response temperature
            json_mode: Whether to expect JSON response
        
        Returns:
            LLMResponse with result
        """
        start_time = time.time()
        
        # Apply rate limiting
        if provider in self.rate_limiters:
            await self.rate_limiters[provider].wait_if_needed()
        
        try:
            if provider == LLMProvider.GEMINI:
                return await self._call_gemini(prompt, system_prompt, max_tokens, temperature, json_mode, start_time)
            elif provider == LLMProvider.GROQ:
                return await self._call_groq(prompt, system_prompt, max_tokens, temperature, json_mode, start_time)
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"{provider.value} API call failed after {response_time:.2f}s: {e}")
            
            return LLMResponse(
                content="",
                provider=provider,
                model=LLMConfig.MODELS[provider.value]["model_name"],
                input_tokens=self.token_counter.count_tokens(prompt + (system_prompt or "")),
                output_tokens=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    async def _call_gemini(self, prompt: str, system_prompt: Optional[str],
                          max_tokens: int, temperature: float, json_mode: bool,
                          start_time: float) -> LLMResponse:
        """Call Gemini API"""
        try:
            # Combine system and user prompts
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}"
            
            if json_mode:
                full_prompt += "\n\nPlease respond with valid JSON only."
            
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Generate response
            response = await asyncio.to_thread(
                self.gemini_client.generate_content,
                full_prompt,
                generation_config=generation_config
            )
            
            response_time = time.time() - start_time
            
            # Extract content
            content = response.text if response.text else ""
            
            # Estimate tokens (Gemini doesn't provide exact counts in free tier)
            input_tokens = self.token_counter.count_tokens(full_prompt)
            output_tokens = self.token_counter.count_tokens(content)
            
            # Log usage
            log_llm_usage(
                provider="gemini",
                model=LLMConfig.MODELS["gemini"]["model_name"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_time=response_time,
                success=True
            )
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.GEMINI,
                model=LLMConfig.MODELS["gemini"]["model_name"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_time=response_time,
                success=True,
                metadata={"finish_reason": "completed"}
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Log failed usage
            log_llm_usage(
                provider="gemini",
                model=LLMConfig.MODELS["gemini"]["model_name"],
                input_tokens=self.token_counter.count_tokens(prompt + (system_prompt or "")),
                output_tokens=0,
                response_time=response_time,
                success=False
            )
            
            raise e
    
    async def _call_groq(self, prompt: str, system_prompt: Optional[str],
                        max_tokens: int, temperature: float, json_mode: bool,
                        start_time: float) -> LLMResponse:
        """Call Groq API"""
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            user_content = prompt
            if json_mode:
                user_content += "\n\nPlease respond with valid JSON only."
            
            messages.append({"role": "user", "content": user_content})
            
            # Make API call
            response = await self.groq_client.chat.completions.create(
                model=LLMConfig.MODELS["groq"]["model_name"],
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            response_time = time.time() - start_time
            
            # Extract content and token counts
            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            
            # Log usage
            log_llm_usage(
                provider="groq",
                model=LLMConfig.MODELS["groq"]["model_name"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_time=response_time,
                success=True
            )
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.GROQ,
                model=LLMConfig.MODELS["groq"]["model_name"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_time=response_time,
                success=True,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Log failed usage
            log_llm_usage(
                provider="groq",
                model=LLMConfig.MODELS["groq"]["model_name"],
                input_tokens=self.token_counter.count_tokens(prompt + (system_prompt or "")),
                output_tokens=0,
                response_time=response_time,
                success=False
            )
            
            raise e
    
    def _handle_provider_error(self, provider: LLMProvider, error: str):
        """
        Handle provider errors and update availability
        
        Args:
            provider: Provider that encountered error
            error: Error message
        """
        status = self.provider_status[provider]
        status["last_error"] = error
        status["error_count"] += 1
        
        # Temporarily disable provider after too many errors
        if status["error_count"] >= 3:
            status["available"] = False
            logger.warning(f"Provider {provider.value} temporarily disabled due to repeated errors")
            
            # Re-enable after some time (simple backoff)
            asyncio.create_task(self._re_enable_provider(provider, delay=300))  # 5 minutes
    
    async def _re_enable_provider(self, provider: LLMProvider, delay: int):
        """
        Re-enable provider after delay
        
        Args:
            provider: Provider to re-enable
            delay: Delay in seconds
        """
        await asyncio.sleep(delay)
        
        self.provider_status[provider]["available"] = True
        self.provider_status[provider]["error_count"] = 0
        logger.info(f"Provider {provider.value} re-enabled")
    
    def get_provider_status(self) -> Dict[str, Any]:
        """
        Get current status of all providers
        
        Returns:
            Dictionary with provider status information
        """
        return {
            provider.value: {
                "available": status["available"],
                "error_count": status["error_count"],
                "last_error": status["last_error"],
                "rate_limiter_calls": len(self.rate_limiters.get(provider, RateLimiter(1)).calls) if provider in self.rate_limiters else 0
            }
            for provider, status in self.provider_status.items()
        }
    
    async def cleanup(self):
        """
        Cleanup LLM handler resources
        """
        logger.info("Cleaning up LLM handler resources")
        
        # Close any open connections
        if self.groq_client:
            await self.groq_client.close()
        
        logger.info("✅ LLM handler resources cleaned up")


# Global LLM handler instance
_llm_handler = None


async def get_llm_handler() -> LLMHandler:
    """
    Get or create the global LLM handler instance
    
    Returns:
        Initialized LLMHandler instance
    """
    global _llm_handler
    
    if _llm_handler is None:
        _llm_handler = LLMHandler()
        await _llm_handler.initialize()
    
    return _llm_handler