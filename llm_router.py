import logging
from typing import Dict, Optional, List
from enum import Enum

from gemini_manager import GeminiAPIManager
import json
import os
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    GEMINI = "gemini"
    NVIDIA = "nvidia"
    GROQ = "groq"
    OLLAMA = "ollama"

class LLMRouter:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        
        gemini_config_file = config_path
        if "api_keys" not in self.config:
            if os.path.exists("gemini_config_10keys.json"):
                gemini_config_file = "gemini_config_10keys.json"
            elif os.path.exists("gemini_config.json"):
                gemini_config_file = "gemini_config.json"
                
        self.gemini_manager = GeminiAPIManager(config_path=gemini_config_file)
        
        # Merge credentials from main config.json if it exists and keys are missing
        if os.path.exists("config.json"):
            try:
                with open("config.json", 'r') as f:
                    main_config = json.load(f)
                    for k, v in main_config.items():
                        if k not in self.config or not self.config[k]:
                            self.config[k] = v
            except Exception as e:
                logger.warning(f"Failed to merge config.json: {e}")
                
        self.provider_rankings = {
            LLMProvider.NVIDIA: 1.0,  # Highest priority
            LLMProvider.GEMINI: 0.9,  # High performance fallback
            LLMProvider.GROQ: 0.8,     # Good fallback
            LLMProvider.OLLAMA: 0.6   # Local fallback
        }
        # Comprehensive Fallback Chains (ALL Confirmed & New Models - May 2026)
        self.nvidia_fallback_chains = {
            "reasoning": [
                "meta/llama-3.1-70b-instruct",
                "meta/llama-3.1-70b-instruct",
                "mistralai/mixtral-8x22b-instruct-v0.1"
            ],
            "extraction": [
                self.config.get("nvidia_model"),
                "meta/llama-3.1-70b-instruct",
                "meta/llama-3.1-8b-instruct",
                "meta/llama3-70b-instruct"
            ],
            "text": [
                self.config.get("nvidia_model"),
                "meta/llama-3.1-70b-instruct",
                "meta/llama-3.1-8b-instruct",
                "meta/llama3-70b-instruct"
            ],
            "default": [
                "meta/llama-3.1-70b-instruct",
                "meta/llama-3.1-70b-instruct"
            ]
        }
        
    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def get_response(self, prompt: str, category: str = "text", preferred_provider: Optional[LLMProvider] = None, **kwargs) -> Dict:
        """
        Get LLM response with automatic provider selection and fallback
        """
        
        # Determine provider order based on preference and availability
        providers = self._get_provider_order(preferred_provider)
        
        last_error = None
        
        for provider in providers:
            try:
                print(f"[TRACE] Attempting {provider.value}...")
                if provider == LLMProvider.GEMINI:
                    result = self._call_gemini(prompt, category, **kwargs)
                    if result.get("status") == "success":
                        return {
                            "content": result["content"],
                            "provider": provider.value,
                            "model": result["model"],
                            "cost_estimate": self._estimate_cost(result["model"], len(prompt)),
                            "status": "success"
                        }
                    else:
                        last_error = result.get("error", "Gemini API error")
                        print(f"[DEBUG] Gemini Failed: {last_error}")
                        logger.warning(f"Gemini failed: {last_error}")
                        continue
                        
                elif provider == LLMProvider.NVIDIA:
                    result = self._call_nvidia(prompt, category=category, **kwargs)
                    if result.get("status") == "success":
                        return {
                            "content": result["content"],
                            "provider": "nvidia",
                            "model": result["model"],
                            "cost_estimate": self._estimate_cost("nvidia", len(prompt)),
                            "status": "success"
                        }
                    else:
                        last_error = result.get("error", "NVIDIA NIM error")
                        print(f"[DEBUG] NVIDIA Failed: {last_error}")
                        logger.warning(f"NVIDIA failed: {last_error}")
                        continue
                        
                elif provider == LLMProvider.GROQ:
                    result = self._call_groq(prompt, **kwargs)
                    if result.get("status") == "success":
                        return {
                            "content": result["content"],
                            "provider": "groq", 
                            "model": result["model"],
                            "cost_estimate": self._estimate_cost("groq", len(prompt)),
                            "status": "success"
                        }
                    else:
                        last_error = result.get("error", "Groq API error")
                        logger.warning(f"Groq failed: {last_error}")
                        continue
                        
                elif provider == LLMProvider.OLLAMA:
                    result = self._call_ollama(prompt, **kwargs)
                    if result.get("status") == "success":
                        return {
                            "content": result["content"],
                            "provider": "ollama",
                            "model": result["model"], 
                            "cost_estimate": 0.0,  # Local models are free
                            "status": "success"
                        }
                    else:
                        last_error = result.get("error", "Ollama error")
                        logger.warning(f"Ollama failed: {last_error}")
                        continue
                        
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Provider {provider.value} failed: {last_error}")
                continue
        
        # All providers failed
        return {
            "error": f"All providers failed. Last error: {last_error}",
            "status": "failed",
            "attempted_providers": [p.value for p in providers]
        }
    
    def _get_provider_order(self, preferred: Optional[LLMProvider]) -> List[LLMProvider]:
        """Get ordered list of providers to try"""
        
        # Override everything if use_ollama is strictly True
        if self.config.get("use_ollama") is True:
            return [LLMProvider.OLLAMA]
            
        providers = list(LLMProvider)
        
        # Sort by ranking (highest first)
        providers.sort(key=lambda x: self.provider_rankings[x], reverse=True)
        
        # Resolve preferred provider dynamically from config if not passed
        if not preferred:
            pref_str = self.config.get("preferred_provider")
            if pref_str:
                try:
                    preferred = LLMProvider(pref_str)
                except ValueError:
                    pass
        
        # If preferred provider specified, move it to front
        if preferred:
            if preferred in providers:
                providers.remove(preferred)
                providers.insert(0, preferred)
        
        return providers
    
    def _call_gemini(self, prompt: str, category: str, **kwargs) -> Dict:
        """Call Gemini API through manager and standardize response"""
        try:
            # Map extraction to text category for Gemini Manager
            if category == "extraction":
                category = "text"
                
            # Check if Gemini is configured
            gemini_stats = self.gemini_manager.get_usage_stats()
            print(f"[TRACE] Gemini Stats: {gemini_stats['available_keys']} keys available")
            if gemini_stats["available_keys"] == 0:
                return {"error": "No available Gemini API keys", "status": "no_keys"}
            
            result = self.gemini_manager.make_api_call(prompt, category, **kwargs)
            print(f"[TRACE] Gemini Result Model: {result.get('model')}")
            if result.get("status") == "success":
                # Gemini manager already returns the flat text string in "content"!
                return {
                    "content": result["content"],
                    "model": result["model"],
                    "status": "success"
                }
            return result
            
        except Exception as e:
            return {"error": f"Gemini API error: {str(e)}", "status": "error"}
    
    def _call_nvidia(self, prompt: str, category: str = "default", **kwargs) -> Dict:
        """Call NVIDIA NIM API with intelligent model rotation and fallback"""
        try:
            api_key = self.config.get("nvidia_api_key") or os.getenv("NVIDIA_API_KEY")
            if not api_key:
                return {"error": "NVIDIA API key not configured", "status": "no_key"}
            
            base_url = self.config.get("nvidia_base_url", "https://integrate.api.nvidia.com/v1")
            
            # Get the fallback chain for the requested category
            models_to_try = self.nvidia_fallback_chains.get(category, self.nvidia_fallback_chains["default"])
            
            # Prioritize the specific model from config if it exists
            preferred_model = self.config.get("nvidia_model")
            if preferred_model and preferred_model not in models_to_try:
                models_to_try = [preferred_model] + models_to_try
            elif preferred_model and preferred_model in models_to_try:
                models_to_try.remove(preferred_model)
                models_to_try = [preferred_model] + models_to_try

            import requests
            last_error = None
            
            for model in models_to_try:
                try:
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": kwargs.get("temperature", 0.1),
                        "top_p": kwargs.get("top_p", 1),
                        "max_tokens": kwargs.get("max_tokens", 1024),
                        "stream": False
                    }
                    
                    resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        return {
                            "content": data["choices"][0]["message"]["content"],
                            "model": model,
                            "status": "success"
                        }
                    elif resp.status_code == 429:
                        logger.warning(f"NVIDIA Model {model} rate limited (429). Falling back...")
                        continue
                    else:
                        error_text = resp.text
                        logger.warning(f"NVIDIA Model {model} failed ({resp.status_code}): {error_text}")
                        last_error = f"Model {model} error: {error_text}"
                        continue
                                
                except Exception as e:
                    logger.error(f"Error calling NVIDIA model {model}: {e}")
                    last_error = str(e)
                    continue
            
            return {"error": f"All NVIDIA models in chain failed. Last error: {last_error}", "status": "error"}
                        
        except Exception as e:
            return {"error": f"NVIDIA API error: {str(e)}", "status": "error"}

    def _call_groq(self, prompt: str, **kwargs) -> Dict:
        try:
            api_key = self.config.get("groq_api_key") or os.getenv("GROQ_API_KEY")
            if not api_key:
                return {"error": "Groq API key not configured. Set GROQ_API_KEY in .env", "status": "no_key"}

            import requests
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": kwargs.get("model", "llama-3.3-70b-versatile"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.2),
                "max_tokens": kwargs.get("max_tokens", 2048)
            }
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "content": data["choices"][0]["message"]["content"],
                    "model": payload["model"],
                    "status": "success"
                }
            else:
                error_text = resp.text
                return {"error": f"Groq API error ({resp.status_code}): {error_text}", "status": "error"}
        except Exception as e:
            return {"error": f"Groq API error: {str(e)}", "status": "error"}
    
    def _call_ollama(self, prompt: str, **kwargs) -> Dict:
        try:
            model = self.config.get("ollama_model") or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
            host = "http://localhost:11434"
            import requests
            import subprocess
            import time
            
            # Auto-start Ollama if it is closed
            try:
                requests.get(host, timeout=1)
            except requests.ConnectionError:
                print(f"[*] Ollama is closed. Auto-starting Ollama daemon for {model}...")
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(4)  # Wait for daemon to boot up
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "num_ctx": 4096
                }
            }
            resp = requests.post(f"{host}/api/generate", json=payload, timeout=600)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "content": data["response"],
                    "model": model,
                    "status": "success"
                }
            else:
                return {"error": f"Ollama error: {resp.text}", "status": "error"}
        except Exception as e:
            return {"error": f"Ollama error: {str(e)}", "status": "error"}
    
    def _estimate_cost(self, model: str, prompt_length: int) -> float:
        """Estimate API call cost in USD"""
        # Rough cost estimates (these should be updated with actual pricing)
        costs = {
            "gemini-2.5-flash": 0.000075,  # per 1K tokens
            "gemini-3-flash": 0.000075,
            "gemini-2.5-flash-lite": 0.00005,
            "gemini-3.1-flash-lite": 0.00005,
            "gemini-2.5-flash-tts": 0.0001,
            "gemini-3.1-flash-tts": 0.0001,
            "gemini-embedding-1": 0.000025,
            "gemini-embedding-2": 0.000025,
            "gemma-3-1b": 0.00005,
            "gemma-3-4b": 0.000075,
            "gemma-4-26b": 0.00015,
            "nvidia": 0.0001,  # Rough estimate for NIM
            "groq": 0.00005  # Rough estimate
        }
        
        cost_per_1k = costs.get(model, 0.00005)
        estimated_tokens = prompt_length * 1.3  # Rough token estimate
        return (estimated_tokens / 1000) * cost_per_1k
    
    def get_provider_stats(self) -> Dict:
        """Get statistics for all providers"""
        gemini_stats = self.gemini_manager.get_usage_stats()
        
        return {
            "gemini": {
                "available": gemini_stats["available_keys"] > 0,
                "keys": gemini_stats["total_keys"],
                "rate_limited": gemini_stats["rate_limited_keys"],
                "banned": gemini_stats["banned_keys"],
                "models": gemini_stats["models"]
            },
            "nvidia": {
                "available": bool(self.config.get("nvidia_api_key") or os.getenv("NVIDIA_API_KEY")),
                "model": self.config.get("nvidia_model", "meta/llama-3.1-70b-instruct")
            },
            "groq": {
                "available": bool(self.config.get("groq_api_key") or os.getenv("GROQ_API_KEY")),
                "model": "llama-3.3-70b-versatile"
            },
            "ollama": {
                "available": True,
                "model": self.config.get("ollama_model") or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
            }
        }
    
    def test_all_providers(self) -> Dict:
        """Test all providers with a simple prompt"""
        test_prompt = "Hello, please respond with 'Test successful'"
        results = {}
        
        for provider in LLMProvider:
            try:
                result = self.get_response(test_prompt, preferred_provider=provider)
                results[provider.value] = {
                    "status": result.get("status", "unknown"),
                    "response_time": result.get("response_time", 0),
                    "error": result.get("error")
                }
            except Exception as e:
                results[provider.value] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
