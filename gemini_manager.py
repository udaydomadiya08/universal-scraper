import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    BANNED = "banned"
    COOLDOWN = "cooldown"

@dataclass
class RateLimit:
    rpm: int
    tpm: int
    rpd: int

@dataclass
class ModelConfig:
    name: str
    category: str
    rate_limit: RateLimit
    performance_score: float
    cost_efficiency: float

@dataclass
class APIKeyStatus:
    key: str
    last_used: datetime
    daily_requests: int
    daily_tokens: int
    minute_requests: int
    minute_tokens: int
    last_minute_reset: datetime
    last_day_reset: datetime
    banned_until: Optional[datetime] = None
    status: ModelStatus = ModelStatus.AVAILABLE

class GeminiAPIManager:
    def __init__(self, config_path: str = "gemini_config_10keys.json"):
        self.config_path = config_path
        self.api_keys: Dict[str, APIKeyStatus] = {}
        self.models: Dict[str, ModelConfig] = {}
        self.current_key_index = 0
        self.key_health_scores: Dict[str, float] = {}
        self.last_health_check = datetime.now()
        self.model_cooldowns: Dict[str, datetime] = {}
        self.key_unsupported_models: Dict[str, set] = {}
        self.load_config()
        
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                
            for key_data in config.get('api_keys', []):
                self._add_api_key(key_data['key'])
            
            self.models = {
                "text": {
                    "gemini-3.1-flash-lite": ModelConfig(
                        name="gemini-3.1-flash-lite",
                        category="text",
                        rate_limit=RateLimit(rpm=15, tpm=250000, rpd=500),
                        performance_score=10.0,
                        cost_efficiency=10.0
                    ),
                    "gemma-4-26b-a4b-it": ModelConfig(
                        name="gemma-4-26b-a4b-it",
                        category="text",
                        rate_limit=RateLimit(rpm=15, tpm=1000000, rpd=1500),
                        performance_score=9.8,
                        cost_efficiency=10.0
                    ),
                    "gemma-4-31b-it": ModelConfig(
                        name="gemma-4-31b-it",
                        category="text",
                        rate_limit=RateLimit(rpm=15, tpm=1000000, rpd=1500),
                        performance_score=9.7,
                        cost_efficiency=10.0
                    ),
                    "gemini-2.5-flash": ModelConfig(
                        name="gemini-2.5-flash",
                        category="text",
                        rate_limit=RateLimit(rpm=5, tpm=250000, rpd=20),
                        performance_score=9.6,
                        cost_efficiency=9.0
                    ),
                    "gemini-3.5-flash": ModelConfig(
                        name="gemini-3.5-flash",
                        category="text",
                        rate_limit=RateLimit(rpm=5, tpm=250000, rpd=20),
                        performance_score=9.5,
                        cost_efficiency=9.0
                    ),
                    "gemini-3-flash": ModelConfig(
                        name="gemini-3-flash",
                        category="text",
                        rate_limit=RateLimit(rpm=5, tpm=250000, rpd=20),
                        performance_score=9.4,
                        cost_efficiency=9.0
                    ),
                    "gemini-2.5-flash-lite": ModelConfig(
                        name="gemini-2.5-flash-lite",
                        category="text",
                        rate_limit=RateLimit(rpm=10, tpm=250000, rpd=20),
                        performance_score=9.3,
                        cost_efficiency=9.0
                    )
                }
            }
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def _add_api_key(self, api_key: str):
        now = datetime.now()
        self.api_keys[api_key] = APIKeyStatus(
            key=api_key,
            last_used=now,
            daily_requests=0,
            daily_tokens=0,
            minute_requests=0,
            minute_tokens=0,
            last_minute_reset=now,
            last_day_reset=now
        )
        self.key_health_scores[api_key] = 100.0
        self.key_unsupported_models[api_key] = set()

    def check_rate_limits(self, api_key: str, model_name: str, estimated_tokens: int) -> Tuple[bool, str]:
        status = self.api_keys[api_key]
        now = datetime.now()
        
        if status.status == ModelStatus.BANNED:
            if status.banned_until and now > status.banned_until:
                status.status = ModelStatus.AVAILABLE
                status.banned_until = None
                self.key_health_scores[api_key] = 50.0
            else:
                return False, "Key is banned"
                
        if status.status == ModelStatus.RATE_LIMITED:
            if (now - status.last_used).total_seconds() > 60:
                status.status = ModelStatus.AVAILABLE
            else:
                return False, "Key is rate limited"

        if (now - status.last_minute_reset).total_seconds() >= 60:
            status.minute_requests = 0
            status.minute_tokens = 0
            status.last_minute_reset = now
            
        if (now - status.last_day_reset).total_seconds() >= 86400:
            status.daily_requests = 0
            status.daily_tokens = 0
            status.last_day_reset = now

        model_config = None
        for cat in self.models.values():
            if model_name in cat:
                model_config = cat[model_name]
                break
                
        if not model_config:
            return False, "Unknown model"
            
        limits = model_config.rate_limit
        
        if status.daily_requests >= limits.rpd:
            return False, "Daily request limit reached"
            
        if status.minute_requests >= limits.rpm:
            return False, "Minute request limit reached"
            
        if status.minute_tokens + estimated_tokens > limits.tpm:
            return False, "Minute token limit reached"
            
        return True, "Available"

    def get_best_available_key_model(self, category: str = "text", estimated_tokens: int = 1000, target_model: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        available_models = [target_model] if target_model else list(self.models.get(category, {}).keys())
        
        healthy_keys = sorted(
            [k for k in self.api_keys.keys()],
            key=lambda k: self.key_health_scores.get(k, 0),
            reverse=True
        )
        
        for model_name in available_models:
            if model_name in self.model_cooldowns and datetime.now() < self.model_cooldowns[model_name]:
                continue
                
            for api_key in healthy_keys:
                if api_key in self.key_unsupported_models and model_name in self.key_unsupported_models[api_key]:
                    continue
                    
                can_use, reason = self.check_rate_limits(api_key, model_name, estimated_tokens)
                if can_use:
                    return api_key, model_name
                    
        return None, None

    def update_usage(self, api_key: str, model_name: str, tokens_used: int):
        status = self.api_keys[api_key]
        now = datetime.now()
        status.last_used = now
        status.minute_requests += 1
        status.daily_requests += 1
        status.minute_tokens += tokens_used
        status.daily_tokens += tokens_used
        
        self.key_health_scores[api_key] = min(100.0, self.key_health_scores[api_key] + 1)

    def handle_rate_limit_error(self, api_key: str, model_name: str, error_msg: str):
        error_lower = error_msg.lower()
        status = self.api_keys[api_key]
        now = datetime.now()
        
        if "404" in error_msg or "not found" in error_lower:
            self.key_unsupported_models[api_key].add(model_name)
        elif "429" in error_msg or "quota" in error_lower:
            if "daily" in error_lower or "limit: 20" in error_lower or "limit: 1500" in error_lower:
                status.status = ModelStatus.BANNED
                status.banned_until = now + timedelta(days=1)
                self.key_health_scores[api_key] = max(0.0, self.key_health_scores[api_key] - 50)
            else:
                status.status = ModelStatus.RATE_LIMITED
                self.key_health_scores[api_key] = max(0.0, self.key_health_scores[api_key] - 20)
                self.model_cooldowns[model_name] = now + timedelta(seconds=60)

    def make_api_call(self, prompt: str, category: str = "text", model_name: Optional[str] = None, **kwargs) -> Dict:
        estimated_tokens = len(prompt.split()) * 4
        max_retries = 15
        last_error = ""
        
        for attempt in range(max_retries):
            api_key, selected_model = self.get_best_available_key_model(category, estimated_tokens, target_model=model_name if attempt == 0 else None)
            
            if not api_key or not selected_model:
                return {"error": "No available Gemini API keys", "status": "rate_limited"}
                
            try:
                result = self._call_gemini_api(api_key, selected_model, prompt, **kwargs)
                
                content = ""
                if "candidates" in result and len(result["candidates"]) > 0:
                    parts = result["candidates"][0].get("content", {}).get("parts", [])
                    if parts and len(parts) > 0:
                        if isinstance(parts[0], dict):
                            content = parts[0].get("text", "")
                        elif isinstance(parts[0], str):
                            content = parts[0]
                
                tokens_used = result.get("usageMetadata", {}).get("totalTokenCount", estimated_tokens * 1.5)
                
                if content:
                    self.update_usage(api_key, selected_model, tokens_used)
                    return {
                        "content": content,
                        "model": selected_model,
                        "api_key": api_key[:10] + "...",
                        "status": "success"
                    }
                else:
                    raise Exception("Empty response or missing 'content' key")
            except Exception as e:
                error_msg = str(e)
                self.handle_rate_limit_error(api_key, selected_model, error_msg)
                last_error = error_msg
                
        return {"error": f"Failed after {max_retries} attempts. Last error: {last_error}", "status": "rate_limited"}

    def _call_gemini_api(self, api_key: str, model_name: str, prompt: str, image_data: Optional[str] = None, **kwargs):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        parts = [{"text": prompt}]
        
        if image_data:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_data
                }
            })
            
        data = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.1),
                "maxOutputTokens": kwargs.get("max_tokens", 2048)
            }
        }
        
        import requests
        resp = requests.post(url, json=data, headers=headers, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception(f"API Error {resp.status_code}: {resp.text}")

    def get_usage_stats(self) -> Dict:
        stats = {
            "total_keys": len(self.api_keys),
            "available_keys": 0,
            "rate_limited_keys": 0,
            "banned_keys": 0,
            "models": {}
        }
        for key_status in self.api_keys.values():
            if key_status.status == ModelStatus.AVAILABLE:
                stats["available_keys"] += 1
        return stats
