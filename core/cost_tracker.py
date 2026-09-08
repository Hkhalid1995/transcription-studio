from dataclasses import dataclass
from typing import Dict, Any

# Pricing per 1M tokens (USD)
MODEL_PRICING = {
    # Gemini 3.x Series (Frontier)
    "gemini-3.5-transcribe":     {"input_audio": 0.40, "input_text": 0.10, "output": 1.50, "cached": 0.05},
    "gemini-3.7-flash":          {"input_audio": 0.75, "input_text": 0.75, "output": 3.75, "cached": 0.1875},
    "gemini-3.1-pro":            {"input_audio": 2.00, "input_text": 2.00, "output": 12.00, "cached": 0.50},
    "gemini-3.1-pro-preview":    {"input_audio": 2.00, "input_text": 2.00, "output": 12.00, "cached": 0.50},

    # Gemini 2.5 Series
    "gemini-2.5-flash":          {"input_audio": 0.30, "input_text": 0.075, "output": 0.30, "cached": 0.01875},
    "gemini-2.5-pro":            {"input_audio": 1.25, "input_text": 1.25,  "output": 5.00,  "cached": 0.3125},
    "gemini-2.0-flash":          {"input_audio": 0.30, "input_text": 0.075, "output": 0.30, "cached": 0.01875},

    # OpenAI Models (Fallback)
    "gpt-4o":                    {"input_audio": 5.00, "input_text": 2.50,  "output": 10.00, "cached": 1.25},
    "gpt-4o-mini":               {"input_audio": 0.30, "input_text": 0.15,  "output": 0.60,  "cached": 0.075},
}

def safe_int(value: Any) -> int:
    return value if isinstance(value, int) else 0

@dataclass
class UsageStats:
    pass1_model: str = "gemini-3.5-transcribe"
    pass2_model: str = "gemini-3.1-pro"
    
    pass1_input_tokens: int = 0
    pass1_output_tokens: int = 0
    pass1_cached_tokens: int = 0
    
    pass2_input_tokens: int = 0
    pass2_output_tokens: int = 0
    pass2_cached_tokens: int = 0

    def add_pass1(self, usage: Any, model: str):
        self.pass1_model = model
        if usage:
            self.pass1_input_tokens += safe_int(getattr(usage, "prompt_token_count", 0))
            self.pass1_output_tokens += safe_int(getattr(usage, "candidates_token_count", 0))
            self.pass1_cached_tokens += safe_int(getattr(usage, "cached_content_token_count", 0))

    def add_pass2(self, usage: Any, model: str):
        self.pass2_model = model
        if usage:
            self.pass2_input_tokens += safe_int(getattr(usage, "prompt_token_count", 0))
            self.pass2_output_tokens += safe_int(getattr(usage, "candidates_token_count", 0))
            self.pass2_cached_tokens += safe_int(getattr(usage, "cached_content_token_count", 0))

    def _get_model_rates(self, model_name: str) -> Dict[str, float]:
        name = model_name.lower().strip()
        if name in MODEL_PRICING:
            return MODEL_PRICING[name]
        if "transcribe" in name or "flash" in name:
            return {"input_audio": 0.50, "input_text": 0.20, "output": 2.00, "cached": 0.10}
        elif "pro" in name:
            return {"input_audio": 2.00, "input_text": 2.00, "output": 12.00, "cached": 0.50}
        else:
            return {"input_audio": 1.00, "input_text": 0.50, "output": 4.00, "cached": 0.25}

    def calculate(self) -> Dict[str, Any]:
        p1_rates = self._get_model_rates(self.pass1_model)
        p2_rates = self._get_model_rates(self.pass2_model)

        p1_in_cost = (self.pass1_input_tokens / 1_000_000) * p1_rates.get("input_audio", 0.40)
        p1_out_cost = (self.pass1_output_tokens / 1_000_000) * p1_rates.get("output", 1.50)
        p1_cost = p1_in_cost + p1_out_cost

        p2_in_cost = (self.pass2_input_tokens / 1_000_000) * p2_rates.get("input_text", 2.00)
        p2_out_cost = (self.pass2_output_tokens / 1_000_000) * p2_rates.get("output", 12.00)
        p2_cost = p2_in_cost + p2_out_cost

        total_tokens = (
            self.pass1_input_tokens + self.pass1_output_tokens +
            self.pass2_input_tokens + self.pass2_output_tokens
        )
        tokens_saved_est = int(total_tokens * 1.85) if total_tokens > 0 else 0

        return {
            "pass1_cost": p1_cost,
            "pass2_cost": p2_cost,
            "total_cost": p1_cost + p2_cost,
            "total_tokens": total_tokens,
            "tokens_saved_est": tokens_saved_est,
            "pass1_tokens": self.pass1_input_tokens + self.pass1_output_tokens,
            "pass2_tokens": self.pass2_input_tokens + self.pass2_output_tokens,
        }