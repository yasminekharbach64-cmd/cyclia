"""
Healthcare Chatbot Logging System
Comprehensive logging for tracking questions, answers, errors, and performance
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import time

class HealthChatLogger:
    """Advanced logger for healthcare chatbot interactions"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        
        self.interactions_file = self.log_dir / "interactions.jsonl"
        self.errors_file = self.log_dir / "errors.log"
        self.metrics_file = self.log_dir / "metrics.jsonl"
        
        
        self._setup_error_logger()
    
    def _setup_error_logger(self):
        """Setup traditional logger for errors"""
        self.error_logger = logging.getLogger("HealthChatbot")
        self.error_logger.setLevel(logging.ERROR)
        
        handler = logging.FileHandler(self.errors_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.error_logger.addHandler(handler)
    
    def log_interaction(
        self,
        question: str,
        answer: str,
        language: str = "unknown",
        category: str = "general",
        response_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a complete interaction"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "language": language,
            "category": category,
            "response_time_seconds": round(response_time, 3),
            "question_length": len(question),
            "answer_length": len(answer),
            "metadata": metadata or {}
        }
        
        with open(self.interactions_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(interaction, ensure_ascii=False) + "\n")
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        question: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log errors with context"""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "question": question,
            "context": context or {}
        }
        
       
        self.error_logger.error(json.dumps(error_data, ensure_ascii=False))
    
    def log_metrics(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Log performance metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric_name,
            "value": value,
            "tags": tags or {}
        }
        
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric, ensure_ascii=False) + "\n")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics from logs"""
        if not self.interactions_file.exists():
            return {"total_interactions": 0}
        
        interactions = []
        with open(self.interactions_file, "r", encoding="utf-8") as f:
            for line in f:
                interactions.append(json.loads(line))
        
        if not interactions:
            return {"total_interactions": 0}
        
        total = len(interactions)
        avg_response_time = sum(i["response_time_seconds"] for i in interactions) / total
        
        languages = {}
        categories = {}
        for i in interactions:
            lang = i.get("language", "unknown")
            cat = i.get("category", "general")
            languages[lang] = languages.get(lang, 0) + 1
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_interactions": total,
            "avg_response_time": round(avg_response_time, 3),
            "languages": languages,
            "categories": categories,
            "last_interaction": interactions[-1]["timestamp"]
        }



def log_agent_call(logger: HealthChatLogger, agent_name: str):
    """Decorator to automatically log agent calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                logger.log_metrics(
                    f"{agent_name}_response_time",
                    response_time,
                    {"agent": agent_name}
                )
                return result
            except Exception as e:
                logger.log_error(
                    error_type=f"{agent_name}_error",
                    error_message=str(e),
                    context={"args": str(args), "kwargs": str(kwargs)}
                )
                raise
        return wrapper
    return decorator



if __name__ == "__main__":
    
    logger = HealthChatLogger()
    
    
    logger.log_interaction(
        question="¿Cuándo es peligrosa una fiebre?",
        answer="Una fiebre es peligrosa cuando supera 40°C...",
        language="es",
        category="health",
        response_time=1.23,
        metadata={"model": "mistral", "agent": "health_agent"}
    )
    
    
    logger.log_error(
        error_type="LLM_timeout",
        error_message="Model took too long to respond",
        question="What causes diabetes?"
    )
    
    
    stats = logger.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))