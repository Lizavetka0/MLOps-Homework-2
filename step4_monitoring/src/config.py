import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ServiceConfig:
    host: str = "localhost"
    port: int = 8000
    base_url: str = "http://localhost:8000"
    endpoints: Dict[str, str] = field(default_factory=lambda: {
        "health": "/health",
        "predict": "/predict"
    })


@dataclass
class MonitoringConfig:
    check_interval_seconds: int = 30
    samples_per_check: int = 3
    request_timeout_seconds: int = 10
    inference_test_interval_minutes: int = 5


@dataclass
class ThresholdsConfig:
    response_time_ms: Dict[str, float] = field(default_factory=lambda: {
        "warning": 2000,
        "critical": 5000
    })
    p95_latency_ms: Dict[str, float] = field(default_factory=lambda: {
        "warning": 3000,
        "critical": 6000
    })
    error_rate_percent: Dict[str, float] = field(default_factory=lambda: {
        "warning": 10,
        "critical": 25
    })
    consecutive_failures: Dict[str, int] = field(default_factory=lambda: {
        "warning": 3,
        "critical": 5
    })


@dataclass
class AlertsConfig:
    enabled: bool = True
    cooldown_minutes: int = 5
    notify_on: List[str] = field(default_factory=lambda: [
        "response_time",
        "error_rate",
        "consecutive_failures",
        "health_status"
    ])


@dataclass
class LoggingConfig:
    console_colors: bool = True
    log_level: str = "INFO"
    log_file: str = "logs/monitoring.log"
    metrics_file: str = "logs/metrics.jsonl"
    max_log_size_mb: int = 10
    backup_count: int = 5


@dataclass
class InferenceTestConfig:
    enabled: bool = True
    test_image_path: str = "test_images/sample.jpg"
    expected_fields: List[str] = field(default_factory=lambda: [
        "filename", "result", "status_code"
    ])


@dataclass
class MonitoringConfig:
    service: ServiceConfig
    monitoring: MonitoringConfig
    thresholds: ThresholdsConfig
    alerts: AlertsConfig
    logging: LoggingConfig
    inference_test: InferenceTestConfig


class ConfigLoader:
    @staticmethod
    def load(config_path: str = "config/monitoring_config.yaml") -> MonitoringConfig:
        """Загружает конфигурацию из YAML файла"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return ConfigLoader._create_config(config_data)
    
    @staticmethod
    def _create_config(data: Dict) -> MonitoringConfig:
        """Создает объект конфигурации из словаря"""
        return MonitoringConfig(
            service=ServiceConfig(**data.get('service', {})),
            monitoring=MonitoringConfig(**data.get('monitoring', {})),
            thresholds=ThresholdsConfig(**data.get('thresholds', {})),
            alerts=AlertsConfig(**data.get('alerts', {})),
            logging=LoggingConfig(**data.get('logging', {})),
            inference_test=InferenceTestConfig(**data.get('inference_test', {}))
        )
    
    @staticmethod
    def save(config: MonitoringConfig, config_path: str = "config/monitoring_config.yaml"):
        """Сохраняет конфигурацию в YAML файл"""
        config_dict = {
            'service': config.service.__dict__,
            'monitoring': config.monitoring.__dict__,
            'thresholds': config.thresholds.__dict__,
            'alerts': config.alerts.__dict__,
            'logging': config.logging.__dict__,
            'inference_test': config.inference_test.__dict__
        }
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
