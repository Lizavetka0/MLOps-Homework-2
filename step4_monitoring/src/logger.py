import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
import colorama
from colorama import Fore, Style


class ColoredFormatter(logging.Formatter):
    """Форматтер с цветным выводом для консоли"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    RESET = Style.RESET_ALL
    
    def format(self, record):
        log_message = super().format(record)
        if hasattr(record, 'color'):
            return f"{record.color}{log_message}{self.RESET}"
        return log_message


class JsonFormatter(logging.Formatter):
    """Форматтер для вывода логов в JSON формате"""
    
    def format(self, record):
        log_object = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'extra'):
            log_object.update(record.extra)
        
        if record.exc_info:
            log_object['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_object, ensure_ascii=False)


class MonitoringLogger:
    """Кастомный логгер для системы мониторинга"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('monitoring')
        self.logger.setLevel(getattr(logging, config.logging.log_level))
        
        # Инициализация colorama для Windows
        colorama.init()
        
        self._setup_handlers()
        self._setup_metrics_logger()
    
    def _setup_handlers(self):
        """Настройка обработчиков логов"""
        self.logger.handlers.clear()
        
        # Консольный обработчик с цветами
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Файловый обработчик с ротацией
        log_file = Path(self.config.logging.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.config.logging.max_log_size_mb * 1024 * 1024,
            backupCount=self.config.logging.backup_count,
            encoding='utf-8'
        )
        json_formatter = JsonFormatter()
        file_handler.setFormatter(json_formatter)
        self.logger.addHandler(file_handler)
    
    def _setup_metrics_logger(self):
        """Настройка логгера для метрик"""
        self.metrics_logger = logging.getLogger('metrics')
        self.metrics_logger.setLevel(logging.INFO)
        self.metrics_logger.handlers.clear()
        
        metrics_file = Path(self.config.logging.metrics_file)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        metrics_handler = logging.FileHandler(
            metrics_file,
            encoding='utf-8'
        )
        metrics_handler.setFormatter(logging.Formatter('%(message)s'))
        self.metrics_logger.addHandler(metrics_handler)
    
    def _add_color(self, level: str, message: str) -> str:
        """Добавляет цвет к сообщению"""
        if not self.config.logging.console_colors:
            return message
        
        colors = {
            'SUCCESS': Fore.GREEN,
            'INFO': Fore.BLUE,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT,
            'DEBUG': Fore.CYAN
        }
        
        color = colors.get(level.upper(), Fore.RESET)
        return f"{color}{message}{Style.RESET_ALL}"
    
    def log(self, level: str, message: str, extra: Optional[Dict] = None):
        """Основной метод логирования"""
        colored_message = self._add_color(level, message)
        
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        
        if extra:
            log_method(colored_message, extra=extra)
        else:
            log_method(colored_message)
    
    def info(self, message: str, extra: Optional[Dict] = None):
        """Информационное сообщение"""
        self.log('INFO', message, extra)
    
    def warning(self, message: str, extra: Optional[Dict] = None):
        """Предупреждение"""
        self.log('WARNING', message, extra)
    
    def error(self, message: str, extra: Optional[Dict] = None):
        """Ошибка"""
        self.log('ERROR', message, extra)
    
    def success(self, message: str, extra: Optional[Dict] = None):
        """Успешное выполнение"""
        self.log('INFO', f"✅ {message}", extra)
    
    def critical(self, message: str, extra: Optional[Dict] = None):
        """Критическая ошибка"""
        self.log('CRITICAL', message, extra)
    
    def debug(self, message: str, extra: Optional[Dict] = None):
        """Отладочное сообщение"""
        self.log('DEBUG', message, extra)
    
    def log_metric(self, metric_name: str, value: float, 
                   status: str = "normal", tags: Optional[Dict] = None):
        """Логирование метрик в JSONL файл"""
        metric_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'metric': metric_name,
            'value': value,
            'status': status,
            'tags': tags or {}
        }
        
        self.metrics_logger.info(json.dumps(metric_data, ensure_ascii=False))
    
    def alert(self, alert_type: str, message: str, level: str = "warning",
              value: Optional[float] = None, threshold: Optional[float] = None):
        """Логирование алертов"""
        alert_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'type': alert_type,
            'level': level.upper(),
            'message': message,
            'value': value,
            'threshold': threshold
        }
        
        # Цветное отображение в консоли
        if level.upper() == 'CRITICAL':
            color = Fore.RED + Style.BRIGHT
            symbol = '🚨'
        elif level.upper() == 'WARNING':
            color = Fore.YELLOW
            symbol = '⚠️'
        else:
            color = Fore.GREEN
            symbol = '✅'
        
        colored_message = f"{symbol} {color}{message}{Style.RESET_ALL}"
        self.logger.info(colored_message, extra={'alert': alert_data})
        
        # Сохранение в метрики
        self.log_metric(
            f"alert_{alert_type}",
            1.0 if level.upper() in ['WARNING', 'CRITICAL'] else 0.0,
            status=level,
            tags={'type': alert_type, 'message': message}
        )
