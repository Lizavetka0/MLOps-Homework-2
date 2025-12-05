import asyncio
import sys
from pathlib import Path

# Добавление src в путь импорта
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import ConfigLoader
from src.logger import MonitoringLogger
from src.monitor import ServiceMonitor


async def main():
    """Основная функция запуска мониторинга"""
    print("Запуск системы мониторинга FastAPI сервиса")
    print("=" * 50)
    
    try:
        # Загрузка конфигурации
        config = ConfigLoader.load()
        
        # Инициализация логгера
        logger = MonitoringLogger(config)
        
        # Инициализация монитора
        monitor = ServiceMonitor(config, logger)
        
        # Запуск мониторинга
        await monitor.start_monitoring()
    
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        print("Создайте конфигурационный файл config/monitoring_config.yaml")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
