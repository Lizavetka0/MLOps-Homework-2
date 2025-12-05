import asyncio
import aiohttp
import time
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import random
import json


@dataclass
class RequestMetrics:
    """Метрики запроса"""
    endpoint: str
    response_time: float
    status_code: int
    success: bool
    timestamp: datetime


@dataclass
class ServiceMetrics:
    """Агрегированные метрики сервиса"""
    timestamp: datetime
    response_time_avg: float
    response_time_p95: float
    error_rate: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    consecutive_failures: int
    health_status: bool


class ServiceMonitor:
    """Монитор FastAPI сервиса"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.base_url = config.service.base_url
        self.endpoints = config.service.endpoints
        self.request_timeout = config.monitoring.request_timeout_seconds
        
        # История запросов
        self.request_history: List[RequestMetrics] = []
        self.metrics_history: List[ServiceMetrics] = []
        
        # Состояние мониторинга
        self.consecutive_failures = 0
        self.last_alert_time: Dict[str, datetime] = {}
        self.health_status = False
        
        # Тестовое изображение для инференса
        self.test_image_path = Path(config.inference_test.test_image_path)
        
        if not self.test_image_path.exists():
            self._create_sample_image()
    
    def _create_sample_image(self):
        """Создает тестовое изображение если его нет"""
        from PIL import Image, ImageDraw
        self.test_image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Создаем простое тестовое изображение
        img = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Test Image", fill='black')
        img.save(self.test_image_path)
        
        self.logger.info(f"Создано тестовое изображение: {self.test_image_path}")
    
    async def check_health(self) -> Tuple[bool, float]:
        """Проверка health endpoint"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}{self.endpoints['health']}"
                async with session.get(url, timeout=self.request_timeout) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        return True, response_time
                    else:
                        self.logger.warning(
                            f"Health check failed: {response.status}",
                            extra={'status_code': response.status}
                        )
                        return False, response_time
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(
                f"Health check error: {str(e)}",
                extra={'error': str(e)}
            )
            return False, response_time
    
    async def test_inference(self) -> Tuple[bool, float, Optional[Dict]]:
        """Тестирование инференса на /predict endpoint"""
        start_time = time.time()
        
        if not self.test_image_path.exists():
            self.logger.error(f"Test image not found: {self.test_image_path}")
            return False, 0, None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}{self.endpoints['predict']}"
                
                with open(self.test_image_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('file', f, filename=self.test_image_path.name)
                    
                    async with session.post(
                        url, 
                        data=data,
                        timeout=self.request_timeout * 2
                    ) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            # Проверка структуры ответа
                            if all(field in result for field in self.config.inference_test.expected_fields):
                                self.logger.success(
                                    f"Inference test passed: {response_time:.2f}ms",
                                    extra={'response_time': response_time}
                                )
                                return True, response_time, result
                            else:
                                self.logger.warning(
                                    f"Inference response missing fields",
                                    extra={'response': result}
                                )
                                return False, response_time, result
                        else:
                            error_text = await response.text()
                            self.logger.error(
                                f"Inference test failed: {response.status}",
                                extra={
                                    'status_code': response.status,
                                    'error': error_text
                                }
                            )
                            return False, response_time, None
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(
                f"Inference test error: {str(e)}",
                extra={'error': str(e)}
            )
            return False, response_time, None
    
    async def perform_request(self, endpoint: str, method: str = 'GET', 
                             data: Optional[Dict] = None) -> RequestMetrics:
        """Выполнение HTTP запроса и сбор метрик"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}{endpoint}"
                
                if method.upper() == 'GET':
                    async with session.get(url, timeout=self.request_timeout) as response:
                        response_time = (time.time() - start_time) * 1000
                        success = response.status < 400
                        
                        return RequestMetrics(
                            endpoint=endpoint,
                            response_time=response_time,
                            status_code=response.status,
                            success=success,
                            timestamp=datetime.now()
                        )
                
                elif method.upper() == 'POST' and data:
                    async with session.post(url, data=data, timeout=self.request_timeout) as response:
                        response_time = (time.time() - start_time) * 1000
                        success = response.status < 400
                        
                        return RequestMetrics(
                            endpoint=endpoint,
                            response_time=response_time,
                            status_code=response.status,
                            success=success,
                            timestamp=datetime.now()
                        )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(
                f"Request failed: {endpoint} - {str(e)}",
                extra={'endpoint': endpoint, 'error': str(e)}
            )
            
            return RequestMetrics(
                endpoint=endpoint,
                response_time=response_time,
                status_code=0,
                success=False,
                timestamp=datetime.now()
            )
    
    def calculate_metrics(self, requests: List[RequestMetrics]) -> ServiceMetrics:
        """Расчет агрегированных метрик"""
        if not requests:
            return ServiceMetrics(
                timestamp=datetime.now(),
                response_time_avg=0,
                response_time_p95=0,
                error_rate=100,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                consecutive_failures=self.consecutive_failures,
                health_status=self.health_status
            )
        
        response_times = [r.response_time for r in requests if r.success]
        successful_requests = [r for r in requests if r.success]
        failed_requests = [r for r in requests if not r.success]
        
        total_requests = len(requests)
        successful_count = len(successful_requests)
        failed_count = len(failed_requests)
        
        # Среднее время ответа
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # P95 латенси
        if response_times:
            sorted_times = sorted(response_times)
            p95_index = int(0.95 * len(sorted_times))
            p95_latency = sorted_times[p95_index]
        else:
            p95_latency = 0
        
        # Error rate
        error_rate = (failed_count / total_requests * 100) if total_requests > 0 else 100
        
        # Обновление счетчика последовательных ошибок
        if failed_count > 0:
            self.consecutive_failures += failed_count
        else:
            self.consecutive_failures = 0
        
        return ServiceMetrics(
            timestamp=datetime.now(),
            response_time_avg=avg_response_time,
            response_time_p95=p95_latency,
            error_rate=error_rate,
            total_requests=total_requests,
            successful_requests=successful_count,
            failed_requests=failed_count,
            consecutive_failures=self.consecutive_failures,
            health_status=self.health_status
        )
    
    def check_thresholds(self, metrics: ServiceMetrics) -> List[Dict]:
        """Проверка метрик на превышение пороговых значений"""
        alerts = []
        
        # Проверка времени ответа
        if metrics.response_time_avg > self.config.thresholds.response_time_ms['critical']:
            alerts.append({
                'type': 'response_time',
                'level': 'critical',
                'message': f'Критическое время ответа: {metrics.response_time_avg:.2f}ms',
                'value': metrics.response_time_avg,
                'threshold': self.config.thresholds.response_time_ms['critical']
            })
        elif metrics.response_time_avg > self.config.thresholds.response_time_ms['warning']:
            alerts.append({
                'type': 'response_time',
                'level': 'warning',
                'message': f'Высокое время ответа: {metrics.response_time_avg:.2f}ms',
                'value': metrics.response_time_avg,
                'threshold': self.config.thresholds.response_time_ms['warning']
            })
        
        # Проверка P95 латенси
        if metrics.response_time_p95 > self.config.thresholds.p95_latency_ms['critical']:
            alerts.append({
                'type': 'p95_latency',
                'level': 'critical',
                'message': f'Критическая P95 латенси: {metrics.response_time_p95:.2f}ms',
                'value': metrics.response_time_p95,
                'threshold': self.config.thresholds.p95_latency_ms['critical']
            })
        elif metrics.response_time_p95 > self.config.thresholds.p95_latency_ms['warning']:
            alerts.append({
                'type': 'p95_latency',
                'level': 'warning',
                'message': f'Высокая P95 латенси: {metrics.response_time_p95:.2f}ms',
                'value': metrics.response_time_p95,
                'threshold': self.config.thresholds.p95_latency_ms['warning']
            })
        
        # Проверка error rate
        if metrics.error_rate > self.config.thresholds.error_rate_percent['critical']:
            alerts.append({
                'type': 'error_rate',
                'level': 'critical',
                'message': f'Критический error rate: {metrics.error_rate:.2f}%',
                'value': metrics.error_rate,
                'threshold': self.config.thresholds.error_rate_percent['critical']
            })
        elif metrics.error_rate > self.config.thresholds.error_rate_percent['warning']:
            alerts.append({
                'type': 'error_rate',
                'level': 'warning',
                'message': f'Высокий error rate: {metrics.error_rate:.2f}%',
                'value': metrics.error_rate,
                'threshold': self.config.thresholds.error_rate_percent['warning']
            })
        
        # Проверка последовательных ошибок
        if metrics.consecutive_failures >= self.config.thresholds.consecutive_failures['critical']:
            alerts.append({
                'type': 'consecutive_failures',
                'level': 'critical',
                'message': f'Критическое количество последовательных ошибок: {metrics.consecutive_failures}',
                'value': metrics.consecutive_failures,
                'threshold': self.config.thresholds.consecutive_failures['critical']
            })
        elif metrics.consecutive_failures >= self.config.thresholds.consecutive_failures['warning']:
            alerts.append({
                'type': 'consecutive_failures',
                'level': 'warning',
                'message': f'Много последовательных ошибок: {metrics.consecutive_failures}',
                'value': metrics.consecutive_failures,
                'threshold': self.config.thresholds.consecutive_failures['warning']
            })
        
        # Проверка health status
        if not metrics.health_status:
            alerts.append({
                'type': 'health_status',
                'level': 'critical',
                'message': 'Сервис недоступен',
                'value': 0,
                'threshold': 1
            })
        
        return alerts
    
    def should_alert(self, alert_type: str, level: str) -> bool:
        """Проверка необходимости отправки алерта (cooldown)"""
        if not self.config.alerts.enabled:
            return False
        
        alert_key = f"{alert_type}_{level}"
        now = datetime.now()
        
        if alert_key in self.last_alert_time:
            time_since_last_alert = now - self.last_alert_time[alert_key]
            cooldown = timedelta(minutes=self.config.alerts.cooldown_minutes)
            
            if time_since_last_alert < cooldown:
                return False
        
        self.last_alert_time[alert_key] = now
        return True
    
    def log_metrics(self, metrics: ServiceMetrics):
        """Логирование метрик"""
        # Определение общего статуса
        if metrics.error_rate > self.config.thresholds.error_rate_percent['critical']:
            overall_status = 'critical'
        elif (metrics.response_time_avg > self.config.thresholds.response_time_ms['critical'] or
              metrics.consecutive_failures >= self.config.thresholds.consecutive_failures['critical']):
            overall_status = 'critical'
        elif metrics.error_rate > self.config.thresholds.error_rate_percent['warning']:
            overall_status = 'warning'
        elif (metrics.response_time_avg > self.config.thresholds.response_time_ms['warning'] or
              metrics.consecutive_failures >= self.config.thresholds.consecutive_failures['warning']):
            overall_status = 'warning'
        else:
            overall_status = 'normal'
        
        # Логирование отдельных метрик
        self.logger.log_metric(
            'response_time_avg',
            metrics.response_time_avg,
            status=overall_status
        )
        
        self.logger.log_metric(
            'response_time_p95',
            metrics.response_time_p95,
            status=overall_status
        )
        
        self.logger.log_metric(
            'error_rate',
            metrics.error_rate,
            status=overall_status
        )
        
        self.logger.log_metric(
            'consecutive_failures',
            metrics.consecutive_failures,
            status=overall_status
        )
        
        self.logger.log_metric(
            'health_status',
            1.0 if metrics.health_status else 0.0,
            status='normal' if metrics.health_status else 'critical'
        )
        
        # Вывод в консоль
        status_color = {
            'normal': '🟢',
            'warning': '🟡',
            'critical': '🔴'
        }
        
        self.logger.info(
            f"{status_color.get(overall_status, '⚪')} "
            f"Metrics: RT={metrics.response_time_avg:.2f}ms, "
            f"P95={metrics.response_time_p95:.2f}ms, "
            f"ER={metrics.error_rate:.2f}%, "
            f"CF={metrics.consecutive_failures}, "
            f"Health={'✅' if metrics.health_status else '❌'}",
            extra=metrics.__dict__
        )
    
    async def monitoring_cycle(self):
        """Один цикл мониторинга"""
        self.logger.info(f"Начало цикла мониторинга: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        requests = []
        
        # Проверка health endpoint
        health_ok, health_response_time = await self.check_health()
        self.health_status = health_ok
        
        requests.append(RequestMetrics(
            endpoint=self.endpoints['health'],
            response_time=health_response_time,
            status_code=200 if health_ok else 500,
            success=health_ok,
            timestamp=datetime.now()
        ))
        
        # Тестирование нескольких запросов
        for _ in range(self.config.monitoring.samples_per_check - 1):
            # Чередуем endpoints
            endpoint = self.endpoints['health']
            req_metrics = await self.perform_request(endpoint)
            requests.append(req_metrics)
            
            # Небольшая задержка между запросами
            await asyncio.sleep(0.5)
        
        # Расчет метрик
        metrics = self.calculate_metrics(requests)
        self.metrics_history.append(metrics)
        
        # Проверка пороговых значений
        alerts = self.check_thresholds(metrics)
        
        # Логирование алертов
        for alert in alerts:
            if self.should_alert(alert['type'], alert['level']):
                self.logger.alert(
                    alert_type=alert['type'],
                    message=alert['message'],
                    level=alert['level'],
                    value=alert['value'],
                    threshold=alert['threshold']
                )
        
        # Логирование метрик
        self.log_metrics(metrics)
        
        # Тестирование инференса (периодически)
        current_minute = datetime.now().minute
        if (self.config.inference_test.enabled and 
            current_minute % self.config.monitoring.inference_test_interval_minutes == 0):
            
            self.logger.info("Запуск тестирования инференса...")
            inference_ok, inference_time, result = await self.test_inference()
            
            if inference_ok:
                self.logger.log_metric(
                    'inference_time',
                    inference_time,
                    status='normal'
                )
            else:
                self.logger.log_metric(
                    'inference_failure',
                    1.0,
                    status='critical'
                )
        
        # Очистка старых данных
        self._cleanup_old_data()
    
    def _cleanup_old_data(self):
        """Очистка старых метрик из истории"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        self.request_history = [
            r for r in self.request_history 
            if r.timestamp > cutoff_time
        ]
        
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]
    
    async def start_monitoring(self):
        """Запуск непрерывного мониторинга"""
        self.logger.success(f"Запуск мониторинга сервиса: {self.base_url}")
        self.logger.info(f"Интервал проверки: {self.config.monitoring.check_interval_seconds} сек")
        
        try:
            while True:
                await self.monitoring_cycle()
                await asyncio.sleep(self.config.monitoring.check_interval_seconds)
        
        except KeyboardInterrupt:
            self.logger.info("Мониторинг остановлен пользователем")
        
        except Exception as e:
            self.logger.error(f"Ошибка в мониторинге: {str(e)}", extra={'error': str(e)})
