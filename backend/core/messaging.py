"""
Kafka/Redpanda messaging system
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError

from .config import settings, KAFKA_TOPICS

logger = logging.getLogger(__name__)

# Global producer and consumers
producer: Optional[AIOKafkaProducer] = None
consumers: Dict[str, AIOKafkaConsumer] = {}
consumer_tasks: Dict[str, asyncio.Task] = {}


async def init_kafka():
    """Initialize Kafka producer and create topics"""
    global producer
    
    try:
        # Create producer
        producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_brokers_list,
            value_serializer=lambda v: json.dumps(
                v, default=_json_serializer, ensure_ascii=False
            ).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            compression_type="snappy",
            max_batch_size=16384,
            linger_ms=10,
            retry_backoff_ms=100,
            request_timeout_ms=30000,
        )
        
        await producer.start()
        
        # Create topics if they don't exist
        await create_topics()
        
        logger.info("Kafka producer initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Kafka: {e}")
        raise


async def create_topics():
    """Create Kafka topics if they don't exist"""
    try:
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError
        
        admin_client = KafkaAdminClient(
            bootstrap_servers=settings.kafka_brokers_list,
            client_id='towerco_admin'
        )
        
        topics = [
            NewTopic(
                name=topic_name,
                num_partitions=3,
                replication_factor=1
            )
            for topic_name in KAFKA_TOPICS.values()
        ]
        
        try:
            admin_client.create_topics(topics)
            logger.info("Kafka topics created successfully")
        except TopicAlreadyExistsError:
            logger.info("Kafka topics already exist")
        
        admin_client.close()
        
    except Exception as e:
        logger.warning(f"Could not create Kafka topics: {e}")


async def close_kafka():
    """Close Kafka connections"""
    global producer, consumers, consumer_tasks
    
    try:
        # Stop all consumer tasks
        for task_name, task in consumer_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"Consumer task {task_name} stopped")
        
        # Close all consumers
        for consumer_name, consumer in consumers.items():
            await consumer.stop()
            logger.info(f"Consumer {consumer_name} closed")
        
        # Close producer
        if producer:
            await producer.stop()
            logger.info("Kafka producer closed")
    
    except Exception as e:
        logger.error(f"Error closing Kafka connections: {e}")


class MessageProducer:
    """Kafka message producer"""
    
    @staticmethod
    async def send_message(
        topic: str, 
        message: Dict[str, Any], 
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """Send message to Kafka topic"""
        if not producer:
            logger.error("Kafka producer not initialized")
            return False
        
        try:
            # Add metadata to message
            enriched_message = {
                **message,
                "_metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "producer": "towerco-backend-api",
                    "topic": topic,
                }
            }
            
            # Convert headers to bytes
            kafka_headers = []
            if headers:
                kafka_headers = [
                    (k.encode('utf-8'), v.encode('utf-8')) 
                    for k, v in headers.items()
                ]
            
            await producer.send(
                topic=topic,
                value=enriched_message,
                key=key,
                headers=kafka_headers
            )
            
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send message to {topic}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message to {topic}: {e}")
            return False
    
    @staticmethod
    async def send_network_metric(
        site_id: str,
        tenant_id: str,
        technology: str,
        metric_name: str,
        metric_value: float,
        unit: str = None,
        quality_score: float = None,
        metadata: Dict[str, Any] = None
    ):
        """Send network metric event"""
        message = {
            "site_id": site_id,
            "tenant_id": tenant_id,
            "technology": technology,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "unit": unit,
            "quality_score": quality_score,
            "metadata": metadata or {}
        }
        
        return await MessageProducer.send_message(
            KAFKA_TOPICS["network_metrics"],
            message,
            key=f"{site_id}:{metric_name}"
        )
    
    @staticmethod
    async def send_energy_metric(
        site_id: str,
        tenant_id: str,
        energy_type: str,
        metric_name: str,
        metric_value: float,
        unit: str = None,
        efficiency_score: float = None,
        metadata: Dict[str, Any] = None
    ):
        """Send energy metric event"""
        message = {
            "site_id": site_id,
            "tenant_id": tenant_id,
            "energy_type": energy_type,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "unit": unit,
            "efficiency_score": efficiency_score,
            "metadata": metadata or {}
        }
        
        return await MessageProducer.send_message(
            KAFKA_TOPICS["energy_metrics"],
            message,
            key=f"{site_id}:{energy_type}:{metric_name}"
        )
    
    @staticmethod
    async def send_event(
        site_id: str,
        tenant_id: str,
        event_type: str,
        severity: str,
        source_system: str,
        title: str,
        description: str = None,
        impact_assessment: str = None,
        correlation_id: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Send event"""
        message = {
            "site_id": site_id,
            "tenant_id": tenant_id,
            "event_type": event_type,
            "severity": severity,
            "source_system": source_system,
            "title": title,
            "description": description,
            "impact_assessment": impact_assessment,
            "correlation_id": correlation_id,
            "metadata": metadata or {}
        }
        
        return await MessageProducer.send_message(
            KAFKA_TOPICS["events"],
            message,
            key=f"{site_id}:{event_type}",
            headers={"severity": severity}
        )
    
    @staticmethod
    async def send_kpi_calculation_request(
        kpi_name: str,
        site_id: str = None,
        tenant_id: str = None,
        time_range: Dict[str, str] = None
    ):
        """Send KPI calculation request"""
        message = {
            "kpi_name": kpi_name,
            "site_id": site_id,
            "tenant_id": tenant_id,
            "time_range": time_range,
            "calculation_type": "scheduled"
        }
        
        return await MessageProducer.send_message(
            KAFKA_TOPICS["kpi_calculations"],
            message,
            key=f"{kpi_name}:{site_id or 'all'}"
        )
    
    @staticmethod
    async def send_aiops_prediction_request(
        model_name: str,
        site_id: str,
        prediction_type: str,
        target_time: str,
        input_data: Dict[str, Any]
    ):
        """Send AIOps prediction request"""
        message = {
            "model_name": model_name,
            "site_id": site_id,
            "prediction_type": prediction_type,
            "target_time": target_time,
            "input_data": input_data
        }
        
        return await MessageProducer.send_message(
            KAFKA_TOPICS["aiops_predictions"],
            message,
            key=f"{model_name}:{site_id}"
        )
    
    @staticmethod
    async def send_alert(
        alert_type: str,
        severity: str,
        title: str,
        description: str,
        site_id: str = None,
        tenant_id: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Send alert"""
        message = {
            "alert_type": alert_type,
            "severity": severity,
            "title": title,
            "description": description,
            "site_id": site_id,
            "tenant_id": tenant_id,
            "metadata": metadata or {}
        }
        
        return await MessageProducer.send_message(
            KAFKA_TOPICS["alerts"],
            message,
            key=f"{alert_type}:{site_id or 'global'}",
            headers={"severity": severity}
        )
    
    @staticmethod
    async def send_notification(
        notification_type: str,
        recipient: str,
        title: str,
        content: str,
        channels: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Send notification"""
        message = {
            "notification_type": notification_type,
            "recipient": recipient,
            "title": title,
            "content": content,
            "channels": channels or ["email"],
            "metadata": metadata or {}
        }
        
        return await MessageProducer.send_message(
            KAFKA_TOPICS["notifications"],
            message,
            key=f"{notification_type}:{recipient}"
        )


class MessageConsumer:
    """Kafka message consumer base class"""
    
    def __init__(self, consumer_name: str, topics: List[str], group_id: str):
        self.consumer_name = consumer_name
        self.topics = topics
        self.group_id = group_id
        self.consumer = None
        self.running = False
    
    async def start(self):
        """Start consumer"""
        try:
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=settings.kafka_brokers_list,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                consumer_timeout_ms=1000,
            )
            
            await self.consumer.start()
            self.running = True
            
            logger.info(f"Consumer {self.consumer_name} started for topics: {self.topics}")
            
            # Start consuming messages
            await self._consume_messages()
            
        except Exception as e:
            logger.error(f"Failed to start consumer {self.consumer_name}: {e}")
            raise
    
    async def stop(self):
        """Stop consumer"""
        self.running = False
        if self.consumer:
            await self.consumer.stop()
            logger.info(f"Consumer {self.consumer_name} stopped")
    
    async def _consume_messages(self):
        """Consume messages from Kafka"""
        try:
            async for message in self.consumer:
                if not self.running:
                    break
                
                try:
                    await self.process_message(
                        topic=message.topic,
                        key=message.key,
                        value=message.value,
                        headers=dict(message.headers) if message.headers else {},
                        timestamp=message.timestamp,
                        partition=message.partition,
                        offset=message.offset
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Error processing message in {self.consumer_name}: {e}"
                    )
                    # Continue processing other messages
                    
        except Exception as e:
            logger.error(f"Consumer {self.consumer_name} error: {e}")
        finally:
            logger.info(f"Consumer {self.consumer_name} finished consuming")
    
    async def process_message(
        self,
        topic: str,
        key: Optional[str],
        value: Dict[str, Any],
        headers: Dict[str, bytes],
        timestamp: int,
        partition: int,
        offset: int
    ):
        """Process a single message - override in subclasses"""
        logger.info(
            f"Processing message from {topic}: key={key}, "
            f"partition={partition}, offset={offset}"
        )


def _json_serializer(obj):
    """JSON serializer for datetime and other objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


async def start_consumer(
    consumer_name: str,
    topics: List[str],
    group_id: str,
    message_handler: Callable
) -> bool:
    """Start a message consumer"""
    global consumers, consumer_tasks
    
    try:
        class DynamicConsumer(MessageConsumer):
            async def process_message(self, topic, key, value, headers, timestamp, partition, offset):
                await message_handler(topic, key, value, headers, timestamp, partition, offset)
        
        consumer = DynamicConsumer(consumer_name, topics, group_id)
        consumers[consumer_name] = consumer.consumer
        
        # Start consumer in background task
        task = asyncio.create_task(consumer.start())
        consumer_tasks[consumer_name] = task
        
        logger.info(f"Started consumer {consumer_name} for topics {topics}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to start consumer {consumer_name}: {e}")
        return False


# Dependency for FastAPI
def get_message_producer() -> MessageProducer:
    """Get message producer instance"""
    return MessageProducer()