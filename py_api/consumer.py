import pika
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

credentials = pika.PlainCredentials(os.getenv('RABBIT_USER'), os.getenv('RABBIT_PASSWORD'))
connection = pika.BlockingConnection(pika.ConnectionParameters(
    host='localhost',
    port=5672,
    credentials=credentials
))
channel = connection.channel()
channel.queue_declare(queue='test_queue', durable=False)

def callback(ch, method, properties, body):
    logger.info(f" [x] Received {body}")

channel.basic_consume(queue='test_queue', on_message_callback=callback, auto_ack=True)

logger.info(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()