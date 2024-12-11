// nlp-module/src/rabbitmq.ts

import amqp from 'amqplib';

export async function sendToQueue(queueName: string, message: any): Promise<any> {
  return new Promise(async (resolve, reject) => {
    try {
      // Connect to RabbitMQ on the correct port
      const connection = await amqp.connect({
        protocol: 'amqp',
        hostname: 'localhost',
        port: 5672, // Change to the correct port if needed
        username: 'admin', // Update if you have different credentials
        password: 'NGVP12345', // Update if you have different credentials
      });


      // Additional error handling
      connection.on('error', (error) => {
        console.error('RabbitMQ connection error:', error);
        reject(error);
      });
      
      console.log(`Connected to RabbitMQ, creating channel...`);
      let channel = await connection.createChannel();

      channel.on('error', (error) => {
        console.error('RabbitMQ channel error:', error);
        reject(error);
      });
      console.log(`Channel created, asserting queue "${queueName}"...`);

      await channel.assertQueue(queueName, { durable: true });
      console.log(`Queue "${queueName}" asserted.`);

      // Assert a callback queue for receiving the response
      const { queue: callbackQueue } = await channel.assertQueue('', { exclusive: true });

      const correlationId = generateUuid();

      channel.consume(
        callbackQueue,
        (msg) => {
          if (msg && msg.properties.correlationId === correlationId) {
            const result = JSON.parse(msg.content.toString());
            resolve(result);
            channel.close();
            connection.close();
          }
        },
        { noAck: true }
      );

      channel.sendToQueue(queueName, Buffer.from(JSON.stringify(message)), {
        correlationId,
        replyTo: callbackQueue,
      });
    } catch (error) {
      reject(error);
    }

    return true;
  });
}

function generateUuid() {
  return Math.random().toString() + Math.random().toString() + Math.random().toString();
}