import amqp from 'amqplib';

async function testProducer() {
  try {
    const connection = await amqp.connect({
      protocol: 'amqp',
      hostname: 'localhost',
      port: 5672,
      username: 'admin',
      password: 'NGVP12345',
    });
    const channel = await connection.createChannel();
    const queue = 'test_queue';

    await channel.assertQueue(queue, { durable: false });
    channel.sendToQueue(queue, Buffer.from('Hello World!'));
    console.log(" [x] Sent 'Hello World!'");

    await channel.close();
    await connection.close();
  } catch (error) {
    console.error('Error in testProducer:', error);
  }
}

testProducer();