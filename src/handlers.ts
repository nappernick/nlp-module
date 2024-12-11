// nlp-module/src/handlers.ts

import { getMongoClient } from './mongo';
import { sendToQueue } from './rabbitmq';

// Initialize caches
console.warn("Initializing caches");
const summaryCache = new Map<string, string>();
const entityCache = new Map<string, any[]>();

export const toolHandlers = {
  summarize_text: async (args: any) => {
    console.warn("Starting summarize_text handler", { args });
    
    if (!args.text) {
      console.warn("Missing text parameter");
      throw new Error('The "text" parameter is required.');
    }


    // Check cache
    console.warn("Checking summary cache");
    if (summaryCache.has(args.text)) {
      console.warn("Cache hit for summary", { text: args.text.substring(0, 50) });
      return { result: summaryCache.get(args.text) };
    }

    // Send task to RabbitMQ and wait for result
    console.warn("Sending to RabbitMQ queue");
    const summary = await sendToQueue('summarize_queue', { text: args.text });
    console.warn("Received summary from queue", { summaryLength: summary?.length });

    // Save to cache
    console.warn("Saving summary to cache");
    summaryCache.set(args.text, summary);

    // Save to MongoDB
    try {
      console.warn("Connecting to MongoDB");
      const client = await getMongoClient();
      const db = client.db('researcher');
      const collection = db.collection('summaries');
      console.warn("Saving summary to MongoDB");
      await collection.insertOne({ text: args.text, summary, timestamp: new Date() });
    } catch (error) {
      console.warn("MongoDB error", error);
      console.error('Error saving summary to MongoDB:', error);
    }

    console.warn("Returning summary result");
    return { result: summary };
  },

  extract_entities: async (args: any) => {
    console.warn("Starting extract_entities handler", { args });

    if (!args.text) {
      console.warn("Missing text parameter");
      throw new Error('The "text" parameter is required.');
    }

    // Check cache
    console.warn("Checking entity cache");
    if (entityCache.has(args.text)) {
      console.warn("Cache hit for entities", { text: args.text.substring(0, 50) });
      return { result: entityCache.get(args.text) };
    }

    // Send task to RabbitMQ and wait for result
    console.warn("Sending to RabbitMQ queue");
    const entities = await sendToQueue('entity_extraction_queue', { text: args.text });
    console.warn("Received entities from queue", { entityCount: entities?.length });

    // Save to cache
    console.warn("Saving entities to cache");
    entityCache.set(args.text, entities);

    // Save to MongoDB
    try {
      console.warn("Connecting to MongoDB");
      const client = await getMongoClient();
      const db = client.db('researcher');
      const collection = db.collection('entities');
      console.warn("Saving entities to MongoDB");
      await collection.insertOne({ text: args.text, entities, timestamp: new Date() });
    } catch (error) {
      console.warn("MongoDB error", error);
      console.error('Error saving entities to MongoDB:', error);
    }

    console.warn("Returning entities result");
    return { result: entities };
  },
};