// nlp-module/src/mongoClient.ts

import { MongoClient } from 'mongodb';

const uri = 'mongodb+srv://neo4j:beepbeep@cluster0.msp0q.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'; // Replace with your MongoDB URI
const client = new MongoClient(uri);

let clientPromise: Promise<MongoClient> | null = null;

export async function getMongoClient(): Promise<MongoClient> {
  if (!clientPromise) {
    clientPromise = client.connect().then(() => client);
  }
  return clientPromise;
}