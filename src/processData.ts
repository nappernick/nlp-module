// nlp-module/src/processData.ts

import { readFile } from 'fs/promises';
import fetch from 'node-fetch'; // Replace 'bun' with 'node-fetch' for compatibility
import { MongoClient } from 'mongodb';

const uri = 'mongodb+srv://neo4j:beepbeep@cluster0.msp0q.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'; // Replace with your MongoDB URI
const client = new MongoClient(uri);
const dbName = 'your_database'; // Replace with your actual database name

// Function to read the input data
async function readInputData(filePath: string): Promise<string> {
  try {
    const data = await readFile(filePath, 'utf8');
    return data;
  } catch (error) {
    console.error(`Error reading file ${filePath}:`, error);
    throw error;
  }
}

// Function to send text to the summarization endpoint
async function summarizeText(text: string): Promise<string> {
  try {
    const response = await fetch('http://localhost:6666/summarize_text', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
    }

    const data: Record<string,any> = await response.json() as Record<string, any>;
    if (data.error) {
      throw new Error(`API error: ${data.error}`);
    }

    return data.result;
  } catch (error) {
    console.error('Error during summarization:', error);
    throw error;
  }
}

// (Optional) Function to send text to the entity extraction endpoint
async function extractEntities(text: string): Promise<any[]> {
  try {
    const response = await fetch('http://localhost:6666/extract_entities', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
    }

    const data: Record<string,any> = await response.json() as Record<string, any>;
    if (data.error) {
      throw new Error(`API error: ${data.error}`);
    }

    return data.result;
  } catch (error) {
    console.error('Error during entity extraction:', error);
    throw error;
  }
}

// Function to split text into manageable chunks
function splitTextIntoChunks(text: string, maxChunkSize: number): string[] {
  const chunks = [];
  let startIndex = 0;

  while (startIndex < text.length) {
    let endIndex = Math.min(startIndex + maxChunkSize, text.length);

    // Try to split at the end of a sentence
    if (endIndex < text.length) {
      const lastPeriodIndex = text.lastIndexOf('.', endIndex);
      if (lastPeriodIndex > startIndex) {
        endIndex = lastPeriodIndex + 1;
      }
    }
    const chunk = text.slice(startIndex, endIndex).trim();
    chunks.push(chunk);
    startIndex = endIndex;
  }

  return chunks;
}

// Function to save summaries to MongoDB
async function saveSummaryToDB(
  dbClient: MongoClient,
  data: any,
  collectionName: string
): Promise<void> {
  try {
    const db = dbClient.db(dbName);
    const collection = db.collection(collectionName);
    await collection.insertOne({ ...data, timestamp: new Date() });
    console.log(`Data saved to collection "${collectionName}".`);
  } catch (error) {
    console.error(`Error saving to MongoDB collection "${collectionName}":`, error);
  }
}

// Main function to process the data
(async () => {
  const filePath = '/path/to/your/input.txt'; // Replace with your actual file path

  try {
    // Step 0: Connect to MongoDB
    await client.connect();

    // Step 1: Read the input data
    const text = await readInputData(filePath);

    // Step 2: Split the text into manageable chunks
    const maxChunkSize = 2000; // Adjust based on model capacity
    const chunks = splitTextIntoChunks(text, maxChunkSize);
    console.log(`Total chunks: ${chunks.length}`);

    // Step 3: Process each chunk sequentially
    const summaries: string[] = [];
    const cache = new Map<string, string>();

    for (let index = 0; index < chunks.length; index++) {
      const chunk = chunks[index];
      console.log(`Processing chunk ${index + 1}/${chunks.length}...`);

      // Check if this chunk has already been processed
      if (cache.has(chunk)) {
        console.log('Using cached summary.');
        summaries.push(cache.get(chunk)!);
        continue;
      }

      try {
        // Summarize the chunk
        const summary = await summarizeText(chunk);
        summaries.push(summary);
        cache.set(chunk, summary);

        // Save individual chunk summaries to MongoDB
        await saveSummaryToDB(client, { chunk, summary }, 'chunk_summaries');

        // Optionally, extract entities from the chunk
        const entities = await extractEntities(chunk);
        await saveSummaryToDB(client, { chunk, entities }, 'chunk_entities');

      } catch (error) {
        console.error(`Failed to process chunk ${index + 1}:`, error);
        // Continue with the next chunk
        summaries.push(''); // Placeholder for failed summary
        continue;
      }
    }

    // Step 4: Combine the summaries
    console.log('Combining summaries...');
    const combinedSummary = summaries.join(' ');

    // Step 5: Further summarize the combined summary
    console.log('Generating final summary...');
    const finalSummary = await summarizeText(combinedSummary);

    // Step 6: Save final summary to MongoDB
    await saveSummaryToDB(client, { finalSummary }, 'final_summaries');

    // Output the final summary
    console.log('\nFinal Summary:\n', finalSummary);

  } catch (error) {
    console.error('An error occurred:', error);
  } finally {
    // Close MongoDB connection
    await client.close();
  }
})();