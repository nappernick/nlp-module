// Import necessary modules
import { readFile } from 'fs/promises';
import { fetch } from 'bun';

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
    const response = await fetch('http://localhost:5000/summarize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    if (data.error) {
      throw new Error(`API error: ${data.error}`);
    }

    return data.summary;
  } catch (error) {
    console.error('Error during summarization:', error);
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

// Main function to process the data
(async () => {
  const filePath = '/Users/nicmattj/Documents/GenAI/mcp/nlp-module/test/data.txt';

  try {
    // Step 1: Read the input data
    const text = await readInputData(filePath);

    // Step 2: Split the text into manageable chunks
    const maxChunkSize = 2000; // Adjust based on model capacity
    const chunks = splitTextIntoChunks(text, maxChunkSize);
    console.log(`Total chunks: ${chunks.length}`);

    // Step 3: Summarize each chunk sequentially
    const summaries = [];
    for (const [index, chunk] of chunks.entries()) {
      console.log(`Summarizing chunk ${index + 1}/${chunks.length}...`);
      const summary = await summarizeText(chunk);
      summaries.push(summary);
    }

    // Step 4: Combine the summaries
    const combinedSummary = summaries.join(' ');

    // Step 5: Optionally, further summarize the combined summary
    console.log('Generating final summary...');
    const finalSummary = await summarizeText(combinedSummary);

    // Output the final summary
    console.log('\nFinal Summary:\n', finalSummary);
  } catch (error) {
    console.error('An error occurred:', error);
  }
})();