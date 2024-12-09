// src/nlp.ts

import { pipeline } from '@xenova/transformers';

// Initialize pipelines once to improve performance
let summarizer: any = null;
let ner: any = null;

export async function summarizeText(text: string): Promise<string> {
  if (!summarizer) {
    summarizer = await pipeline('summarization', 'DISLab/SummLlama3.2-3B');
  }
  const summary = await summarizer(text, {
    max_new_tokens: 150,
    temperature: 0.7,
  });
  return summary[0].summary_text;
}

export async function extractEntities(text: string): Promise<any[]> {
  if (!ner) {
    ner = await pipeline('ner', 'Xenova/dbmdz-bert-large-cased-finetuned-conll03-english');
  }
  const nerResults = await ner(text);
  const entities = nerResults.map((entity: any) => ({
    name: entity.word,
    type: entity.entity,
    start: entity.start,
    end: entity.end,
    score: entity.score,
  }));
  return entities;
}