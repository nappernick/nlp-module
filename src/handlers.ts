// src/handlers.ts

import { summarizeText, extractEntities } from './nlp';

export const toolHandlers = {
  summarize_text: async (args: any) => {
    if (!args.text) {
      throw new Error('The "text" parameter is required.');
    }
    const summary = await summarizeText(args.text);
    return { result: summary };
  },
  extract_entities: async (args: any) => {
    if (!args.text) {
      throw new Error('The "text" parameter is required.');
    }
    const entities = await extractEntities(args.text);
    return { result: entities };
  },
};