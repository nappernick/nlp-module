// nlp-module/src/tools.ts

export const summarizeTextTool = {
  name: 'summarize_text',
  description: 'Summarizes the given text.',
  input_schema: {
    type: 'object',
    properties: {
      text: { type: 'string' },
    },
    required: ['text'],
    additionalProperties: false,
  },
};

export const extractEntitiesTool = {
  name: 'extract_entities',
  description: 'Extracts entities from the given text.',
  input_schema: {
    type: 'object',
    properties: {
      text: { type: 'string' },
    },
    required: ['text'],
    additionalProperties: false,
  },
};

export const tools = [summarizeTextTool, extractEntitiesTool];