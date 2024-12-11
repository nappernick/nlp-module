// nlp-module/src/loggerInstance.ts

import { createLogger, format, transports } from 'winston';

const loggerInstance = createLogger({
  level: 'info',
  format: format.combine(
    format.timestamp(),
    format.printf(
      ({ timestamp, level, message, ...meta }) =>
        `${timestamp} [${level.toUpperCase()}]: ${message} ${
          Object.keys(meta).length ? JSON.stringify(meta) : ''
        }`
    )
  ),
  transports: [new transports.Console()],
});

export default loggerInstance;
export interface ILogger {
  info: (message: string, meta?: any) => void;
  error: (message: string, meta?: any) => void;
  warn: (message: string, meta?: any) => void;
  debug: (message: string, meta?: any) => void;
}