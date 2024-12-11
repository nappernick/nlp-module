// nlp-module/src/main.ts

import { toolHandlers } from './handlers';
import { createServer } from 'http';
import { parse } from 'url';
import { getMongoClient } from './mongo';

// Server setup
const server = createServer(async (req, res) => {
  console.warn("IN SERVER:", req.method)
  const parsedUrl = parse(req.url!, true);
  const { pathname } = parsedUrl;
  
  console.warn("PARSED PATH:", pathname)
  
  if (req.method === 'POST' && pathname) {
    console.warn("IN SERVER POST:", req)

    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
    });

    req.on('end', async () => {
      try {
        console.warn("IN END")
        const args = JSON.parse(body);
        console.warn("END ARGS", args)
        const handlerName = pathname.slice(1) as keyof typeof toolHandlers;
        console.warn("HANDLER NAME", handlerName)
        const handler = toolHandlers[handlerName];
        console.warn("HANDLER", handler)
        if (handler) {
          const result = await handler(args);
          console.warn("RESULT", result)
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(result));
        } else {
          console.warn("INNER CATCH")
          res.writeHead(404);
          res.end('Handler not found.');
        }
      } catch (error) {
        console.warn("OUTER CATCH", error)
        console.error('Error handling request:', error);
        res.writeHead(500);
        res.end('Internal server error.');
      }
    });
  } else {
    res.writeHead(404);
    res.end('Not found.');
  }
});

const PORT = process.env.PORT || 6666; // Changed port to 6666
server.listen(PORT, () => {
  console.log(`Server is listening on port ${PORT}`);
});

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('\nShutting down server...');
  server.close(async () => {
    console.log('Server closed.');

    // Close MongoDB connection
    try {
      const client = await getMongoClient();
      await client.close();
      console.log('MongoDB connection closed.');
    } catch (error) {
      console.error('Error closing MongoDB connection:', error);
    }

    process.exit(0);
  });
});