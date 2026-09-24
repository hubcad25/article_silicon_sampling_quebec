import fs from 'node:fs';
import path from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Datasets live in the repo, outside this app: served read-only by the dev server.
const DATASETS = path.resolve(__dirname, '../../data/datasets');

const datasets = () => ({
  name: 'datasets',
  configureServer(server) {
    server.middlewares.use('/api/files', (_req, res) => {
      const files = fs.existsSync(DATASETS)
        ? fs.readdirSync(DATASETS).filter((f) => f.endsWith('.jsonl') || f.endsWith('.csv')).sort()
        : [];
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify(files.filter((f) => f.endsWith('.jsonl'))));
    });
    server.middlewares.use('/data/datasets', (req, res, next) => {
      const name = path.basename(decodeURIComponent(req.url.split('?')[0]));
      const file = path.join(DATASETS, name);
      if (!name || !fs.existsSync(file)) return next();
      res.setHeader('Content-Type', name.endsWith('.csv') ? 'text/csv' : 'application/x-ndjson');
      fs.createReadStream(file).pipe(res);
    });
  },
});

export default defineConfig({
  plugins: [react(), datasets()],
  server: { port: 5173, open: false },
});
