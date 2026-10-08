/**
 * TINY STATIC FILE SERVER
 * =======================
 * Serves the built `dist/` directory over http://127.0.0.1:<port> using only
 * node:http + node:fs (no new dependencies). Returns { url, close() }.
 */
import { createReadStream, statSync } from 'node:fs';
import type { Stats } from 'node:fs';
import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { extname, join, normalize, resolve, sep } from 'node:path';

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.ico': 'image/x-icon',
  '.webp': 'image/webp',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain; charset=utf-8',
};

export interface StaticServer {
  url: string;
  close(): Promise<void>;
}

export function serveDir(dir: string, port: number): Promise<StaticServer> {
  const root = resolve(dir);

  const server = createServer((req, res) => handleRequest(req, res, root));

  return new Promise<StaticServer>((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', () => {
      server.removeListener('error', reject);
      resolveListen({
        url: `http://127.0.0.1:${port}`,
        close: () =>
          new Promise<void>((resolveClose, rejectClose) => {
            server.close((err) => (err ? rejectClose(err) : resolveClose()));
          }),
      });
    });
  });
}

function handleRequest(req: IncomingMessage, res: ServerResponse, root: string): void {
  let pathname: string;
  try {
    const raw = (req.url ?? '/').split('?')[0]!.split('#')[0]!;
    pathname = decodeURIComponent(raw);
  } catch {
    res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Bad Request');
    return;
  }
  if (pathname === '/') pathname = '/index.html';

  // Reject path traversal: the resolved path must stay inside `root`.
  const filePath = normalize(join(root, pathname));
  if (filePath !== root && !filePath.startsWith(root + sep)) {
    res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Forbidden');
    return;
  }

  let stat: Stats;
  try {
    stat = statSync(filePath);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not Found');
    return;
  }
  if (!stat.isFile()) {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not Found');
    return;
  }

  const ext = extname(filePath).toLowerCase();
  res.writeHead(200, {
    'Content-Type': MIME[ext] ?? 'application/octet-stream',
    'Cache-Control': 'no-store',
    'Content-Length': stat.size,
  });
  const stream = createReadStream(filePath);
  stream.on('error', () => res.destroy());
  stream.pipe(res);
}
