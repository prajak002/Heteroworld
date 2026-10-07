// Local server for docs/hw_viewer (wraps index.html in the same skeleton the artifact host adds). Usage: node docs/hw_viewer/serve.mjs docs/hw_viewer -> http://127.0.0.1:5231/
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path';
const root = process.argv[2], port = 5231;
const T = { '.json': 'application/json', '.bin': 'application/octet-stream', '.html': 'text/html; charset=utf-8', '.js': 'text/javascript' };
http.createServer((req, res) => {
  let u = decodeURIComponent(new URL(req.url, 'http://x').pathname).replace(/^\/+/, '') || 'index.html';
  if (u.includes('..')) { res.writeHead(403); return res.end(); }
  const f = path.join(root, u);
  if (!fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  if (u === 'index.html') {
    res.writeHead(200, { 'content-type': T['.html'], 'cache-control': 'no-cache' });
    return res.end('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>' + fs.readFileSync(f, 'utf8') + '</body></html>');
  }
  res.writeHead(200, { 'content-type': T[path.extname(f)] || 'application/octet-stream', 'cache-control': 'no-cache' }); fs.createReadStream(f).pipe(res);
}).listen(port, '127.0.0.1', () => console.log('viewer test on', port));
