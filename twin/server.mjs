// Static server for the built BoundlessNYC client (MIT; city data ODbL) with on-demand city assets.
// dist/ holds the built client; tiles/, models/ and textures/ are read from the shared cache in
// boundless-nyc/boundlessjs/public and fetched once from the public Hugging Face dataset when missing.
// /twin/* serves this folder's replay files. Usage: node server.mjs  ->  http://127.0.0.1:5220/?api=1&twin=1
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.join(here, 'dist');
const CACHE = path.resolve(here, '../../boundless-nyc/boundlessjs/public');
const HF = 'https://huggingface.co/datasets/mehmetkeremturkcan/boundless-nyc/resolve/v0.2.0/Content';
const BANKS = ['tiles/', 'models/', 'textures/'];
const PORT = Number(process.env.PORT || 5220);
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.ktx2': 'image/ktx2',
  '.glb': 'model/gltf-binary', '.wasm': 'application/wasm', '.ttf': 'font/ttf', '.cube': 'text/plain', '.txt': 'text/plain' };
const inflight = new Map();

async function ensure(rel) {
  const dest = path.join(CACHE, rel);
  if (fs.existsSync(dest)) return dest;
  if (!inflight.has(rel)) {
    inflight.set(rel, (async () => {
      const r = await fetch(`${HF}/${rel}`);
      if (!r.ok) { fs.appendFileSync(path.join(here, 'misses.log'), `${r.status} ${rel}\n`); return null; }
      const buf = Buffer.from(await r.arrayBuffer());
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.writeFileSync(dest + '.part', buf);
      fs.renameSync(dest + '.part', dest);
      return dest;
    })().finally(() => inflight.delete(rel)));
  }
  return inflight.get(rel);
}

function send(res, file, fresh = false) {
  const ext = path.extname(file).toLowerCase();
  res.writeHead(200, { 'content-type': TYPES[ext] || 'application/octet-stream', 'content-length': fs.statSync(file).size,
    'cache-control': ext === '.html' || fresh ? 'no-cache' : 'public, max-age=86400' });
  fs.createReadStream(file).pipe(res);
}

http.createServer(async (req, res) => {
  let url;
  try { url = decodeURIComponent(new URL(req.url, 'http://x').pathname).replace(/^\/+/, ''); } catch { res.writeHead(400); return res.end(); }
  if (url === '' ) url = 'index.html';
  if (url.includes('..')) { res.writeHead(403); return res.end(); }
  try {
    if (url.startsWith('twin/')) {
      const f = path.join(here, url.slice(5));
      if (fs.existsSync(f) && fs.statSync(f).isFile()) return send(res, f, true);
    } else if (BANKS.some((b) => url.startsWith(b))) {
      const f = await ensure(url);
      if (f) return send(res, f);
    } else {
      if (url === 'index.html' && /[?&]twin=/.test(req.url)) {
        // Inject the HETERO-WORLD replay module into the BoundlessNYC page.
        const html = fs.readFileSync(path.join(DIST, 'index.html'), 'utf8')
          .replace('</body>', '<script type="module" src="twin/replay.js"></script></body>');
        res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-cache' });
        return res.end(html);
      }
      for (const root of [DIST, CACHE]) {
        const f = path.join(root, url);
        if (fs.existsSync(f) && fs.statSync(f).isFile()) return send(res, f);
      }
    }
  } catch (e) { console.warn(url, e.message); }
  res.writeHead(404, { 'content-type': 'text/plain' }); res.end('not found');
}).listen(PORT, '127.0.0.1', () => console.log(`BoundlessNYC twin server on http://127.0.0.1:${PORT}/`));
