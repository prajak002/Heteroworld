// Vite config for running the BoundlessNYC client (MIT; city data ODbL) locally with on-demand assets.
// It reuses the project's own vite.config.js and adds one middleware: a request for a binary-bank file
// (tiles/, models/, textures/, assets/) that is not on disk yet is fetched once from the public Hugging Face
// dataset mehmetkeremturkcan/boundless-nyc (v0.2.0) and cached under boundlessjs/public/, so only the
// neighbourhood the camera visits is ever downloaded. It also serves this folder's replay files under /twin/.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import base from '../../boundless-nyc/boundlessjs/vite.config.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(here, '../../boundless-nyc/boundlessjs');
const PUBLIC = path.join(ROOT, 'public');
const HF = 'https://huggingface.co/datasets/mehmetkeremturkcan/boundless-nyc/resolve/v0.2.0/Content';
const BANKS = ['tiles/', 'models/', 'textures/'];
const inflight = new Map();

async function fetchToCache(rel) {
  const dest = path.join(PUBLIC, rel);
  if (fs.existsSync(dest)) return true;
  if (inflight.has(rel)) return inflight.get(rel);
  const job = (async () => {
    const r = await fetch(`${HF}/${rel}`);
    if (!r.ok) { fs.appendFileSync(path.join(here, 'misses.log'), `${r.status} ${rel}\n`); return false; }
    const buf = Buffer.from(await r.arrayBuffer());
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(dest + '.part', buf);
    fs.renameSync(dest + '.part', dest);
    return true;
  })().finally(() => inflight.delete(rel));
  inflight.set(rel, job);
  return job;
}

const onDemand = {
  name: 'boundless-on-demand-assets',
  configureServer(server) {
    server.middlewares.use(async (req, res, next) => {
      const url = decodeURIComponent((req.url || '').split('?')[0]).replace(/^\/+/, '');
      if (url.startsWith('twin/')) {
        const file = path.join(here, url.slice(5));
        if (file.startsWith(here) && fs.existsSync(file) && fs.statSync(file).isFile()) {
          res.setHeader('content-type', file.endsWith('.json') ? 'application/json' : file.endsWith('.js') ? 'text/javascript' : 'application/octet-stream');
          fs.createReadStream(file).pipe(res);
          return;
        }
      }
      if (BANKS.some((b) => url.startsWith(b)) && !url.includes('..')) {
        let ok = false;
        try { ok = await fetchToCache(url); } catch (e) { console.warn('[on-demand]', url, e.message); }
        // Serve the cached file ourselves: Vite lists public files once at startup and does not see later downloads.
        const file = path.join(PUBLIC, url);
        if (ok && fs.existsSync(file)) {
          const ext = path.extname(file).toLowerCase();
          const type = { '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp',
                         '.ktx2': 'image/ktx2', '.glb': 'model/gltf-binary' }[ext] || 'application/octet-stream';
          res.setHeader('content-type', type);
          res.setHeader('content-length', fs.statSync(file).size);
          res.setHeader('cache-control', 'public, max-age=86400');
          fs.createReadStream(file).pipe(res);
          return;
        }
        res.statusCode = 404; res.end('not found'); return;
      }
      next();
    });
  },
};

export default {
  ...base,
  root: ROOT,
  publicDir: PUBLIC,
  plugins: [...(base.plugins || []), onDemand],
  server: { ...base.server, fs: { allow: [path.resolve(ROOT, '..'), here] } },
};
