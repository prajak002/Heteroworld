#!/usr/bin/env bash
# Start the HETERO-WORLD replay in the BoundlessNYC digital twin (Two Bridges, Manhattan) and open it in Chrome.
cd "$(dirname "$0")"
pgrep -f "node server.mjs" >/dev/null || (nohup node server.mjs > server.log 2>&1 &)
sleep 1
open -a "Google Chrome" "http://127.0.0.1:5220/?api=1&twin=1&x=-2258&z=7847&y=4&gi=0&crowd=0&gfx=motionBlur:0"
