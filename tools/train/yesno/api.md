# Yes or No: the private API on bymartins

What Codex adds to https://bymartins.ca/yesno so batches go up and Immy's answers come back without anyone moving
files by hand. tools/train/yesno/sync.py is the client on the Mac. The key lives only in two places: the site's secret
settings (`YESNO_API_KEY`) and `~/.littleguy/yesno.key` on the Mac (made with `openssl rand -hex 32`, mode 600).
Never in git, never in a chat.

## Instructions for Codex

Add a small private API to the yes/no page at bymartins.ca/yesno, for a script on my Mac. The page itself stays exactly
as it is, behind the ChatGPT sign-in.

**Auth**
- Every API request carries `Authorization: Bearer <key>`. Compare it in constant time with the server secret
  `YESNO_API_KEY` (an environment secret, never in the code or the repo). Wrong or missing key: `401` with
  `{"error": "unauthorized"}`, nothing else. If `YESNO_API_KEY` isn't set, every API request gets `503`.
- The API routes must NOT go through the ChatGPT sign-in redirect, and must not get a Cloudflare bot challenge: the
  caller is a script (curl or Python), so it needs plain JSON back, never a 302 or an HTML page. The key is the only
  gate for these routes.
- No CORS headers (a browser should never call these). Send `Cache-Control: no-store`. Don't log request or response
  bodies (they hold private conversations); logging method, path, status and size is fine.
- Rate limit: about 60 requests a minute per IP is plenty.

**1. Upload a batch: `POST /yesno/api/items`**
- Body: JSON Lines (`Content-Type: application/x-ndjson`), one item per line, in the exact format in the README
  (`id, batch, n, kind, when, heard, reply, cut`). Max body 2 MB, max 1000 items per request.
- Validate each line: `id` matches `^b\d+-\d{4}$`, `kind` is `reply` or `thought`, `batch`/`n` are integers, `cut` is
  a boolean, strings under 8000 characters. If any line is invalid, reject the whole request with `400` and say which
  line and why. Ignore unknown fields.
- Insert or replace by `id`, so uploading the same batch twice is harmless. Never touch existing answers.
- Reply `200` with `{"received": N, "new": N, "replaced": N, "total_items": N}`.
- The page shows new items without a redeploy (newest batch first, as it does now).

**2. Download answers: `GET /yesno/api/labels`**
- Optional `?since=<ms>` returns only answers saved or changed after that time (milliseconds since 1970).
- Reply `200`, JSON Lines, one line per judged item, the latest answer only:
  `{"id": "b1-0007", "label": "yes", "note": "", "t": 1792470000000}`. Same format as the page's Export button.
  Skipped items are not included.

**3. Status: `GET /yesno/api/status`**
- Reply `200` with `{"items": N, "judged": N, "yes": N, "no": N, "batches": [{"batch": 1, "items": N, "judged": N}]}`.

**Test it** with a throwaway key on a preview or dev deploy: upload two fake items, judge one on the page, download
the labels, check the status counts, then delete the fake items. Tell me the final URLs when it's live.
