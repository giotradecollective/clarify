# CLARIFY backend (minimal)

A single-file Express server that gives the CLARIFY frontend (`clarify-mvp.html`)
the five endpoints it already expects, proxying the actual contract analysis
to the Anthropic API so the API key never reaches the browser.

## Run locally

```bash
npm install
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
npm start
```

Server listens on `http://localhost:3001` by default. In the CLARIFY frontend,
set the "Backend URL" / `apiBase` field to that address.

## Deploy on Render

1. Push this folder to a GitHub repo.
2. On Render: New → Web Service → connect the repo.
3. Build command: `npm install`
4. Start command: `npm start`
5. Add environment variables from `.env.example` under the service's
   Environment tab (at minimum `ANTHROPIC_API_KEY`).
6. Once deployed, set `ALLOWED_ORIGIN` to wherever you host the frontend
   HTML file, then redeploy.

## Endpoints

| Method | Path                          | Purpose                              |
|--------|-------------------------------|---------------------------------------|
| POST   | /api/sessions                 | Start a session, get a `sessionId`    |
| POST   | /api/sessions/:id/text        | Submit pasted contract text           |
| POST   | /api/sessions/:id/upload      | Upload a .txt/.pdf/.docx file         |
| POST   | /api/sessions/:id/analyze     | Run the analysis, get back `analysis` |
| POST   | /api/sessions/:id/ask         | Ask a follow-up question              |
| POST   | /api/sessions/:id/end         | Delete the session's in-memory data   |

Sessions live only in memory and expire automatically after 1 hour or on
server restart — nothing is written to disk.
