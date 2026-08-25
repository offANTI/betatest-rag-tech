# rag-tech-docs

A personal RAG bot for technical documentation. The idea is simple: take docs for tools I actually use (Python, later SQL/dbt/Airflow), run them through my own pipeline, and get search that understands meaning, not just keywords. Eventually this gets wrapped in a Telegram bot — ask a question in chat, get an answer with a source link.

## How it works right now

```
   docs.python.org
         │
         │  sitemap didn't work (only 8 urls in there),
         │  so — recursive crawl following links instead
         ▼
   ┌─────────────┐
   │   crawler   │   BFS over /3/, blacklist filter,
   │  crawl.py   │   dedup (including #anchor variants)
   └──────┬──────┘
          │  raw html
          ▼
   ┌─────────────┐
   │  extractor  │   pull out div[role="main"],
   │ extract.py  │   strip headerlink junk, convert to .md
   └──────┬──────┘
          │  clean markdown with headings
          ▼
   ┌─────────────┐
   │  chunking   │   split by ## sections,
   │ chunker.py  │   re-split large sections by paragraph
   └──────┬──────┘
          │  chunks + metadata (json)
          ▼
     data/chunks/
     python_docs.json
```

30 Python docs pages → 881 chunks. Retrieval core is next in line.

## What's next

- [ ] embeddings via `sentence-transformers` (local, no API — 881 chunks isn't worth cloud calls)
- [ ] BM25 for lexical search
- [ ] hybrid search — combine both
- [ ] check quality against real questions
- [ ] Groq — only for the final step, once relevant chunks are found and need to become an actual answer
- [ ] wrap it in a Telegram bot
- [ ] add more sources: SQL/dbt/Airflow

## Structure

```
rag-tech-docs/
├── data/
│   ├── raw/          # raw html per source
│   ├── processed/     # clean markdown
│   └── chunks/        # chunked text, ready for embeddings
├── src/
│   ├── crawler/
│   ├── extractor/
│   └── chunking/
└── utils/
    └── logger.py
```