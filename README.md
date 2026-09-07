# rag-tech-docs

This project, RAG-BOT, is currently under development. The initial goal is to search through Python documentation. Later, it will be expanded to include other programming languages and helpful tools, saving the need to manually search across different websites.

Right now, a basic RAG system has been built to crawl and search documentation. The plan is to add an LLM later to generate answers and then integrate everything into a Telegram bot.
## Pipeline structure

```
   source (e.g. docs.python.org)
         │
         ▼
   ┌─────────────┐
   │   crawler   │   walks the site within an allowed path,
   │  crawl.py   │   skips blacklisted pages, dedupes by URL,
   └──────┬──────┘   skips pages already downloaded
          │  raw html → data/raw/<source>/
          ▼
   ┌─────────────┐
   │  extractor  │   pulls the main content using a per-site
   │ extract.py  │   selector, converts it to markdown
   └──────┬──────┘   skips files that haven't changed
          │  clean markdown → data/processed/<source>/
          ▼
   ┌─────────────┐
   │  chunking   │   splits by headings, breaks up
   │ chunker.py  │   oversized sections by paragraph
   └──────┬──────┘
          │  chunks + metadata → data/chunks/<source>.json
          ▼
   ┌─────────────┐
   │  retrieval  │   dense search + BM25, combined with
   │             │   Reciprocal Rank Fusion, then reranked
   │             │   by a cross-encoder
   └─────────────┘
```

Run the whole thing with one command:

```
- "python pipeline.py <source_name>" source_name= #dbt_docs #python_docs
- "python -m src.retrieval.hybrid --sources python_docs dbt_docs"  
```
## Pytest
```
pytest -v     
```
## Current Status

At the moment, the pipeline follows a crawl → extract → embed workflow to prepare text and chunks for search and retrieval.

The RAG system crawls documentation directly from websites. It currently indexes 24,104 chunks from Python documentation and 2,358 chunks from dbt documentation, with more sources planned.
Search is performed using both dense retrieval and BM25. The results from both methods are combined using Reciprocal Rank Fusion (RRF) and then reranked by a cross-encoder for the final ranking.

The system is managed and configured through a YAML configuration file.
What’s Next
- Add a third documentation source (e.g. Airflow) to test how the pipeline generalizes beyond Python and dbt
- Integrate Groq to generate natural-language answers from retrieved chunks, instead of returning raw text passages
- Add query translation for non-English questions — the current embedding model handles English well but degrades on other languages against English-only docs
- Wrap the system in a Telegram bot as the user-facing interface
- Deploy to a server for production use

## Structure

```
rag-tech-docs/
├──common
│  └──config.py
├── config/
│   └── sources.yaml
├── data/
│   ├── raw/          # raw html per source
│   ├── processed/    # clean markdown per source
│   └── chunks/       # chunks + embeddings per source
├── src/
    ├── chunking/        # chunker.py
    ├── crawler/         # crawl.py, extract.py
│   └── retrieval/       # dense_search.py, bm25.py, hybrid.py, embed.py
├── tests/
├── utils/
└── pipeline.py          # runs everything for one source
```

