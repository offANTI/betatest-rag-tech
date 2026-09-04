# rag-tech-docs

RAG-BOT
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
python pipeline.py <source_name>
```

## Current Status

At the moment, the pipeline follows a crawl → extract → embed workflow to prepare text and chunks for search and retrieval.

The RAG system crawls documentation directly from websites. Currently, it contains around 465 pages and 25,000 text chunks.

Search is performed using both dense retrieval and BM25. The results from both methods are then combined using the Reciprocal Rank Fusion (RRF) algorithm.

The system is managed and configured through a YAML configuration file.
What’s Next
Add additional documentation sources for other programming languages, as well as useful technical documentation such as SQL, dbt, and others.
Integrate Groq to generate answers for users and make the overall question-answering process more focused on technical and documentation-based queries before performing the search.
Later, integrate the system into a Telegram bot to provide a user-friendly interface.
Deploy the entire system to a server for production use.

## Structure

```
rag-tech-docs/
├── config/
│   └── sources.yaml
├── data/
│   ├── raw/          # raw html per source
│   ├── processed/     # clean markdown per source
│   └── chunks/        # chunks + embeddings per source
├── src/
│   ├── crawler/        # crawl.py, extract.py
│   ├── chunking/        # chunker.py
│   └── retrieval/        # dense_search.py, bm25.py, hybrid.py, embed.py
├── tests/
└── pipeline.py          # runs everything for one source
```

