# Multi-Agent Research Assistant

Given a broad research question, the assistant will plan subquestions, retrieve evidence from local PDFs/documents or optional live web search, write a sourced report, and have a critic review it before delivery.

## Proposed pipeline

`Question → Planner → Retrieval (local RAG / optional web) → Writer → Critic → Final report`

- **Planner:** break the question into focused search tasks.
- **Retrieval:** return excerpts with stable evidence IDs and source locators (such as PDF pages or web URLs); distinguish local from web evidence.
- **Writer:** build explicit claims linked to retrieved evidence.
- **Critic:** check whether evidence actually supports each claim, identify gaps, and reject or request a revision when necessary. A valid citation ID alone is **not** proof that a claim is true.
- **Output:** publish only a reviewed report with traceable citations; flag unanswered parts of the question instead of inventing facts.

## Build roadmap

1. **Foundation (done):** Python project, shared contracts, agent interfaces, and citation-link checks with tests. No agents or report generation are implemented yet.
2. **Local chunk extraction (in progress):** extract PDF structure, split into citation-rich chunks, quarantine low-quality pages. Embeddings and search indexing are not yet implemented.
3. **Planner:** decompose questions into scoped search tasks and test the plan quality.
4. **Local retrieval:** embed accepted chunks, index in local Qdrant, and retrieve relevant excerpts.
5. **Optional live web retrieval:** add a Tavily adapter with explicit credentials and source URLs; work offline when it is disabled.
6. **Writer:** synthesize a draft with claim-level evidence IDs and render readable citations.
7. **Critic and orchestration:** evaluate support and coverage, allow bounded revisions, and refuse unsupported final claims.
8. **Interface and evaluation:** add a CLI or API and test against a small, sourced set of research questions.

Qdrant is the chosen future local search store; no embeddings or Qdrant collection have been created. The provider, embedding model, and user interface are not selected yet. Do not commit private documents or API keys. Place local documents under `data/`, which is ignored by Git.

## PDF structure and chunks

The private AI/ML arXiv snapshot is under `data/ai-ml-arxiv-2026-11-07/`. Install the optional ingestion dependencies, then run a small pilot before processing every PDF:

    python3 -m pip install '.[ingest]'
    python3 -m research_assistant.ingest --max-papers 5

Omit `--max-papers` to continue through the full snapshot. Re-running the same command skips papers already processed under the same settings, PDF checksum, and chunking version. Outputs are saved under `data/ai-ml-structured-chunks/`, outside the 5 GB source snapshot: one JSON Lines chunk file and one status/diagnostics file per paper. Conversion is local; model files may be downloaded the first time Docling is used.

This stage uses PyMuPDF4LLM's page-aware layout extraction and escalates low-quality pages to Docling (at most one page per paper by default). It chunks prose by sentence with a **256-token ceiling** and up to **25 overlapping tokens** at word boundaries, using the `cl100k_base` tokenizer. Table chunks repeat the description and named column values for each complete row; very wide rows may split by named cell. **OCR, formula transcription and visual image interpretation are not enabled**: image-only pages and malformed or unverified figure/formula contents are **not** turned into factual evidence. Failed pages and uncertain elements are recorded in the per-paper status file; inspect those warnings before trusting the chunk set. Qdrant is not written by this command.

PyMuPDF4LLM uses an AGPL/commercial dual license; review its terms before distributing software built on this ingestion pipeline. Paper licenses are independent and are not verified by this tool.

## Run foundation checks

Requires Python 3.10 or later. Foundation tests need no third-party dependencies; ingestion needs its optional dependencies.

From the project directory:

    python3 -m unittest discover -s tests -v

To install the package for development, optionally create a virtual environment and run `python3 -m pip install -e .`. There is no end-to-end research CLI yet; the extraction command only writes local chunks.