# Multi-Agent Research Assistant

- Use Python 3.10+ and the standard library for the foundation.
- Keep the agent boundaries in `research_assistant/agents.py` and shared data in `research_assistant/models.py`.
- Preserve source locators (PDF page or web URL) and evidence IDs throughout the pipeline.
- Do not present citation-link checks as factual verification; the future critic must inspect whether sources actually support claims.
- Build one roadmap milestone at a time. Foundation and local chunk extraction are implemented; embeddings, Qdrant ingestion, and agents are not.
- Keep the PDF corpus, manifests, and extracted chunks in ignored `data/`; never commit downloaded papers.
- Run `python3 -m unittest discover -s tests -v` after changes.