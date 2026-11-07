"""Resumable, offline PDF-to-JSONL chunk extraction (no embeddings or Qdrant writes).

Fast page structure is supplied by PyMuPDF4LLM's layout engine. Docling is
invoked only for unusable pages, with a bounded per-paper budget; ambiguous
tables/images/formulas are quarantined rather than invented as evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
from uuid import NAMESPACE_URL, uuid5

from research_assistant.chunking import (
    MAX_TOKENS, TOKENIZER_NAME, Element, clean_text,
    prose_chunks, table_chunks, token_count, tokenizer,
)


CONFIG_VERSION = "structure-v5"


def elements_from_fast_page(page: dict) -> list[Element]:
    text = page["text"]
    page_number = page["metadata"]["page_number"]
    return [
        Element(box["class"], text[box["pos"][0]:box["pos"][1]],
                page_number, box["index"], tuple(box["bbox"]))
        for box in page["page_boxes"]
    ]


@lru_cache(maxsize=1)
def _docling_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True,
                              enable_remote_services=False)
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
    )


def elements_from_docling(pdf: Path, page_no: int) -> list[Element]:
    """Use Docling's layout/table pipeline on exactly one original PDF page."""
    result = _docling_converter().convert(pdf, page_range=(page_no, page_no))
    if str(result.status).lower().split(".")[-1] != "success":
        raise ValueError(f"Docling conversion failed: {result.status}")
    elements = []
    for index, (item, _) in enumerate(result.document.iterate_items()):
        prov = (getattr(item, "prov", None) or [])
        if not prov or prov[0].page_no != page_no:
            continue
        name = str(item.label)
        if name.startswith("DocItemLabel."):
            name = name.split(".")[-1].lower()
        if name == "table":
            text = item.export_to_html(result.document, add_caption=False)
        else:
            text = getattr(item, "text", "") or ""
        bbox = prov[0].bbox
        elements.append(Element(name.replace("_", "-"), text, page_no, index,
                                (bbox.l, bbox.t, bbox.r, bbox.b)))
    return elements


def needs_docling(elements: list[Element]) -> bool:
    """Escalate unusable structure, not every page that happens to have an image."""
    body = sum(len(clean_text(e.text)) for e in elements if e.kind in ("text", "list-item"))
    from research_assistant.chunking import table_rows

    tables = [e for e in elements if e.kind == "table"]
    if any(_invalid_table(e.text, table_rows) for e in tables):
        return True
    return body < 60 and not tables


def _invalid_table(html, parse):
    try:
        parse(html)
    except (ValueError, KeyError):
        return True
    return False


def records_for_page(elements: list[Element], source: dict, section: str,
                     encoding, extractor: str, section_id: int = 0) -> tuple[list[dict], str, int, list[str]]:
    """Convert verified page structures to Qdrant-ready child records."""
    records: list[dict] = []
    warnings: list[str] = []
    pending_caption = ""
    page = elements[0].page if elements else 0
    for element in elements:
        kind = element.kind
        if kind in ("page-header", "page-footer", "footnote"):
            continue
        if kind in ("section-header", "title"):
            section = re.sub(r"^#+\s*", "", clean_text(element.text)) or section
            section_id += 1
            continue
        if kind == "caption":
            caption = clean_text(element.text)
            if caption.lower().startswith("table"):
                pending_caption = caption
            else:
                warnings.append("figure_caption_requires_review")
            continue
        if kind in ("picture", "figure", "image"):
            warnings.append("image_not_interpreted")
            continue
        if kind in ("formula", "equation"):
            warnings.append("formula_requires_verification")
            continue
        try:
            if kind == "table":
                if not pending_caption:
                    raise ValueError("Table caption missing or unverified")
                grouped = table_chunks(element.text, pending_caption, section, encoding)
                payloads = [(text, start, end) for text, start, end in grouped]
                pending_caption = ""
            elif kind in ("text", "list-item", "paragraph", "code"):
                payloads = [(text, None, None) for text in
                            prose_chunks(element.text, section, encoding)]
            else:
                continue
        except ValueError as exc:
            warnings.append(f"skipped_{kind}: {exc}")
            continue
        for part_no, (text, row_start, row_end) in enumerate(payloads):
            count = token_count(text, encoding)
            if not text or count > MAX_TOKENS:
                warnings.append(f"oversized_{kind}_not_indexed")
                continue
            reference = f"{source['arxiv_id']}:{page}:{element.position}:{kind}:{part_no}:{text}"
            records.append({
                "id": str(uuid5(NAMESPACE_URL, reference)),
                "parent_id": (f"{source['arxiv_id']}:table:{page}:{element.position}"
                              if kind == "table" else f"{source['arxiv_id']}:section:{section_id}"),
                "paper_id": source["arxiv_id"], "title": source["title"],
                "source_url": source["source_url"], "pdf_path": source["file"],
                "pdf_sha256": source["sha256"], "published_at": source["published"],
                "source_updated_at": source["updated"], "categories": source["categories"],
                "page_start": page, "page_end": page, "section": section,
                "element_type": kind, "element_index": element.position,
                "bbox": element.bbox, "table_row_start": row_start,
                "table_row_end": row_end, "extractor": extractor,
                "text": text, "tokens": count, "tokenizer": TOKENIZER_NAME,
                "chunking_version": CONFIG_VERSION,
            })
    return records, section, section_id, warnings


def process_paper(entry: dict, corpus: Path, output: Path, encoding,
                  docling_budget: int = 1) -> dict:
    import pymupdf
    import pymupdf4llm

    pdf = corpus / entry["file"]
    output.mkdir(parents=True, exist_ok=True)
    chunks_path = output / f"{entry['arxiv_id']}.jsonl"
    status_path = output / f"{entry['arxiv_id']}.status.json"
    if status_path.exists() and chunks_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if (existing.get("pdf_sha256") == entry["sha256"] and
                existing.get("version") == CONFIG_VERSION and
                existing.get("docling_budget") == docling_budget):
            return existing
    working = chunks_path.with_suffix(".jsonl.part")
    diagnostics: list[dict] = []
    total = 0
    section = ""
    section_id = 0
    configured_budget = docling_budget
    with pymupdf.open(pdf) as document, working.open("w", encoding="utf-8") as destination:
        for begin in range(0, document.page_count, 12):
            try:
                pages = pymupdf4llm.to_markdown(
                    document, pages=list(range(begin, min(begin + 12, document.page_count))),
                    page_chunks=True, use_ocr=False, table_output="html",
                    write_images=False, embed_images=False,
                )
            except Exception as exc:
                diagnostics.append({"pages": [begin + 1, min(begin + 12, document.page_count)],
                                    "issue": f"fast_extraction_failed: {type(exc).__name__}: {exc}"})
                continue
            for page in pages:
                page_no = page["metadata"]["page_number"]
                elements = elements_from_fast_page(page)
                extractor = "pymupdf4llm"
                if needs_docling(elements):
                    if docling_budget:
                        try:
                            candidate = elements_from_docling(pdf, page_no)
                            if not needs_docling(candidate):
                                elements = candidate
                                extractor = "docling"
                            else:
                                diagnostics.append({"page": page_no, "issue": "docling_did_not_resolve_structure"})
                        except Exception as exc:
                            diagnostics.append({"page": page_no, "issue":
                                                f"docling_failed: {type(exc).__name__}: {exc}"})
                        docling_budget -= 1
                    else:
                        diagnostics.append({"page": page_no, "issue": "docling_budget_exhausted"})
                    if needs_docling(elements):
                        if sum(len(clean_text(e.text)) for e in elements
                               if e.kind in ("text", "list-item")) < 60:
                            diagnostics.append({"page": page_no, "issue": "page_quarantined"})
                            continue
                        diagnostics.append({"page": page_no, "issue": "table_unverified_prose_only"})
                records, section, section_id, warnings = records_for_page(
                    elements, entry, section, encoding, extractor, section_id
                )
                if warnings:
                    diagnostics.append({"page": page_no, "issues": sorted(set(warnings))})
                for record in records:
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += len(records)
    working.replace(chunks_path)
    status = {"paper_id": entry["arxiv_id"], "pdf_sha256": entry["sha256"],
              "version": CONFIG_VERSION, "chunks": total,
              "docling_budget": configured_budget,
              "indexed_at": datetime.now(timezone.utc).isoformat(),
              "diagnostics": diagnostics}
    state_part = status_path.with_suffix(".json.part")
    state_part.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state_part.replace(status_path)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract local PDF chunks; no Qdrant writes")
    parser.add_argument("--corpus", type=Path, default=Path("data/ai-ml-arxiv-2026-11-07"))
    parser.add_argument("--output", type=Path, default=Path("data/ai-ml-structured-chunks"))
    parser.add_argument("--max-papers", type=int, default=0, help="0 means all papers")
    parser.add_argument("--docling-budget", type=int, default=1, help="max pages per paper; 0 disables fallback")
    args = parser.parse_args(argv)
    if args.max_papers < 0 or args.docling_budget < 0:
        parser.error("budgets must not be negative")
    encoding = tokenizer()
    totals: Counter = Counter()
    with (args.corpus / "manifest.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if args.max_papers and index >= args.max_papers:
                break
            entry = json.loads(line)
            try:
                result = process_paper(entry, args.corpus, args.output, encoding, args.docling_budget)
            except Exception as exc:
                print(f"ERROR {entry['arxiv_id']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                totals["failed"] += 1
                continue
            totals["papers"] += 1
            totals["chunks"] += result["chunks"]
            totals["quarantined_pages"] += sum(d.get("issue") == "page_quarantined" for d in result["diagnostics"])
            print(f"{index + 1}: {entry['arxiv_id']} -> {result['chunks']} chunks; "
                  f"{totals['quarantined_pages']} pages quarantined total", flush=True)
    print(dict(totals), flush=True)
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())