"""Page-traceable, structure-aware 256-token chunks for the local PDF corpus.

No embeddings are generated here. Tables are serialized as named cells; no
unverified image or equation contents are presented as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
import re


TOKENIZER_NAME = "cl100k_base"
MAX_TOKENS = 256
OVERLAP_TOKENS = 25


def tokenizer():
    import tiktoken  # Optional ingestion dependency; foundation stays dependency-free.

    return tiktoken.get_encoding(TOKENIZER_NAME)


def token_count(text: str, encoding) -> int:
    return len(encoding.encode(text))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text).replace("**", "")).strip()


@dataclass(frozen=True)
class Element:
    kind: str
    text: str
    page: int
    position: int = 0
    bbox: tuple[float, float, float, float] | None = None


class _TableParser(HTMLParser):
    """Keep cell span information for multi-level table headers."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, bool, int, int]]] = []
        self.row: list[tuple[str, bool, int, int]] | None = None
        self.cell: list[str] | None = None
        self.is_header = False
        self.colspan = 1
        self.rowspan = 1

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            attributes = dict(attrs)
            self.is_header = tag == "th"
            self.colspan = min(100, max(1, int(attributes.get("colspan", "1"))))
            self.rowspan = min(100, max(1, int(attributes.get("rowspan", "1"))))
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append((clean_text("".join(self.cell)), self.is_header, self.colspan, self.rowspan))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def table_rows(html: str) -> tuple[list[str], list[list[str]]]:
    """Return unique column paths and complete data rows, including merged cells."""
    parser = _TableParser()
    parser.feed(html)
    if not parser.rows:
        raise ValueError("Table contains no rows")
    grid: list[list[tuple[str, bool]]] = []
    pending: dict[int, tuple[int, str, bool]] = {}
    for row in parser.rows:
        current: list[tuple[str, bool]] = []
        col = 0

        def place_pending() -> None:
            nonlocal col
            while col in pending:
                remaining, value, is_header = pending[col]
                current.append((value, is_header))
                if remaining == 1:
                    del pending[col]
                else:
                    pending[col] = (remaining - 1, value, is_header)
                col += 1

        for value, is_header, span, row_span in row:
            place_pending()
            for _ in range(span):
                current.append((value, is_header))
                if row_span > 1:
                    pending[col] = (row_span - 1, value, is_header)
                col += 1
        place_pending()
        grid.append(current)
    width = max(map(len, grid))
    if width > 100 or any(len(row) != width for row in grid):
        raise ValueError("Table has inconsistent column counts")
    header_rows = 0
    for row in grid:
        if any(is_header for _, is_header in row):
            header_rows += 1
        else:
            break
    if header_rows == 0 or header_rows == len(grid):
        raise ValueError("Table has no usable header or data rows")
    column_names = []
    for col in range(width):
        names = [grid[row][col][0] for row in range(header_rows)]
        names = [name for i, name in enumerate(names) if name and name not in names[:i]]
        column_names.append(" / ".join(names) or f"Column {col + 1}")
    return column_names, [[value for value, _ in row] for row in grid[header_rows:]]


def _pack_units(prefix: str, units: list[str], encoding, limit: int) -> list[str]:
    """Pack whole semantic units; split only a unit too large to fit."""
    if token_count(prefix, encoding) >= limit:
        raise ValueError("Context alone exceeds token limit")
    output: list[str] = []
    current: list[str] = []
    for unit in units:
        unit = clean_text(unit)
        if not unit:
            continue
        pieces = re.findall(r"\S+\s*", unit)
        if token_count(prefix + " ".join(current + [unit]), encoding) <= limit:
            current.append(unit)
            continue
        if current:
            output.append(prefix + " ".join(current))
            current = []
        if token_count(prefix + unit, encoding) <= limit:
            current.append(unit)
            continue
        part = ""
        for word in pieces:
            if token_count(prefix + part + word, encoding) > limit:
                if not part:
                    raise ValueError("Single word exceeds token limit")
                output.append(prefix + part.strip())
                part = ""
            part += word
        if part:
            current = [part.strip()]
    if current:
        output.append(prefix + " ".join(current))
    return output


def prose_chunks(text: str, section: str, encoding, limit: int = MAX_TOKENS,
                 overlap: int = OVERLAP_TOKENS) -> list[str]:
    """Split at sentence boundaries, with up to 25 token-aligned word overlap."""
    prefix = f"Section: {section}. " if section else ""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean_text(text))
    if not 0 <= overlap < limit:
        raise ValueError("overlap must be smaller than the chunk limit")
    # Reserve room for overlap in all but the first chunk. This prevents a
    # completely full first chunk from silently losing its successor overlap.
    base = _pack_units(prefix, sentences, encoding, limit - overlap if overlap else limit)
    results: list[str] = []
    for item in base:
        if results and overlap:
            previous = results[-1][len(prefix):]
            words = previous.split()
            context = ""
            for word in reversed(words):
                candidate = word + (" " + context if context else "")
                if token_count(candidate, encoding) > overlap:
                    break
                context = candidate
            if context and token_count(prefix + context + " " + item[len(prefix):], encoding) <= limit:
                item = prefix + context + " " + item[len(prefix):]
            elif context:
                # Leave room for overlap without trimming the new semantic unit.
                spare = limit - token_count(item, encoding)
                for amount in range(min(overlap, spare), 0, -1):
                    tail = " ".join(context.split()[-amount:])
                    candidate = prefix + tail + " " + item[len(prefix):]
                    if token_count(candidate, encoding) <= limit:
                        item = candidate
                        break
        results.append(item)
    return results


def table_chunks(html: str, description: str, section: str, encoding,
                 limit: int = MAX_TOKENS) -> list[tuple[str, int, int]]:
    """Keep named cells and complete rows; repeat description/headers on overflow."""
    headers, rows = table_rows(html)
    prefix = f"Section: {section}. Table: {clean_text(description)}. "
    if token_count(prefix, encoding) >= limit:
        raise ValueError("Table description exceeds token limit")
    chunks: list[tuple[str, int, int]] = []
    current: list[str] = []
    first = 1
    for number, row in enumerate(rows, start=1):
        pairs = [f"{name}: {value}" for name, value in zip(headers, row) if value]
        if not pairs:
            continue
        line = f"Row {number}. " + "; ".join(pairs) + "."
        if token_count(prefix + " ".join(current + [line]), encoding) <= limit:
            current.append(line)
            continue
        if current:
            chunks.append((prefix + " ".join(current), first, number - 1))
            current = []
        first = number
        if token_count(prefix + line, encoding) <= limit:
            current.append(line)
            continue
        # A wide table row can exceed the budget; repeat its row identifier and
        # column names in separate chunks without associating a value with the
        # wrong column. A single oversized cell is flagged for manual review.
        row_prefix = prefix + f"Row {number}. "
        if token_count(row_prefix, encoding) >= limit:
            raise ValueError("Table row context exceeds token limit")
        cells: list[str] = []
        for name, value in zip(headers, row):
            if not value:
                continue
            cell = f"{name}: {value}."
            if token_count(row_prefix + cell, encoding) <= limit:
                cells.append(cell)
                continue
            # Repeat the column name on *every* fragment of a long cell.
            cell_prefix = row_prefix + f"{name}: "
            if token_count(cell_prefix, encoding) >= limit:
                raise ValueError("Table column name exceeds token limit")
            part = ""
            for word in value.split():
                candidate = f"{part} {word}".strip()
                if token_count(cell_prefix + candidate + ".", encoding) > limit:
                    if not part:
                        raise ValueError("Single table-cell word exceeds token limit")
                    cells.append(f"{name}: {part}.")
                    part = word
                else:
                    part = candidate
            if part:
                cells.append(f"{name}: {part}.")
        row_parts = _pack_units(row_prefix, cells, encoding, limit)
        chunks.extend((part, number, number) for part in row_parts)
    if current:
        chunks.append((prefix + " ".join(current), first, len(rows)))
    return chunks


def stable_chunk_id(paper_id: str, page: int, kind: str, ordinal: int, text: str) -> str:
    return sha256(f"{paper_id}\0{page}\0{kind}\0{ordinal}\0{text}".encode()).hexdigest()