import unittest

from research_assistant.chunking import (
    Element, MAX_TOKENS, OVERLAP_TOKENS, prose_chunks, table_chunks,
    table_rows, token_count, tokenizer,
)
from research_assistant.ingest import elements_from_fast_page, needs_docling, records_for_page


class StructuredChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.encoding = tokenizer()
        except ImportError:
            raise unittest.SkipTest("Install the ingest extra to run token-boundary tests")

    def test_prose_chunks_respect_limit_and_retain_overlap(self) -> None:
        text = " ".join(f"Sentence {number} describes experimental findings." for number in range(90))
        parts = prose_chunks(text, "Findings", self.encoding)
        self.assertGreater(len(parts), 2)
        self.assertTrue(all(token_count(part, self.encoding) <= MAX_TOKENS for part in parts))
        for earlier, later in zip(parts, parts[1:]):
            left = earlier.removeprefix("Section: Findings. ")
            right = later.removeprefix("Section: Findings. ")
            self.assertTrue(any(
                right.startswith(" ".join(left.split()[-length:]))
                for length in range(1, 10)
            ), "Prose chunks must overlap at word boundaries")

    def test_table_repeats_description_and_unique_column_names(self) -> None:
        rows = "".join(f"<tr><td>Model {i}</td><td>{i}%</td></tr>" for i in range(1, 81))
        html = "<table><tr><th>Model</th><th>Accuracy</th></tr>" + rows + "</table>"
        parts = table_chunks(html, "Benchmark results", "Evaluation", self.encoding)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(token_count(text, self.encoding) <= 256 for text, _, _ in parts))
        self.assertTrue(all(text.startswith("Section: Evaluation. Table: Benchmark results.") for text, _, _ in parts))
        self.assertEqual(parts[0][1], 1)
        self.assertEqual(parts[-1][2], 80)
        self.assertIn("Model: Model 1; Accuracy: 1%", parts[0][0])
        self.assertIn("Model: Model 80; Accuracy: 80%", parts[-1][0])

    def test_colspan_headers_are_attached_to_the_correct_values(self) -> None:
        html = ("<table><tr><th rowspan='2'>Model</th><th colspan='2'>Dataset A</th>"
                "</tr><tr><th>Accuracy</th><th>Loss</th></tr>"
                "<tr><td>Small</td><td>91%</td><td>0.1</td></tr></table>")
        self.assertEqual(table_rows(html),
                         (["Model", "Dataset A / Accuracy", "Dataset A / Loss"],
                          [["Small", "91%", "0.1"]]))

    def test_long_table_cell_keeps_name_on_each_fragment(self) -> None:
        html = ("<table><tr><th>Property</th><th>Description</th></tr>"
                "<tr><td>Run 1</td><td>" + " ".join(["verylongdescription"] * 400) +
                "</td></tr></table>")
        parts = table_chunks(html, "Diagnostic", "Results", self.encoding)
        self.assertGreater(len(parts), 2)
        self.assertTrue(all(token_count(text, self.encoding) <= MAX_TOKENS for text, _, _ in parts))
        self.assertTrue(all("Row 1." in text and
                            ("Property: Run 1" in text or "Description: " in text)
                            for text, _, _ in parts))

    def test_figure_and_formula_are_not_indexed_as_verified_evidence(self) -> None:
        source = {"arxiv_id": "1234.56789v1", "title": "Test", "source_url": "https://arxiv.org/abs/1234.56789v1",
                  "file": "pdfs/test.pdf", "sha256": "abc", "published": "2026-01-01",
                  "updated": "2026-01-02", "categories": ["cs.AI"]}
        elements = [Element("section-header", "Results", 3), Element("text", "The model improved.", 3, 1),
                    Element("figure", "", 3, 2), Element("caption", "Figure 1: a graph.", 3, 3),
                    Element("formula", "", 3, 4)]
        records, section, section_id, warnings = records_for_page(elements, source, "", self.encoding, "fast")
        self.assertEqual(section, "Results")
        self.assertEqual(section_id, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["page_start"], 3)
        self.assertIn("image_not_interpreted", warnings)
        self.assertIn("figure_caption_requires_review", warnings)
        self.assertIn("formula_requires_verification", warnings)

    def test_low_text_or_bad_tables_require_escalation(self) -> None:
        self.assertTrue(needs_docling([Element("text", "", 1)]))
        self.assertTrue(needs_docling([Element("text", "enough prose " * 20, 1), Element("table", "broken", 1)]))
        self.assertFalse(needs_docling([Element("table", "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>", 1)]))
        sample = {"text": "Section title", "metadata": {"page_number": 4},
                  "page_boxes": [{"class": "section-header", "pos": (0, 13),
                                  "bbox": (0, 0, 10, 10), "index": 0}]}
        self.assertEqual(elements_from_fast_page(sample)[0].page, 4)


if __name__ == "__main__":
    unittest.main()