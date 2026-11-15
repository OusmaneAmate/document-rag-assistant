
import unittest

from src.document_loader import (
    clean_text,
    load_corpus,
    corpus_statistics
)


class TestDocumentLoader(unittest.TestCase):

    def test_clean_text(self):
        text = "Bonjour    monde\n\n\nTest"
        result = clean_text(text)

        self.assertEqual(
            result,
            "Bonjour monde\n\nTest"
        )

    def test_corpus_structure(self):
        corpus = load_corpus()

        self.assertGreater(
            len(corpus), 0,
            "Ajoute des PDF dans data/documents/"
        )

        for document in corpus:
            self.assertIn("text", document)
            self.assertIn("metadata", document)
            self.assertIn("source", document["metadata"])
            self.assertIn("page", document["metadata"])

    def test_statistics(self):
        corpus = load_corpus()
        stats = corpus_statistics(corpus)

        self.assertEqual(
            stats["total_pages"],
            len(corpus)
        )

        self.assertEqual(
            stats["pages_with_text"] + stats["empty_pages"],
            stats["total_pages"]
        )


if __name__ == "__main__":
    unittest.main()
