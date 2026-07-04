from pathlib import Path

import pytest

from rag import retrieve
from utils.validation import InputValidationError


class Encoded:
    def tolist(self):
        return [[0.1, 0.2, 0.3]]


class FakeModel:
    def encode(self, values):
        assert len(values) == 1
        return Encoded()


class FakeCollection:
    def __init__(self):
        self.requested_k = None

    def query(self, query_embeddings, n_results):
        assert query_embeddings == [[0.1, 0.2, 0.3]]
        self.requested_k = n_results
        return {
            "ids": [["R001", "R002", "R003"]],
            "documents": [["one", "two", "three"]],
            "metadatas": [[{"tags": "a"}, {"tags": "b"}, {"tags": "c"}]],
            "distances": [[0.1, 0.2, 0.3]],
        }


def test_retrieval_shape_and_top_k(monkeypatch):
    monkeypatch.setattr(retrieve, "_get_model", lambda: FakeModel())
    collection = FakeCollection()

    results = retrieve.retrieve_rules(collection, "oily acne skin", k=2)

    assert collection.requested_k == 2
    assert len(results) == 2
    assert set(results[0]) == {"id", "document", "metadata", "distance"}
    assert results[0]["id"] == "R001"
    assert isinstance(results[0]["distance"], float)


def test_empty_query_returns_no_results(monkeypatch):
    monkeypatch.setattr(
        retrieve,
        "_get_model",
        lambda: pytest.fail("Embedding model should not be loaded"),
    )
    assert retrieve.retrieve_rules(FakeCollection(), "   ", k=5) == []


@pytest.mark.parametrize("k", [0, -1, 21, "many", True])
def test_invalid_k_is_rejected(k):
    with pytest.raises(InputValidationError):
        retrieve.retrieve_rules(FakeCollection(), "query", k=k)


def test_missing_index_has_actionable_error(tmp_path: Path):
    with pytest.raises(retrieve.IndexNotReadyError, match="bootstrap"):
        retrieve.get_collection(str(tmp_path / "missing"))


def test_malformed_chroma_result_is_rejected(monkeypatch):
    monkeypatch.setattr(retrieve, "_get_model", lambda: FakeModel())

    class BrokenCollection:
        def query(self, **kwargs):
            return {"ids": "not-a-list"}

    with pytest.raises(retrieve.RetrievalError):
        retrieve.retrieve_rules(BrokenCollection(), "query", k=3)
