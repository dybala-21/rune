"""Prevent reuse of vectors from another model or preprocessing version."""

import pytest

from rune.conversation.store import ConversationStore
from rune.llm.embedding_models import EmbeddingVector
from rune.memory.types import VectorMetadata
from rune.memory.vector import VectorStore


def test_transcript_cache_separates_same_content_by_model(tmp_path):
    store = ConversationStore(tmp_path / 'conversations.db')
    store.cache_embeddings({'text': [1.0, 0.0]})
    store.cache_embeddings({'text': [0.0, 1.0]}, fingerprint='a' * 64)
    assert not store.get_cached_embeddings(['text'], fingerprint='b' * 64)
    assert store.get_cached_embeddings(['text'], fingerprint='a' * 64)['text'] == pytest.approx([0, 1])
    assert store.get_cached_embeddings(['text'])['text'] == pytest.approx([1, 0])
    store._conn.close()


def test_vector_spaces_are_separate_and_legacy_files_preserved(tmp_path):
    pytest.importorskip('faiss')
    legacy = tmp_path / 'index.faiss'
    legacy.write_bytes(b'legacy index')
    store = VectorStore(index_path=tmp_path, require_identity=True)
    a = EmbeddingVector([1.0, 0.0], 'a' * 64)
    b = EmbeddingVector([0.0, 1.0], 'b' * 64)
    store.add(a, VectorMetadata(type='episode', id='A'))
    store.save()
    assert store.search(b) == []
    store.add(b, VectorMetadata(type='episode', id='B'))
    store.save()
    assert [r.id for r in store.search(a)] == ['A']
    assert [r.id for r in store.search(b)] == ['B']
    assert legacy.read_bytes() == b'legacy index'
    with pytest.raises(ValueError, match='provenance'):
        store.search([1.0, 0.0])


def test_index_upsert_after_reload_replaces_old_entry(tmp_path):
    pytest.importorskip('faiss')
    v = EmbeddingVector([1.0, 0.0], 'a' * 64)
    store = VectorStore(index_path=tmp_path, require_identity=True)
    store.add(v, VectorMetadata(type='episode', id='A', summary='before'))
    store.save()
    reloaded = VectorStore(index_path=tmp_path, require_identity=True)
    reloaded.upsert(v, VectorMetadata(type='episode', id='A', summary='after'))
    assert [r.text for r in reloaded.search(v)] == ['after']
    reloaded.upsert(v, VectorMetadata(type='episode', id='A', summary='latest'))
    assert [r.text for r in reloaded.search(v)] == ['latest']


async def test_indexer_rebuilds_missing_cache_and_deletes_after_reload(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import rune.memory.markdown_indexer as indexer

    pytest.importorskip('faiss')
    monkeypatch.setenv('RUNE_HOME', str(tmp_path / 'home'))
    chunks = [{'id': key, 'text': key, 'hash': key, 'type': 'md_fact'} for key in ('one', 'two')]
    active = ['a' * 64]

    async def embed(text):
        return EmbeddingVector([1.0, 0.0], active[0])

    monkeypatch.setattr(indexer, 'collect_all_chunks', lambda: list(chunks))
    monkeypatch.setattr('rune.llm.local_embedding.get_embedding_provider',
                        lambda: SimpleNamespace(embed_single=embed))
    path = tmp_path / 'vectors'

    def reload():
        return VectorStore(index_path=path, require_identity=True)

    assert (await indexer.incremental_reindex(reload()))['added'] == 2
    assert (await indexer.incremental_reindex(reload()))['unchanged'] == 2
    chunks.pop()
    assert (await indexer.incremental_reindex(reload()))['removed'] == 1
    assert [r.id for r in reload().search(await embed('query'))] == ['one']
    (path / active[0] / 'index.faiss').write_bytes(b'corrupted cache')
    assert (await indexer.incremental_reindex(reload()))['updated'] == 1
    active[0] = 'b' * 64
    assert (await indexer.incremental_reindex(reload()))['added'] == 1
    active[0] = 'a' * 64
    assert (await indexer.incremental_reindex(reload()))['added'] == 1
    chunks.clear()
    assert (await indexer.incremental_reindex(reload()))['removed'] == 1
    assert reload().search(await embed('query')) == []
