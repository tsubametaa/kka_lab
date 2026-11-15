"""In-memory Astra DB Document Collection and client doubles for offline unit tests."""

from copy import deepcopy
from types import SimpleNamespace


def _matches(doc, query):
    if not query:
        return True
    for key, expected in query.items():
        if key == '$or':
            if not any(_matches(doc, clause) for clause in expected):
                return False
            continue
        val = doc.get(key)
        if isinstance(expected, dict):
            if '$in' in expected and val not in expected['$in']:
                return False
            if '$ne' in expected and val == expected['$ne']:
                return False
            if '$exists' in expected and ((key in doc) != bool(expected['$exists'])):
                return False
        elif val != expected:
            return False
    return True


def _project(doc, projection):
    if not projection:
        return deepcopy(doc)
    included = {k for k, v in projection.items() if v}
    if not included:
        return deepcopy(doc)
    if '_id' not in projection or projection.get('_id', True):
        included.add('_id')
    return deepcopy({k: v for k, v in doc.items() if k in included})


class InMemoryAstraCollection:
    """Simulasi ringan astrapy.Collection (NoSQL) untuk pengujian tanpa jaringan."""

    def __init__(self):
        self._docs = []

    def insert_one(self, doc):
        doc_id = doc.get('_id')
        if any(d.get('_id') == doc_id for d in self._docs):
            raise ValueError(f'Duplicate _id: {doc_id}')
        self._docs.append(deepcopy(doc))
        return SimpleNamespace(inserted_id=doc_id)

    def find(self, filter=None, *, projection=None, sort=None, limit=None):
        matched = [
            (idx, doc)
            for idx, doc in enumerate(self._docs)
            if _matches(doc, filter)
        ]
        if sort:
            for field, direction in reversed(list(sort.items())):
                reverse = direction == -1
                matched.sort(
                    key=lambda item: (item[1].get(field, ''), item[0]),
                    reverse=reverse,
                )
        docs = [item[1] for item in matched]
        if limit is not None:
            docs = docs[:limit]
        return [_project(doc, projection) for doc in docs]

    def find_one(self, filter=None, *, projection=None, sort=None):
        rows = self.find(filter, projection=projection, sort=sort, limit=1)
        return rows[0] if rows else None

    def update_one(self, filter, update, *, upsert=False):
        set_fields = update.get('$set', {})
        for doc in self._docs:
            if _matches(doc, filter):
                doc.update(deepcopy(set_fields))
                return SimpleNamespace(matched_count=1, modified_count=1, upserted_id=None)
        if upsert:
            new_doc = {k: v for k, v in filter.items() if not isinstance(v, dict)}
            new_doc.update(deepcopy(set_fields))
            self._docs.append(new_doc)
            return SimpleNamespace(
                matched_count=0, modified_count=0, upserted_id=new_doc.get('_id')
            )
        return SimpleNamespace(matched_count=0, modified_count=0, upserted_id=None)

    def delete_one(self, filter):
        for idx, doc in enumerate(self._docs):
            if _matches(doc, filter):
                self._docs.pop(idx)
                return SimpleNamespace(deleted_count=1)
        return SimpleNamespace(deleted_count=0)

    def delete_many(self, filter):
        before = len(self._docs)
        self._docs = [doc for doc in self._docs if not _matches(doc, filter)]
        return SimpleNamespace(deleted_count=before - len(self._docs))


class FakeInsertResult:
    """Minimal insert_many result stub exposing inserted_ids."""

    def __init__(self, ids):
        self.inserted_ids = ids


class FakeDeleteResult:
    """Minimal delete_many result stub exposing deleted_count."""

    def __init__(self, count):
        self.deleted_count = count


class RecordingAstraCollection:
    """Astra collection double that records find, insert_many, and delete_many calls."""

    def __init__(self, fail_first_insert=False, delete_count=3, find_rows=None):
        self.inserted = []
        self.inserted_batches = []
        self.find_filters = []
        self.delete_filters = []
        self.fail_first_insert = fail_first_insert
        self.delete_count = delete_count
        self.find_rows = (
            find_rows
            if find_rows is not None
            else [
                {
                    '_id': 'doc_1:0000',
                    'text': 'Materi uji',
                    'source_title': 'Modul 1',
                    '$similarity': 0.92,
                }
            ]
        )

    def find(self, query_filter=None, **kwargs):
        self.find_filters.append(query_filter)
        return iter(self.find_rows)

    def insert_many(self, docs, **kwargs):
        if self.fail_first_insert:
            self.fail_first_insert = False
            raise TimeoutError('Simulated transient network timeout')
        batch = list(docs)
        self.inserted.extend(batch)
        self.inserted_batches.append(batch)
        return FakeInsertResult([d['_id'] for d in batch])

    def delete_many(self, delete_filter, **kwargs):
        self.delete_filters.append(delete_filter)
        return FakeDeleteResult(self.delete_count)


class FakeDB:
    """Minimal astrapy Database stub returning a preconfigured collection."""

    def __init__(self, collection):
        self.collection = collection

    def get_collection(self, *args, **kwargs):
        return self.collection


class FakeClient:
    """Minimal astrapy DataAPIClient stub returning FakeDB."""

    def __init__(self, collection):
        self.collection = collection

    def get_database(self, *args, **kwargs):
        return FakeDB(self.collection)
