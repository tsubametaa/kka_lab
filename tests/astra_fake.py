"""In-memory Astra DB Document Collection double for offline unit tests."""

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

