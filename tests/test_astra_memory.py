import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from app.features.memory.astra_store import AstraChatStore
from app.features.memory.store import ChatStore
from tests.astra_fake import InMemoryAstraCollection


class AstraMemoryTests(unittest.TestCase):
    def setUp(self):
        self.mock_collection = MagicMock()
        self.store = AstraChatStore(collection_name='test_chat', keyspace='rag')
        self.store._collection = self.mock_collection

    def test_create_session(self):
        session_id = self.store.create_session('student_1')
        self.assertTrue(len(session_id) > 10)
        self.mock_collection.insert_one.assert_called_once()
        inserted_doc = self.mock_collection.insert_one.call_args[0][0]
        self.assertEqual(inserted_doc['type'], 'session')
        self.assertEqual(inserted_doc['user_id'], 'student_1')
        self.assertEqual(inserted_doc['session_id'], session_id)

    def test_session_belongs_to(self):
        self.mock_collection.find_one.return_value = {'_id': 'sess_1', 'user_id': 'student_1'}
        self.assertTrue(self.store.session_belongs_to('s1', 'student_1'))

        self.mock_collection.find_one.return_value = None
        self.assertFalse(self.store.session_belongs_to('s1', 'student_2'))

    def test_history_ordering(self):
        self.mock_collection.find.return_value = [
            {'question': 'q1', 'answer': 'a1', 'timestamp': 100},
            {'question': 'q2', 'answer': 'a2', 'timestamp': 200},
        ]
        history = self.store.history('sess_1', limit=2)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]['content'], 'q1')
        self.assertEqual(history[1]['content'], 'a1')
        self.assertEqual(history[2]['content'], 'q2')
        self.assertEqual(history[3]['content'], 'a2')

    def test_save_exchange(self):
        res = self.store.save_exchange(
            user_id='student_1',
            session_id='sess_1',
            question='Apa itu dekomposisi komputasional?',
            answer='Dekomposisi adalah teknik memecah masalah besar menjadi bagian kecil.',
            mode='terstruktur',
            sources=[],
            training_consent=False,
        )
        self.mock_collection.insert_one.assert_called_once()
        doc = self.mock_collection.insert_one.call_args[0][0]
        self.assertEqual(doc['type'], 'exchange')
        self.assertEqual(doc['user_id'], 'student_1')
        self.assertEqual(doc['session_id'], 'sess_1')
        self.assertEqual(doc['mode'], 'terstruktur')

    def test_save_knowledge_and_persona_use_upsert(self):
        self.store.save_knowledge('student_1', 'Apa itu loop?', 'Perulangan.')
        self.mock_collection.update_one.assert_called_once()
        filt, update = self.mock_collection.update_one.call_args[0]
        kwargs = self.mock_collection.update_one.call_args[1]
        self.assertTrue(filt['_id'].startswith('know_student_1_'))
        self.assertEqual(update['$set']['type'], 'knowledge')
        self.assertNotIn('_id', update['$set'])
        self.assertTrue(kwargs['upsert'])

        self.mock_collection.update_one.reset_mock()
        self.store.save_persona('student_1', 'Suka contoh game')
        filt, update = self.mock_collection.update_one.call_args[0]
        self.assertEqual(filt, {'_id': 'persona_student_1'})
        self.assertEqual(update['$set']['persona'], 'Suka contoh game')
        self.assertTrue(self.mock_collection.update_one.call_args[1]['upsert'])

    def test_chat_store_pure_astradb_rag_and_user_delete(self):
        store = ChatStore(collection=InMemoryAstraCollection())
        self.assertEqual(store.backend, 'astradb')

        user_job = store.create_ingestion_job(
            user_id='u1', filename='notes.pdf', source_title='Notes', scope='user',
        )
        global_job = store.create_ingestion_job(
            user_id='u1', filename='modul.pdf', source_title='Modul', scope='global',
        )
        store.update_ingestion_job(user_job['job_id'], status='completed', chunks_inserted=3)
        self.assertEqual(store.get_ingestion_job(user_job['job_id'])['status'], 'completed')

        store.save_persona('u1', 'Suka ringkas')
        store.delete_user_data('u1')
        self.assertEqual(store.get_persona('u1'), '')
        self.assertIsNone(store.get_document_record(user_job['doc_id']))
        self.assertIsNotNone(store.get_document_record(global_job['doc_id']))

    def test_collection_lazy_init_is_thread_safe(self):
        store = AstraChatStore(collection_name='test_chat', keyspace='rag')
        fake_col = MagicMock()
        call_count = 0

        def slow_get_collection(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            time.sleep(0.05)
            return fake_col

        results = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(store.collection)

        with patch('app.features.memory.astra_store.get_astra_chat_collection', side_effect=slow_get_collection):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(call_count, 1)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(col is fake_col for col in results))


if __name__ == '__main__':
    unittest.main()

