"""Tests for app.features.rag.context (context compilation, follow-up topic, history filtering)."""

import unittest

from app.features.rag import (
    build_context,
    filter_relevant_history,
    is_follow_up_question,
    resolve_query_topic,
    retrieval_query,
)
from tests.fakes import WordTokenizer


class RagContextTests(unittest.TestCase):
    def test_build_context_strips_metadata_headers_and_truncates_at_sentence(self):
        tok = WordTokenizer()
        docs = [
            {
                '_id': 'doc-1',
                'source_title': 'Modul Prompt',
                '$similarity': 0.91,
                'pages_pdf': [3],
                'text': (
                    'Topik: Prompt Engineering | Sumber: Modul KKA\n'
                    'PKTO memandu siswa menulis prompt secara terstruktur. '
                    'Setiap elemen menentukan peran, konteks, tugas, serta bentuk output yang jelas dan akurat. '
                    'Kalimat tambahan panjang sekali yang seharusnya terpotong bersih sebelum batas akhir token terlampaui.'
                ),
            }
        ]
        ctx, sources = build_context(docs, tok, max_tokens=36)
        self.assertNotIn('[S1]', ctx)
        self.assertNotIn('Topik:', ctx)
        self.assertIn('Materi (Modul Prompt):', ctx)
        self.assertTrue(ctx.endswith('.'))
        self.assertNotIn('Kalimat tambahan', ctx)
        self.assertEqual(sources[0]['label'], 'S1')
        self.assertEqual(sources[0]['pages_pdf'], [3])

    def test_build_context_strips_page_markers_while_keeping_template_placeholders(self):
        tok = WordTokenizer()
        docs = [
            {
                '_id': 'doc-page',
                'source_title': 'Bab 1 & 7B',
                '$similarity': 0.88,
                'text': (
                    'Topik: Berpikir Komputasional\n'
                    'Solusi optimal perlu diuji. [Halaman buku 4] Gunakan [HALAMAN/SCREEN] pada template PKTO.'
                ),
            }
        ]
        ctx, _ = build_context(docs, tok, max_tokens=100)
        self.assertNotIn('[Halaman buku 4]', ctx)
        self.assertIn('[HALAMAN/SCREEN]', ctx)
        self.assertIn('Solusi optimal perlu diuji. Gunakan [HALAMAN/SCREEN]', ctx)

    def test_retrieval_query_and_topic_resolution(self):
        merged = retrieval_query('contohnya?', 'apa itu data latih?')
        self.assertEqual(merged, 'apa itu data latih? contohnya?')

        query, next_topic = resolve_query_topic('contohnya?', 'apa itu data latih?')
        self.assertEqual(query, 'apa itu data latih? contohnya?')
        self.assertEqual(next_topic, 'apa itu data latih?')

        long_q = 'bagaimana cara membagi data latih dan data validasi secara tepat?'
        self.assertEqual(retrieval_query(long_q, 'topik lama'), long_q)
        q2, t2 = resolve_query_topic(long_q, 'topik lama')
        self.assertEqual(q2, long_q)
        self.assertEqual(t2, long_q)

        self.assertEqual(retrieval_query('apa itu?', ''), 'apa itu?')

        self.assertFalse(is_follow_up_question('Apa itu SQL injection?', 'Jelaskan algoritma K-Means'))
        self.assertEqual(
            retrieval_query('Jelaskan SQL injection dong', 'Apa itu algoritma K-Means?'),
            'Jelaskan SQL injection dong',
        )
        self.assertEqual(
            retrieval_query('What is Docker?', 'Explain overfitting in ML'),
            'What is Docker?',
        )
        self.assertTrue(is_follow_up_question('How does it work?', 'Explain Docker'))
        self.assertEqual(
            retrieval_query('Contohnya di Python?', 'Apa itu algoritma K-Means?'),
            'Apa itu algoritma K-Means? Contohnya di Python?',
        )

    def test_filter_relevant_history_drops_stale_topic_on_switch(self):
        history = [
            {'role': 'user', 'content': 'Apa itu algoritma K-Means?'},
            {'role': 'assistant', 'content': 'K-Means mengelompokkan data ke dalam klaster berdasarkan jarak centroid.'},
            {'role': 'user', 'content': 'Apa itu SQL injection?'},
            {'role': 'assistant', 'content': 'SQL injection adalah celah keamanan ketika input pengguna disisipkan ke query basis data.'},
        ]
        self.assertEqual(filter_relevant_history('Apa itu Docker?', history), [])

        active = filter_relevant_history('Contohnya bagaimana cara mencegahnya?', history)
        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]['content'], 'Apa itu SQL injection?')

        related_only = filter_relevant_history('Kenapa centroid K-Means harus dihitung ulang?', history[:2])
        self.assertEqual(len(related_only), 2)

    def test_student_asking_back_about_assistant_question_preserves_history_and_context(self):
        history = [
            {'role': 'user', 'content': 'apa itu labeling data?'},
            {
                'role': 'assistant',
                'content': (
                    'Labeling data adalah proses memberi kategori kepada setiap item data. '
                    'Menurutmu, apa risiko jika seorang siswa diberi label Salah hanya karena '
                    'menggambar garis agak miring?'
                ),
            },
        ]
        reply = 'saya tidak tahu, memangnya risikonya apa?'
        active = filter_relevant_history(reply, history)
        self.assertEqual(len(active), 2)

        merged_query = retrieval_query(reply, history[0]['content'], history[1]['content'])
        self.assertIn('labeling data', merged_query.lower())
        self.assertIn('risiko', merged_query.lower())
        self.assertIn(reply, merged_query)


if __name__ == '__main__':
    unittest.main()
