"""CLI runner for testing Astra DB RAG retrieval."""

import argparse
import json
from pathlib import Path

from app.features.rag.exceptions import RagUnavailableError
from app.features.rag.retriever import retrieve

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Astra DB RAG retrieval utility')
    parser.add_argument('--question', default='Apa perbedaan data latih dan data uji?')
    parser.add_argument('--user', default='', help='ID siswa untuk menyertakan dokumen privatnya')
    parser.add_argument('--save', default='')
    args = parser.parse_args()
    try:
        docs = retrieve(args.question, user_id=args.user or None)
    except (ValueError, RagUnavailableError) as err:
        raise SystemExit(f'Gagal mengambil sumber RAG: {err}')
    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({'question': args.question, 'docs': docs}, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8',
        )
    print('Jumlah hasil:', len(docs))
    for d in docs:
        print('\nID:', d.get('_id'), 'Skor:', d.get('$similarity'))
        print('Sumber:', d.get('source_title') or d.get('metadata', {}).get('source_title'))
        print(str(d.get('text', ''))[:650])