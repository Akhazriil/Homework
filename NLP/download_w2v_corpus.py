"""
Скачивание готовых корпусов Википедии для обучения Word2Vec (эксперимент 2 в lab2.ipynb).

Использование (из папки NLP):
    python download_w2v_corpus.py            # ru + pl, по 5 МБ символов
    python download_w2v_corpus.py pl 10000000
    python download_w2v_corpus.py ru 5000000

Результат: resources/wiki_ru_full.txt, resources/wiki_pl_full.txt
(статьи разделены пустыми строками). Скачивание возобновляемое.

Источник: готовый датасет wikimedia/wikipedia через HuggingFace datasets-server API
(100 статей за один запрос, ~0.8 МБ/запрос, без установки extra-библиотек).
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = 'https://datasets-server.huggingface.co/rows'
CONFIG = {'ru': '20231101.ru', 'pl': '20231101.pl'}
TARGET_DEFAULT = 5_000_000
BATCH = 100          # строк за запрос (максимум API)


def fetch_rows(lang, offset):
    q = urllib.parse.urlencode({
        'dataset': 'wikimedia/wikipedia', 'config': CONFIG[lang],
        'split': 'train', 'offset': offset, 'length': BATCH})
    for attempt in range(5):
        try:
            req = urllib.request.Request(API + '?' + q,
                                         headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r).get('rows', [])
        except Exception as e:
            print(f'  попытка {attempt+1}: {e}', flush=True)
            time.sleep(2 * (attempt + 1))
    return None


def download(lang, out_path, target_chars):
    total = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    offset = (total // 8000) * BATCH if total else 0   # грубая оценка для resume
    mode = 'a' if total else 'w'
    print(f'{lang}: в файле {total:,} байт, старт с offset={offset}', flush=True)

    with open(out_path, mode, encoding='utf-8', newline='\n') as f:
        while total < target_chars:
            rows = fetch_rows(lang, offset)
            if not rows:
                print('сервер не вернул строки, стоп', flush=True)
                break
            parts = [r['row']['title'] + '\n' + r['row']['text'] for r in rows]
            block = '\n\n'.join(parts)
            if total:
                f.write('\n\n')
            f.write(block)
            f.flush()
            total += len(block) + 2
            offset += len(rows)
            print(f'  {lang}: {total:,}/{target_chars:,} символов', flush=True)

    print(f'{lang}: готово ({total:,} символов) -> {out_path}', flush=True)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:]]
    if not args or args[0] == 'all':
        langs = ['ru', 'pl']
        args = args[1:]
    else:
        langs = [args[0]]
        args = args[1:]
    target = int(args[0]) if args else TARGET_DEFAULT
    os.makedirs('resources', exist_ok=True)
    for lang in langs:
        download(lang, f'resources/wiki_{lang}_full.txt', target)
