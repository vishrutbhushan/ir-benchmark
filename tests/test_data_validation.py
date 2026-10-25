import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.step_02_validation.archive import initialize_worker, validate_document
from src.step_02_validation.inspection import inspect_html
from src.step_02_validation.paths import local_html_path, normalize_corpus_path


class CorpusValidationTests(unittest.TestCase):
    def test_portable_paths_reject_traversal(self):
        self.assertEqual(normalize_corpus_path(r'documents\html\Python\page.html'),
                         'documents/html/Python/page.html')
        for path in ['../page.html', '/documents/html/Python/page.html',
                     r'C:\documents\html\Python\page.html',
                     'documents/html/../../page.html']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                normalize_corpus_path(path)

    def test_troubleshooting_article_is_not_an_error_page(self):
        raw = ('<html lang="en"><title>Handling 404 errors in an API</title><main><h1>Error handling</h1>'
               + '<p>Configure error handling in the application server.</p>' * 10
               + '</main></html>').encode()
        result = inspect_html(raw, {'content_languages': 'eng'})
        self.assertEqual(result['status'], 'eligible')
        self.assertNotIn('error_or_challenge_page', result['reasons'])

    def test_error_page_and_json_are_excluded(self):
        result = inspect_html(b'<html><title>404 - Page not found</title><body>Page not found</body></html>', {})
        self.assertIn('error_or_challenge_page', result['reasons'])
        self.assertIn('not_html_markup', inspect_html(b'{"message":"hello"}', {})['reasons'])

    def test_preformatted_rfc_is_prose_not_non_html_or_code(self):
        raw = ('<pre>Request for Comments: 5594\n' +
               'This document describes a network protocol and its security requirements.\n' * 20 + '</pre>').encode()
        result = inspect_html(raw, {'technology': 'RFCs', 'content_languages': 'eng'})
        self.assertEqual(result['status'], 'eligible')
        self.assertEqual(result['code_chars'], 0)
        self.assertGreater(result['prose_chars'], 1000)

    def test_long_utf8_filename_has_stable_portable_alias(self):
        original = 'documents/html/AWS/' + '語' * 100 + '.html'
        aliased = local_html_path(original)
        self.assertEqual(aliased, local_html_path(original))
        self.assertLessEqual(len(Path(aliased).name.encode()), 255)

    def test_verified_materialization_and_mismatch_protection(self):
        raw = b'<html lang="en"><title>Example</title><body><pre>return value\n  next</pre></body></html>'
        record = {'doc_id': 'one', 'html_path': r'documents\html\Python\page.html',
                  'html_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                  'content_languages': 'eng'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root/'input.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('technical_web_corpus/documents/html/Python/page.html', raw)
            initialize_worker(str(archive), str(root/'corpus'))
            result, normalized = validate_document(record)
            self.assertTrue(result['integrity_ok'])
            output = root/'corpus'/normalized['html_path']
            self.assertEqual(output.read_bytes(), raw)
            self.assertNotIn('\\', normalized['html_path'])
            result, normalized = validate_document({**record, 'sha256': '0'*64})
            self.assertFalse(result['integrity_ok'])
            self.assertIsNone(normalized)
            output.write_bytes(b'pre-existing different content')
            result, normalized = validate_document(record)
            self.assertFalse(result['integrity_ok'])
            self.assertEqual(output.read_bytes(), b'pre-existing different content')


class CorrectedDatasetTests(unittest.TestCase):
    def test_fix_replaces_repairs_prunes_files_and_preserves_backup(self):
        import json
        import sqlite3
        from unittest.mock import patch
        from src.step_04_filtering.fix import fix_dataset, validate_dataset
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, extraction, backup = base/'corpus', base/'extraction', base/'backup'
            (root/'metadata').mkdir(parents=True)
            extraction.mkdir()
            original = []
            for doc_id, raw in [('good', b'good'), ('repair', b'bad'), ('exclude', b'error')]:
                relative = f'documents/html/test/{doc_id}.html'
                path = root/relative; path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(raw)
                original.append(dict(doc_id=doc_id, html_path=relative, filename=path.name,
                                     html_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                     category='test', technology='test'))
            manifest = root/'metadata/corpus_manifest.jsonl'
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in original))
            repaired_path = root/'documents/html/repairs/repair.html'
            repaired_path.parent.mkdir();repaired_path.write_bytes(b'fixed')
            repaired = {**original[1], 'html_path':repaired_path.relative_to(root).as_posix(),
                        'html_bytes':5, 'sha256':hashlib.sha256(b'fixed').hexdigest()}
            accepted = [original[0], repaired]
            (extraction/'eligible_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in accepted))
            (extraction/'summary.json').write_text(json.dumps({'fingerprint':'frozen-text-inputs'}))
            db = sqlite3.connect(extraction/'corpus.sqlite')
            db.execute('CREATE TABLE documents (doc_index INTEGER PRIMARY KEY,doc_id TEXT,record_json TEXT)')
            for i,row in enumerate(accepted):
                db.execute('INSERT INTO documents VALUES (?,?,?)',(i,row['doc_id'],json.dumps({'metadata':row,'body_text':'unchanged'})))
            db.commit();db.close()
            with patch('src.step_04_filtering.fix.report'):
                stats = fix_dataset(root,extraction,backup,2)
            self.assertEqual(stats['documents'],2)
            self.assertEqual((root/original[1]['html_path']).read_bytes(),b'fixed')
            self.assertEqual((backup/original[1]['html_path']).read_bytes(),b'bad')
            self.assertTrue((backup/original[2]['html_path']).exists())
            self.assertFalse((root/original[2]['html_path']).exists())
            self.assertFalse((extraction/'eligible_manifest.jsonl').exists())
            self.assertEqual(json.loads((extraction/'summary.json').read_text())['fingerprint'],'frozen-text-inputs')
            self.assertEqual(fix_dataset(root,extraction,backup,2),stats)
            self.assertEqual(validate_dataset(root,2)['documents'],2)
            (root/original[0]['html_path']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'differs from its manifest'):
                validate_dataset(root,2)

    def test_validation_rejects_unlisted_html(self):
        import json
        from src.step_04_filtering.fix import validate_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'metadata').mkdir()
            path=root/'documents/html/test/extra.html';path.parent.mkdir(parents=True);path.write_bytes(b'extra')
            (root/'metadata/corpus_manifest.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'coverage mismatch'):
                validate_dataset(root,2)


    def test_corrected_corpus_extracts_without_original_audit_or_zip(self):
        import json
        from unittest.mock import patch
        from src.step_04_filtering.filter import run_extraction
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'corpus';metadata=root/'metadata';metadata.mkdir(parents=True)
            raw=b'<title>API</title><main><h1>Usage</h1><p>'+b'Preserve technical words and negation in this API reference. '*10+b'</p></main>'
            path=root/'documents/html/test/api.html';path.parent.mkdir(parents=True);path.write_bytes(raw)
            row=dict(doc_id='one',html_path=path.relative_to(root).as_posix(),title='API',url='https://example.com/api',
                     html_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),technology='test',category='test')
            (metadata/'corpus_manifest.jsonl').write_text(json.dumps(row)+'\n')
            (metadata/'corpus_state.json').write_text('{}')
            with patch('src.step_04_filtering.filter.report'):
                stats=run_extraction(root,base/'missing-original-audit',base/'extraction',{},1)
            self.assertEqual(stats['retained_documents'],1)
            self.assertFalse((base/'extraction/eligible_manifest.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
