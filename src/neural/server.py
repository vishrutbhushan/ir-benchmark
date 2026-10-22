"""Local-only search UI/API; pagination reuses cached ranked results."""
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs,urlsplit


def serve(engine,host='127.0.0.1',port=8000):
    print('Preparing the search model and index...',flush=True)
    warmup_ms=engine.warmup()
    print(f'Search warmup complete: {warmup_ms:.0f} ms',flush=True)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed=urlsplit(self.path)
            try:
                if parsed.path=='/':
                    data=Path(__file__).with_name('web').joinpath('index.html').read_bytes()
                    content_type='text/html; charset=utf-8'
                elif parsed.path=='/api/options':
                    data=json.dumps({'documents':len(engine.documents),'passages':len(engine.matrix),'technologies':engine.technology_options,'categories':engine.category_options,'index_version':engine.summary['fingerprint'],'startup_warmup_ms':engine.startup_warmup_ms}).encode()
                    content_type='application/json'
                elif parsed.path=='/api/search':
                    values=parse_qs(parsed.query)
                    def get(name,default=''):return values.get(name,[default])[0]
                    page=max(1,int(get('page','1')))
                    if page>100:
                        raise ValueError('Only the first 100 pages are available')
                    result=engine.search(get('q'),limit=1000,technology=get('technology'),category=get('category'),offset=(page-1)*10,page_size=10)
                    result={**result,'page':page,'page_size':10}
                    data=json.dumps(result,ensure_ascii=False).encode()
                    content_type='application/json; charset=utf-8'
                else:
                    self.send_error(404);return
                self.send_response(200)
                self.send_header('Content-Type',content_type)
                self.send_header('Content-Length',str(len(data)))
                self.send_header('X-Content-Type-Options','nosniff')
                self.end_headers();self.wfile.write(data)
            except (ValueError,KeyError) as error:
                data=json.dumps({'error':str(error)}).encode()
                self.send_response(400);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=HTTPServer((host,port),Handler)
    print(f'Technical search: http://{host}:{port} ({len(engine.documents):,} documents)',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        engine.close()
