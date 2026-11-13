"""Resume strict dual-judge labeling for previously unjudged result pairs only."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime,timezone
import requests
from config import load_pipeline_config,resolve_path
from .common import read_csv,write_csv,write_json
from neural.src.common.gpu_lock import gpu_job_lock

class WindowsOllama:
    """Send sequential Ollama requests through Windows localhost."""

    SCRIPT = r'''
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
while ($null -ne ($line = [Console]::In.ReadLine())) {
  try {
    $item = ConvertFrom-Json -InputObject $line
    $uri = $item.base + $item.path
    if ($item.method -eq "GET") {
      $result = Invoke-RestMethod -Uri $uri -Method Get -TimeoutSec 300
    } else {
      $body = ConvertTo-Json -InputObject $item.payload -Depth 32 -Compress
      $result = Invoke-RestMethod -Uri $uri -Method Post -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec 300
    }
    [Console]::Out.WriteLine((ConvertTo-Json -InputObject $result -Depth 32 -Compress))
  } catch {
    [Console]::Out.WriteLine((ConvertTo-Json -InputObject @{__error=$_.Exception.Message} -Compress))
  }
}
'''

    def __init__(self, url):
        self.url = url
        executable = shutil.which("powershell.exe")
        if not executable:
            raise RuntimeError("powershell.exe is unavailable; cannot reach Windows Ollama")
        self.process = subprocess.Popen(
            [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", self.SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def request(self, method, path, payload=None):
        request = json.dumps({"method": method, "path": path, "payload": payload, "base": self.url}, ensure_ascii=True)
        self.process.stdin.write(request + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read().strip()
            raise RuntimeError(f"Windows Ollama bridge exited: {error}")
        result = json.loads(line)
        if "__error" in result:
            raise RuntimeError(result["__error"])
        return result

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)

def prompt_for(query, record):
    title = record.get("title", "")
    headings = " | ".join(record.get("headings", [])[:8])
    body = " ".join(str(record.get("body_text", "")).split())[:1800]
    code = " ".join(str(record.get("code_text", "")).split())[:1400]
    return f"""You judge search results for a technical documentation corpus.

Decide whether this document is relevant to the user's query.
Label 1 only when the shown document content materially helps answer the query, explains the requested concept, or provides applicable instructions or code. Label 0 when it only shares a technology or keyword but does not help answer the query. Treat exact identifiers, error codes, API names, and code behavior as meaningful evidence. Do not infer content that is not shown. Judge the document as a whole from its title, headings, and excerpts.

Query: {query}

Document title: {title}
Headings: {headings}
Body excerpt: {body}
Code excerpt: {code}

Return only JSON with this schema: {{"relevant": 0 or 1, "rationale": "brief evidence-based reason"}}"""


class OllamaClient:
    def __init__(self, url, timeout):
        self.url=url.rstrip('/');self.timeout=timeout;self.bridge=None
        self.session=requests.Session()
        try:
            response=self.session.get(self.url+'/api/tags',timeout=5)
            response.raise_for_status()
        except requests.RequestException:
            self.bridge=WindowsOllama(self.url)

    def request(self, method, path, payload=None):
        if self.bridge:return self.bridge.request(method,path,payload)
        response=self.session.request(method,self.url+path,json=payload,timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def close(self):
        self.session.close()
        if self.bridge:self.bridge.close()


def active_pool(config):
    queries={q['query_id']:q['query_text'] for q in (json.loads(l) for l in resolve_path(config['paths']['queries']).read_text().splitlines() if l.strip())}
    pool=set()
    for spec in config['evaluation']['result_snapshots']:
        for row in read_csv(resolve_path(config['paths']['evaluation_results_dir'])/spec['path']):
            if row['query_id'] not in queries or row['query_text']!=queries[row['query_id']]:
                raise ValueError('Results and current query texts disagree')
            pool.add((row['query_id'],row['doc_id']))
    return queries,pool


def judge_missing(config_path):
    config=load_pipeline_config(config_path);settings=config['evaluation']['judges']
    queries,pool=active_pool(config)
    query_file=resolve_path(config["paths"]["judgments"]).parent/"judgment_queries.json"
    known=json.loads(query_file.read_text()) if query_file.exists() else {}
    if any(qid in known and known[qid]!=text for qid,text in queries.items()):
        raise ValueError("A judged query text changed under its existing ID")
    known.update(queries)
    write_json(query_file,known)
    path=resolve_path(config['paths']['judgments'])
    master={(r['query_id'],r['doc_id']):r for r in read_csv(path)}
    missing=sorted(pool-master.keys())
    print(f'{len(pool)} active pairs; {len(missing)} require strict judgments.',flush=True)
    if not missing:return
    data=path.parent;decision_path=data/'judge_decisions.csv'
    fields=['query_id','doc_id','judge','model','relevant','rationale','rubric_version','prompt_sha256','updated_at']
    decisions={(r['query_id'],r['doc_id'],r['judge']):r for r in read_csv(decision_path)} if decision_path.exists() else {}
    database=sqlite3.connect(f"file:{resolve_path(config['paths']['neural_output_dir'])/'step_01_extraction/corpus.sqlite'}?mode=ro",uri=True)
    docs={did:json.loads(database.execute('SELECT record_json FROM documents WHERE doc_id=?',(did,)).fetchone()[0]) for _,did in missing};database.close()
    with gpu_job_lock():
        client=OllamaClient(settings['url'],settings['timeout_seconds'])
        try:
            installed={m['name'] for m in client.request('GET','/api/tags').get('models',[])}
            if set(settings['models'].values())-installed:
                raise RuntimeError('Configured strict judge models are not installed in Ollama')
            for name,model in settings['models'].items():
                for active in client.request('GET','/api/ps').get('models',[]):
                    active_name=active.get('name',active.get('model'))
                    if active_name not in settings['models'].values():
                        raise RuntimeError(f'Another Ollama model is loaded: {active_name}')
                    client.request('POST','/api/generate',{'model':active_name,'keep_alive':0})
                try:
                    for i,(qid,did) in enumerate(missing,1):
                        prompt=prompt_for(queries[qid],docs[did]);prompt_hash=hashlib.sha256(prompt.encode()).hexdigest();key=(qid,did,name)
                        if key in decisions and decisions[key]['prompt_sha256']==prompt_hash:
                            continue
                        for attempt in range(settings['retries']):
                            try:
                                response=client.request('POST','/api/chat',dict(model=model,messages=[{'role':'user','content':prompt}],format='json',stream=False,think=False,keep_alive='5m',options=dict(temperature=0,num_predict=settings['max_tokens'],num_thread=1)))
                                content=response['message']['content'].strip()
                                label=json.loads(content[content.index('{'):content.rindex('}')+1])
                                if type(label['relevant']) is not int or label['relevant'] not in (0,1) or not label.get('rationale'):raise ValueError('Invalid judge decision')
                                break
                            except Exception:
                                if attempt+1==settings['retries']:raise
                                time.sleep(attempt+1)
                        decisions[key]=dict(query_id=qid,doc_id=did,judge=name,model=model,relevant=label['relevant'],rationale=str(label['rationale']),rubric_version=settings['rubric_version'],prompt_sha256=prompt_hash,updated_at=datetime.now(timezone.utc).isoformat())
                        write_csv(decision_path,list(decisions.values()),fields)
                        a=decisions.get((qid,did,'nemotron'));b=decisions.get((qid,did,'gemma'))
                        if a and b and a['prompt_sha256']==prompt_hash and b['prompt_sha256']==prompt_hash:
                            first,second=int(a['relevant']),int(b['relevant']);grade=first+second
                            master[qid,did]=dict(query_id=qid,doc_id=did,j1_nemotron=first,j2_gemma=second,consensus_grade=grade,relevant=int(grade>0))
                            write_csv(path,[master[k] for k in sorted(master)],['query_id','doc_id','j1_nemotron','j2_gemma','consensus_grade','relevant'])
                        print(f'{name}: {i}/{len(missing)} new pairs',flush=True)
                finally:
                    client.request('POST','/api/generate',{'model':model,'keep_alive':0})
        finally:
            client.close()
    if pool-master.keys():raise RuntimeError('Strict judgments remain incomplete')
    write_json(resolve_path(config['paths']['evaluation_results_dir'])/'judgment_run.json',dict(rubric=settings['rubric_version'],new_pairs=len(missing),active_pairs=len(pool),master_pairs=len(master),models=settings['models']))
