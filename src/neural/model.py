"""Pinned pretrained encoder; no retrieval wrappers or downloaded Python code."""
from __future__ import annotations
from src.neural.common import MODEL_ID, MODEL_REVISION, QUERY_INSTRUCTION


def download_model(path):
    from pathlib import Path
    import json
    from src.neural.common import digest_file,write_json
    path=Path(path)
    manifest=path/'verified_files.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved.get('revision')==MODEL_REVISION and all((path/name).exists() and digest_file(path/name)==sha for name,sha in saved['files'].items()):
            return
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID,revision=MODEL_REVISION,allow_patterns=['*.json','*.txt','*.safetensors'],local_dir=path,max_workers=8)
    expected='0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd'
    if digest_file(path/'model.safetensors')!=expected:
        raise RuntimeError('Pinned model weight SHA-256 mismatch')
    files={str(p.relative_to(path)):digest_file(p) for p in path.rglob('*') if p.is_file() and '.cache' not in p.parts and p.name!='verified_files.json'}
    write_json(manifest,{'model':MODEL_ID,'revision':MODEL_REVISION,'files':files})


def tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left')


class Encoder:
    def __init__(self,path,device='cuda'):
        import torch
        from transformers import AutoModel
        self.torch = torch
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Select --device cpu or use a compatible PyTorch/CUDA installation.')
        self.device = device
        torch.set_num_threads(8)
        self.tokenizer = tokenizer(path)
        self.tokenizer.padding_side = 'right'
        self.model = AutoModel.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16 if device=='cuda' else torch.float32,attn_implementation='sdpa').to(device).eval()
        if self.model.config.model_type != 'qwen3':
            raise ValueError('This padding/pooling implementation requires the pinned causal Qwen3 model')
        self.dimension = self.model.config.hidden_size

    def encode(self,texts,max_length=512,query=False):
        torch = self.torch
        if query:
            texts = [f'Instruct: {QUERY_INSTRUCTION}\nQuery: {text}' for text in texts]
        inputs = self.tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
        if inputs['input_ids'].shape[1] > max_length:
            raise ValueError(f'Input exceeds {max_length} tokens; chunk instead of silent truncation')
        last_positions = (inputs.pop('attention_mask').sum(dim=1)-1).to(self.device)
        inputs = {key:value.to(self.device) for key,value in inputs.items()}
        with torch.inference_mode():
            hidden = self.model(**inputs,use_cache=False).last_hidden_state
            # Qwen3 is causal: real tokens cannot attend to future right-padding.
            # A padding mask is redundant for these positions; omitting it enables Flash SDPA.
            rows = torch.arange(len(texts),device=self.device)
            result = torch.nn.functional.normalize(hidden[rows,last_positions,:].float(),p=2,dim=1)
        return result.cpu().numpy()
