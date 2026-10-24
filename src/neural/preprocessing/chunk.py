"""Fixed tokenizer windows preserving every source character."""
from src.neural.common import OVERLAP


def passage_parts(text,tok,budget,overlap=OVERLAP):
    encoded = tok(text,add_special_tokens=False,return_offsets_mapping=True)
    offsets = encoded['offset_mapping']
    if not offsets:
        return
    start = 0
    while start < len(offsets):
        end = min(start+budget,len(offsets))
        char_start = 0 if start==0 else offsets[start][0]
        char_end = len(text) if end==len(offsets) else offsets[end][0]
        # Tokenizers may map multiple byte tokens to the same Unicode character.
        while end < len(offsets) and offsets[end][0] <= char_start:
            end += 1
            char_end = len(text) if end==len(offsets) else offsets[end][0]
        if char_end > char_start:
            yield text[char_start:char_end],char_start,char_end
        if end == len(offsets):
            break
        start = max(start+1,end-min(overlap,budget//4))
