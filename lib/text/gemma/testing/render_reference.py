# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent pinned-HF Gemma reference; never imports the product recognizer."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import site
import sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--template',required=True,type=Path)
    p.add_argument('--tokenizer',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    assert sys.prefix!=sys.base_prefix and not site.ENABLE_USER_SITE
    versions={n:importlib.metadata.version(n) for n in ['transformers','Jinja2','tokenizers','MarkupSafe']}
    assert versions=={'transformers':'4.57.1','Jinja2':'3.1.6','tokenizers':'0.22.2','MarkupSafe':'3.0.4'}
    import transformers,jinja2,tokenizers
    from transformers.utils.chat_template_utils import _compile_jinja_template
    for m in [transformers,jinja2,tokenizers]:
        assert Path(m.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    source=a.template.read_bytes()
    assert hashlib.sha256(source).hexdigest()=='7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4'
    tokenizer=tokenizers.Tokenizer.from_file(str(a.tokenizer))
    template=_compile_jinja_template(source.decode())
    cases=[
        ('one-user',[{'role':'user','content':'Hello'}]),
        ('system-user',[{'role':'system','content':' sys '},{'role':'user','content':' user '}]),
        ('text-parts',[{'role':'user','content':[{'type':'text','text':' A '},{'type':'text','text':' B '}]}]),
        ('system-parts',[{'role':'system','content':[{'type':'text','text':' first '},{'type':'text','text':' ignored '}]},{'role':'user','content':'U'}]),
        ('multiturn',[{'role':'user','content':'A'},{'role':'assistant','content':' B '},{'role':'user','content':'C'}]),
        ('unicode',[{'role':'user','content':'\u2003café 🙂\u00a0'}]),
        ('empty-user',[{'role':'user','content':''}]),
        ('system-only',[{'role':'system','content':'S'}]),
        ('assistant-first',[{'role':'assistant','content':'A'}]),
        ('two-users',[{'role':'user','content':'A'},{'role':'user','content':'B'}]),
        ('empty-messages',[]),
        ('empty-system-list',[{'role':'system','content':[]},{'role':'user','content':'U'}]),
        ('empty-user-list',[{'role':'user','content':[]}]),
        ('assistant-last',[{'role':'user','content':'A'},{'role':'assistant','content':'B'}]),
        ('whitespace',[{'role':'user','content':' \t\n A  B \n '}]),
        ('replay',[{'role':'user','content':'Hello'},{'role':'assistant','content':'  spaced café 🙂  '},{'role':'user','content':'Next'}]),
    ]
    rows=[]
    for name,messages in cases:
        for gen in [False,True]:
            row={'case':name,'request':{'messages':messages,'model_options':{'add_generation_prompt':gen}},'kwargs':{'add_generation_prompt':gen}}
            try: text=template.render(messages=messages,bos_token='<bos>',add_generation_prompt=gen)
            except Exception as e: row['error']=type(e).__name__+': '+str(e)
            else: row.update(text=text,token_ids=tokenizer.encode(text,add_special_tokens=False).ids)
            rows.append(row)
    assert len(rows)==32 and sum('error'in row for row in rows)==8
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    a.output.joinpath('reference.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    a.output.joinpath('identity.json').write_text(json.dumps({'python':sys.executable,'user_site':site.ENABLE_USER_SITE,'versions':versions,'imports':{m.__name__:m.__file__ for m in [transformers,jinja2,tokenizers]},'template_sha256':hashlib.sha256(source).hexdigest(),'tokenizer_sha256':hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':len(rows),'scope':'Independent HF only; no native parity claimed'},indent=2)+'\n')
    print('Gemma HF:32rows,24exactpositive,8sourceexceptions')
if __name__=='__main__':main()
