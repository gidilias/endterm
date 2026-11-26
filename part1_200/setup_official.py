"""Fetch upstream AutoDAN, pin its commit, preserve license; no substitute algorithm."""
import json,urllib.request,zipfile,io,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'BreakPoint-research'})
    with urllib.request.urlopen(req,timeout=90) as r:return r.read()

def main():
    dest=ROOT/'upstream'
    if (dest/'provenance.json').exists():
        print('Using pinned upstream:',json.loads((dest/'provenance.json').read_text())['commit']);return
    info=json.loads(get('https://api.github.com/repos/SheltonLiu-N/AutoDAN/commits/main'));sha=info['sha']
    raw=get(f'https://codeload.github.com/SheltonLiu-N/AutoDAN/zip/{sha}')
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        prefix=z.namelist()[0].split('/')[0]+'/'
        for name in z.namelist():
            rel=name[len(prefix):]
            if not rel or name.endswith('/'):continue
            target=(dest/rel).resolve()
            if not target.is_relative_to(dest.resolve()):raise ValueError('Invalid archive path')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    required=['utils/opt_utils.py','assets/prompt_group.pth','LICENSE']
    for name in required:
        if not (dest/name).exists():raise RuntimeError('Upstream layout changed: '+name)
    (dest/'provenance.json').write_text(json.dumps({'repository':'https://github.com/SheltonLiu-N/AutoDAN','commit':sha,'zip_sha256':hashlib.sha256(raw).hexdigest(),'files':{n:hashlib.sha256((dest/n).read_bytes()).hexdigest() for n in required}},indent=2))
    print('Pinned official AutoDAN:',sha)
if __name__=='__main__':main()
