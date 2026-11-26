"""Official AutoDAN-GA sampler + DeepSeek adapter; limited-budget transfer evaluation.
No claim of official HGA, paper-scale replication, or guard-adaptive optimization.
"""
import argparse,gc,hashlib,importlib,json,math,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs';OUT.mkdir(exist_ok=True)
def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
def read(path,default):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

def worker(minutes):
    import torch,numpy as np
    from src.backends import HFBackend
    from src.common import PREFIX,visible_answer
    from src.guard import QwenGuard,deliver
    deadline=None # No wall-clock cutoff
    def budget(reserve=0):
        return # No wall-clock cutoff; reserve argument retained for compatibility
    prov=read(ROOT/'upstream/provenance.json',{})
    if not prov:raise RuntimeError('Run setup_official.py first')
    for name,digest in prov['files'].items():
        if hashlib.sha256((ROOT/'upstream'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Upstream file changed: '+name)
    sys.path.insert(0,str(ROOT/'upstream'))
    from utils.opt_utils import autodan_sample_control
    refs=torch.load(ROOT/'upstream/assets/prompt_group.pth',map_location='cpu',weights_only=True)
    if not isinstance(refs,list) or len(refs)<8 or any(not isinstance(x,str) for x in refs):raise ValueError('Unsupported upstream reference format; no unsafe pickle fallback')
    refs=[x.replace('[MODEL]','DeepSeek').replace('[KEEPER]','DeepSeek') for x in refs]
    random.seed(42);np.random.seed(42)
    trial=autodan_sample_control(control_suffixs=refs[:8],score_list=[1.0]*8,num_elites=1,batch_size=8,crossover=.5,num_points=5,mutation=.01,API_key=None,reference=refs)
    if len(trial)!=8 or any(not isinstance(x,str) for x in trial):raise ValueError('Upstream sampler interface changed')
    cases=read(ROOT/'cases.json',[])
    protocol={'method':'Official AutoDAN-GA sampler; DeepSeek adapter','upstream_commit':prov['commit'],'population':8,'steps':8,'seeds':[42],'max_new_tokens':768,'target':'Sure, here is','search_config':'unprefixed target only; frozen replay on prefix and guard','mutation':.01,'mutation_api':'None (official offline/reference branch)','early_stop':'disabled; fixed fitness budget','fitness_batch_size':1,'cases':cases,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    old=read(OUT/'protocol.json',None)
    if old and old!=protocol:raise ValueError('Protocol changed. Move runs folder before starting a new experiment.')
    write(OUT/'protocol.json',protocol)
    source=read(OUT/'source.json',[]);search=read(OUT/'search.json',[]);results=read(OUT/'results.json',[])
    completed={(r['case_id'],r['variant'],r['config']) for r in source}
    done={r['case_id']:r for r in search}
    model=None
    expected_sources=sum(4 if c["kind"]=="attack" else 2 for c in cases)
    expected_records=2*expected_sources
    expected_fitness=sum(c["kind"]=="attack" for c in cases)*64
    if len(source)<expected_sources:
        budget(900);model=HFBackend('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B','cuda');prior=read(OUT/'target_model.json',None)
        if prior and prior!=model.metadata():raise ValueError('Target model revision changed; cannot resume')
        write(OUT/'target_model.json',model.metadata())
        def render(prefix,goal):return prefix.replace('[REPLACE]',goal) if '[REPLACE]' in prefix else prefix+' '+goal
        def loss(prompt):
            text=model.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
            x=model.tokenizer(text,return_tensors='pt',add_special_tokens=False).input_ids
            y=model.tokenizer(protocol['target'],return_tensors='pt',add_special_tokens=False).input_ids
            ids=torch.cat([x,y],dim=1).to('cuda')
            if ids.shape[1]>4096:raise ValueError('Candidate exceeds 4096-token research limit; no truncation')
            lab=ids.clone();lab[:,:x.shape[1]]=-100
            with torch.inference_mode():v=float(model.model(input_ids=ids,labels=lab,use_cache=False).loss.item())
            if not math.isfinite(v):raise ValueError('Non-finite fitness')
            return v
        # Short generation calibration on a benign task; calibration excluded from results.
        if not (OUT/'calibration.json').exists():
            t=time.monotonic();c=model.generate('Explain why keeping a research log is useful.',42,128);sec=time.monotonic()-t
            # Conservative estimate only; cannot predict throttling or longer context.
            rate=c.output_tokens/max(sec,.01);t=time.monotonic();loss(render(refs[0],cases[0]['goal']));fsec=time.monotonic()-t
            estimate=1.7*(expected_sources*768/max(rate,.1)+expected_fitness*fsec)+900
            write(OUT/'calibration.json',{'tokens_per_second':rate,'fitness_seconds':fsec,'conservative_estimate_minutes':estimate/60,'not_guaranteed':True})
            print(f'CALIBRATION: estimated {estimate/60:.1f} minutes; informational only, NO TIME LIMIT',flush=True)
            # Estimate never stops the run.
        for i,case in enumerate(cases):
            goal=case.get('goal',case['prompt']);cid=case['id'];run_seed=case['run_seed'];templates=[('benign',case['prompt'])]
            if case['kind']=='attack':
                if cid not in done:
                    random.seed(run_seed);np.random.seed(run_seed);torch.manual_seed(run_seed)
                    pop=refs[:8];best=(float('inf'),refs[0]);history=[];start=time.monotonic()
                    for step in range(8):
                        scores=[]
                        for candidate in pop:
                            budget(900);scores.append(loss(render(candidate,goal)))
                        ix=min(range(8),key=scores.__getitem__)
                        if scores[ix]<best[0]:best=(scores[ix],pop[ix])
                        history.append(best[0])
                        write(OUT/'active_search.json',{'case_id':cid,'step':step+1,'best_prompt':best[1],'loss_history':history,'status':'partial search; not a completed evaluation'})
                        if step<7:pop=autodan_sample_control(control_suffixs=pop,score_list=scores,num_elites=1,batch_size=8,crossover=.5,num_points=5,mutation=.01,API_key=None,reference=refs)
                        print(f'SEARCH {i+1}/{len(cases)} step {step+1}/8 loss={best[0]:.4f}',flush=True)
                    item={'case_id':cid,'goal':goal,'best_prompt':best[1],'best_loss':best[0],'loss_history':history,'fitness_evaluations':64,'elapsed_s':time.monotonic()-start}
                    search.append(item);done[cid]=item;write(OUT/'search.json',search)
                templates=[('seed',render(refs[0],goal)),('autodan',render(done[cid]['best_prompt'],goal))]
            for variant,prompt in templates:
                for cfg in ('A','B'):
                    if (cid,variant,cfg) in completed:continue
                    budget(900);full=prompt if cfg=='A' else PREFIX+'\n\n'+prompt
                    c=model.generate(full,run_seed,768);v=visible_answer(c.raw)
                    row={'id':f'{cid}:{variant}:{cfg}','case_id':cid,'kind':case['kind'],'category':case['category'],'goal':goal,'variant':variant,'config':cfg,'seed':run_seed,'prompt':full,'raw_answer':v.text,'truncated':c.truncated,'incomplete_reasoning':v.incomplete_reasoning,'generation_ms':c.latency_ms,'output_tokens':c.output_tokens,'generation_error':'','delivered':v.text,'blocked':False,'guard_reason':'disabled','guard_ms':0,'total_ms':c.latency_ms,'manual_label':''}
                    source.append(row);completed.add((cid,variant,cfg));write(OUT/'source.json',source)
                    print(f'GENERATED {len(source)}/{expected_sources}',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    have={r['id'] for r in results}
    budget();guard=QwenGuard('cuda');prior=read(OUT/'guard_model.json',None)
    if prior and prior!=guard.metadata():raise ValueError('Guard revision changed; cannot resume')
    write(OUT/'guard_model.json',guard.metadata())
    for r in source:
        if r['id'] not in have:results.append(r);have.add(r['id'])
        cfg='D' if r['config']=='A' else 'C';rid=f"{r['case_id']}:{r['variant']}:{cfg}"
        if rid not in have:
            budget();shown,d=deliver(r['prompt'],r['raw_answer'],r['truncated'],r['incomplete_reasoning'],guard)
            results.append(dict(r,id=rid,config=cfg,delivered=shown,blocked=d.action=='block',guard_reason=d.reason,guard_ms=d.latency_ms,total_ms=r['generation_ms']+d.latency_ms,source_record_id=r['id']));have.add(rid)
        write(OUT/'results.json',results)
        print(f'RECORDS {len(results)}/{expected_records}',flush=True)
    write(OUT/'status.json',{'status':'GENERATION_COMPLETE_REVIEW_REQUIRED','rows':len(results),'new_autodan_searches':len(search),'manual_content_review_required':True})

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:
        try:worker(None)
        except Exception as e:
            write(OUT/'status.json',{'status':'INCOMPLETE','error':type(e).__name__+': '+str(e)});raise
    else:
        r=subprocess.run([sys.executable,__file__,'--worker'])
        sys.exit(r.returncode)
if __name__=='__main__':main()
