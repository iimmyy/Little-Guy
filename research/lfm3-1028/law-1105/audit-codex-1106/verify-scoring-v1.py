#!/usr/bin/env python3
"""Read-only source audit. Writes receipts and redirected scorer output only here."""
import ast, builtins, collections, contextlib, datetime, hashlib, importlib.util, io, json, math, os, runpy, statistics, sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LAW = HERE.parent
SCI = LAW.parent / 'science'
sys.path.insert(0, str(SCI/'grades-v4'))
from keyed_grade import keyed_grade, key_text
from sci_grade2 import grade

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jl(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def count(rows, field): return dict(collections.Counter(r[field] for r in rows))
def tally(rows, field):
 c=collections.Counter(r[field] for r in rows)
 return {k:c[k] for k in ('right','wrong','unclear','cut')}
def machine(r):
 g=keyed_grade(r.get('finish'),r.get('reply') or '',r['id'],r['answer'],r.get('scale'))
 return 'wrong' if g=='sign' else g

manifest=json.loads((HERE/'manifest-v1.json').read_text())
verdicts={r['item']:r for r in jl(HERE/'verdicts-v1.jsonl')}
amap=json.loads((HERE/'.map-v1.json').read_text())
full={r['item']:r for r in jl(HERE/'full-replies.jsonl')}
lock=(HERE/'verdicts-v1.sha256').read_text().strip()
assert sha(HERE/'verdicts-v1.jsonl')==lock
assert len(verdicts)==len(amap)==len(full)==759
assert set(verdicts)==set(amap)==set(full)
item_of={(r['arm'],r['run'],r['id']):item for item,r in amap.items()}
assert len(item_of)==759
held=jl(LAW/'mix1/sci-held-81.jsonl')
ids=[r['id'] for r in held]
assert len(ids)==len(set(ids))==81
hard=set(json.loads((LAW/'reach1/f5-wrong-twice-ids-v1.json').read_text()))
assert len(hard)==13 and hard<=set(ids)
files={
 'reach':{u:LAW/f'reach1/reach-{u}.jsonl' for u in ('g','s1')},
 'f5room':{u:LAW/f'f5room1/f5room-{u}.jsonl' for u in ('g','s1','s2','s3')},
 'c5room':{u:LAW/f'f5room1/c5room-{u}.jsonl' for u in ('g','s1')},
 'scimix':{'g':LAW/'mix1/eval-scimix.jsonl','s1':LAW/'mix1/eval-scimix-s1.jsonl'},
 'scic5':{'g':LAW/'mix1/eval-scic5.jsonl','s1':LAW/'mix1/eval-scic5-s1.jsonl'},
}
scorers=[HERE/'audit_room.py',LAW/'reach1/reach_score.py',LAW/'f5room1/f5room_score.py',LAW/'mix1/scimix_score.py',SCI/'grades-v4/keyed_grade.py',SCI/'grades-v4/sci_grade2.py',SCI/'grades-v4/keys-v2.json']
baseline_paths=[SCI/f'final-f5-1104/f5-{u}.jsonl' for u in ('g','s1','s2','s3')]+[SCI/f'final-lfm3-1104/lfm3-base-{u}.jsonl' for u in ('g','s1')]
goals=[LAW/'reach1/GOAL-reach-v1.md',LAW/'f5room1/GOAL-f5room-v1.md',LAW/'mix1/GOAL-sci-v1.md']
tracked=set(Path(p) for p in manifest['inputs'])|set(scorers)|set(baseline_paths)|set(goals)|{LAW/'reach1/f5-wrong-twice-ids-v1.json'}
before={str(p):sha(p) for p in sorted(tracked)}
assert all(sha(p)==h for p,h in manifest['inputs'].items())
assert sha(HERE/'.map-v1.json')==manifest['map_sha256']
assert sha(HERE/'audit_room.py')==manifest['tool_sha256']
for r in json.loads((HERE/'review-input-hashes-v1.json').read_text()): assert sha(HERE/r['file'])==r['sha256']

A=[]
for arm, runs in files.items():
 for run,path in runs.items():
  raw=jl(path)
  rawids=[r['id'] for r in raw]
  assert len(rawids)==len(set(rawids)),(arm,run,'duplicate raw IDs')
  assert set(ids)<=set(rawids)
  byid={r['id']:r for r in raw}
  for pid in ids:
   r=byid[pid]
   item=item_of.get((arm,run,pid))
   assert (item is not None)==(r.get('finish')=='stop')
   if item:
    assert full[item]['reply']==(r.get('reply') or '')
    assert full[item]['thinking']==(r.get('thinking') or '')
    hand=verdicts[item]['verdict']
    flag=verdicts[item]['flag']
   else: hand,flag='cut',''
   A.append(dict(arm=arm,run=run,id=pid,item=item,machine=machine(r),hand=hand,flag=flag,finish=r.get('finish'),tokens=(r.get('think_tokens') or 0)+(r.get('reply_tokens') or 0),thinking=r.get('think_tokens') or 0,reply=r.get('reply_tokens') or 0,reply_empty=not(r.get('reply') or '').strip()))
assert len(A)==972 and sum(r['item'] is None for r in A)==213

summary={}
for arm in files:
 rows=[r for r in A if r['arm']==arm]
 summary[arm]={'n':len(rows),'tokens':sum(r['tokens'] for r in rows),'machine':tally(rows,'machine'),'hand':tally(rows,'hand'),'by_run':{},'bad_reasoning_right':sum(r['hand']=='right' and r['flag']=='bad_reasoning' for r in rows),'right_flags':dict(collections.Counter(r['flag'] for r in rows if r['hand']=='right')),'disagreements':dict(collections.Counter(r['machine']+'->'+r['hand'] for r in rows if r['machine']!=r['hand']))}
 for run in files[arm]:
  rr=[r for r in rows if r['run']==run]
  summary[arm]['by_run'][run]={'machine':tally(rr,'machine'),'hand':tally(rr,'hand'),'tokens':sum(r['tokens'] for r in rr)}

curves={}
for arm in ('reach','f5room','c5room'):
 rows=[r for r in A if r['arm']==arm]
 curves[arm]={}
 for cap in (6000,8000,12000,16000,20000,24000):
  curves[arm][str(cap)]={field:dict(collections.Counter('cut' if r['tokens']>cap else r[field] for r in rows)) for field in ('machine','hand')}
late=[r for r in A if r['arm']=='f5room' and r['tokens']>6000]
hardrows=[r for r in A if r['arm']=='reach' and r['id'] in hard]
late_counts={f:tally(late,f) for f in ('machine','hand')}
hard_counts={f:tally(hardrows,f) for f in ('machine','hand')}
f5matched=[r for r in A if r['arm']=='f5room' and r['run'] in ('g','s1')]
f5base=[]
for run in ('g','s1','s2','s3'):
 for r in jl(SCI/f'final-f5-1104/f5-{run}.jsonl'):
  if r['id'] in ids:f5base.append(dict(run=run,machine=machine(r),tokens=(r.get('think_tokens') or 0)+(r.get('reply_tokens') or 0)))
assert len(f5base)==324

# Exercise unchanged scorer modules. Reads are explicit and every write is redirected here.
original_open=builtins.open
redirect_root=HERE/'scorer-probes-v1'
redirect_root.mkdir(exist_ok=True)
read_allow={p.resolve() for p in tracked}|{(HERE/f'corrections-{a}-v1.json').resolve() for a in files}
full_corrections={f"{r['arm']}:{r['run']}:{r['id']}":r['hand'] for r in A if r['finish']=='stop'}
reach_full={f"{r['run']}:{r['id']}":r['hand'] for r in A if r['arm']=='reach' and r['finish']=='stop'}
probe_stats={}
def exercise(label, script, data_dir, corrections=None, drop=None):
 out=io.StringIO(); writes=[]
 fake_fix=HERE/'probe-corrections-in-memory.json'
 def guarded_open(file,mode='r',*args,**kwargs):
  if isinstance(file,int): return original_open(file,mode,*args,**kwargs)
  path=Path(file).resolve()
  if any(c in mode for c in 'wax+'):
   dest=redirect_root/(label+'--'+path.name)
   writes.append(str(dest.relative_to(HERE)))
   return original_open(dest,mode,*args,**kwargs)
  if path==fake_fix.resolve() and isinstance(corrections,dict): return io.StringIO(json.dumps(corrections))
  if path not in read_allow: raise RuntimeError('unexpected scorer read: '+str(path))
  if drop and path==drop[0].resolve():
   return io.StringIO(''.join(json.dumps(r)+'\n' for r in jl(path) if r['id']!=drop[1]))
  return original_open(file,mode,*args,**kwargs)
 argv=[str(script),str(data_dir)]
 if corrections is not None:argv.append(str(fake_fix if isinstance(corrections,dict) else corrections))
 oldargv=sys.argv[:]
 builtins.open=guarded_open;sys.argv=argv
 status=0
 try:
  with contextlib.redirect_stdout(out),contextlib.redirect_stderr(out):
   runpy.run_path(str(script),run_name='__main__')
 except SystemExit as e:status=e.code
 finally:builtins.open=original_open;sys.argv=oldargv
 (redirect_root/(label+'.txt')).write_text(out.getvalue())
 probe_stats[label]={'exit':status,'writes_redirected':writes,'selected_lines':[s for s in out.getvalue().splitlines() if any(k in s for k in ('problems','hand','VERDICT'))]}

exercise('reach-exported-corrections',LAW/'reach1/reach_score.py',LAW/'reach1',HERE/'corrections-reach-v1.json')
exercise('reach-complete-hand',LAW/'reach1/reach_score.py',LAW/'reach1',reach_full)
exercise('f5room-exported-corrections',LAW/'f5room1/f5room_score.py',LAW/'f5room1',HERE/'corrections-f5room-v1.json')
exercise('f5room-complete-hand',LAW/'f5room1/f5room_score.py',LAW/'f5room1',full_corrections)
exercise('scimix-exported-corrections',LAW/'mix1/scimix_score.py',LAW/'mix1',HERE/'corrections-scimix-v1.json')
exercise('scimix-complete-hand',LAW/'mix1/scimix_score.py',LAW/'mix1',full_corrections)
exercise('scimix-missing-one-row',LAW/'mix1/scimix_score.py',LAW/'mix1',drop=(files['scimix']['g'],ids[0]))

# Small adversarial cases exercise the exact imported numeric grader.
cases=[
 ('wrong final answer after a correct earlier number','6','Earlier we estimated 6. Final answer: 11.',0,'wrong'),
 ('wrong physical unit','54.394','Final answer: 54.4 kg.',0,'wrong'),
 ('arbitrary thousand-fold scaling of a dimensionless ratio','1.33','Final ratio: 0.00133.',0,'wrong'),
 ('real kcal-to-kJ conversion','27','Final energy: 112.968 kJ per mole.',0,'right'),
 ('integer key rounding exceeds the internal seven-percent cap','5','Final answer: 4.6.',0,'right'),
 ('coarse wrong final value retained after a fine box','1.92',r'\boxed{1.92 \times 10^{12}}. The final acceleration is 2.0e12 m/s^2.',12,'wrong'),
 ('unanswered request containing the key','6','Please solve for the final count; the intermediate species has 6 electrons.',0,'wrong'),
]
grade_probes=[]
for label,answer,reply,scale,expected in cases:
 got=grade(dict(finish='stop',answer=answer,scale=scale,reply=reply))
 grade_probes.append(dict(case=label,answer=answer,scale=scale,reply=reply,expected=expected,actual=got))
grade_probes.append(dict(case='cut cannot receive credit from an embedded answer',expected='cut',actual=grade(dict(finish='length',answer='6',scale=0,reply='Final answer: 6.'))))

# Extract unchanged bar branches into memory for exact numerical boundary probes.
def branch_result(path,prefix,env):
 tree=ast.parse(path.read_text())
 selected=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.If) and ast.unparse(node.test).startswith(prefix):selected.append(node)
 assert len(selected)==1,(path,prefix,len(selected))
 ns=dict(env)
 exec(compile(ast.Module(body=[selected[0]],type_ignores=[]),str(path), 'exec'),ns)
 return ns['v']
boundary=[]
for name,path in [('c5',LAW/'f5room1/f5room_score.py'),('mix',LAW/'mix1/scimix_score.py')]:
 dt=90/100-1
 got=branch_result(path,'dr <= -8',dict(dr=0,dw=0,dt=dt))
 expected='THE COMPRESSOR KEEPS HIS REACH AND SAVES WORDS' if name=='c5' else 'SAME RIGHT, FEWER WORDS'
 boundary.append(dict(scorer=name,right_change=0,word_tokens=90,baseline_tokens=100,computed_change=dt,expected=expected,actual=got))

where={}
for arm in ('reach','f5room','c5room'):
 where[arm]={}
 for outcome in ('right','wrong','unclear','cut'):
  rows=[r for r in A if r['arm']==arm and r['hand']==outcome]
  where[arm][outcome]={'n':len(rows),'median_thinking':statistics.median([r['thinking'] for r in rows]) if rows else None,'median_reply':statistics.median([r['reply'] for r in rows]) if rows else None,'no_reply':sum(r['reply_empty'] for r in rows)}
 bands=[]
 for low,high in ((0,500),(500,1000),(1000,1500),(1500,2500),(2500,4000),(4000,8000),(8000,99999)):
  rows=[r for r in A if r['arm']==arm and low<=r['reply']<high]
  bands.append({'from_inclusive':low,'to_exclusive':high,'hand':tally(rows,'hand')})
 where[arm]['reply_length_bands']=bands

assert all(sha(p)==h for p,h in before.items()),'A protected input changed during checks'
assert sha(HERE/'verdicts-v1.jsonl')==lock
result={
 'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'integrity':{'all_attempts':972,'finished_replies':759,'cuts':213,'blind_reply_source_text_matches':759,'unique_source_ids':True,'fixed_wrong_twice_problems':13,'packet_hashes_match':84,'manifest_source_hashes_match':len(manifest['inputs']),'tool_hash_matches_manifest':True,'protected_files_unchanged':len(before),'verdict_sha256':lock},
 'arms':summary,'f5room_matched_g_s1':{'machine':tally(f5matched,'machine'),'hand':tally(f5matched,'hand'),'tokens':sum(r['tokens'] for r in f5matched)},
 'f5_6000_saved_machine':{'counts':tally(f5base,'machine'),'tokens':sum(r['tokens'] for r in f5base),'g_s1_tokens':sum(r['tokens'] for r in f5base if r['run'] in ('g','s1'))},
 'curves':curves,'f5room_past_6000':late_counts,'reach_wrong_twice':hard_counts,
 'where_the_work_is':where,'scorer_probes':probe_stats,'grade_probes':grade_probes,'bar_boundary_probes':boundary,'protected_hashes':before,
}
(HERE/'scoring-verification-v1.json').write_text(json.dumps(result,indent=2)+'\n')
(HERE/'joined-attempts-v1.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in A))
print(json.dumps({k:result[k] for k in ('integrity','arms','f5room_matched_g_s1','f5_6000_saved_machine','f5room_past_6000','reach_wrong_twice','scorer_probes','grade_probes','bar_boundary_probes')},indent=2))
