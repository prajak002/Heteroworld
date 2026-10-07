import time, sys
from hetero_world.config import *
from hetero_world.benchmark import *
cfg=CorridorConfig(); pcfg=PredictionConfig(); ccfg=ConsensusConfig(outer_iters=2)
n=int(sys.argv[1]); methods=sys.argv[2].split(',')
for m in methods:
    t=time.time(); c=make_controller(m,cfg,pcfg,ccfg)
    rec=run_episodes(c,cfg,n,n)
    s=summarize(rec)
    print(m, '%.1fs'%(time.time()-t), {k:(round(v['mean'],3) if v['mean'] is not None else None) for k,v in s.items() if k in ['collision_rate_pct','deadlock_rate_pct','completed_rate_pct','bandwidth_kBps','plan_disagreement_m','belief_w2_m','makespan_s','min_separation']}, flush=True)
