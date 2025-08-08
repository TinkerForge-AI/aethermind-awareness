#Continual learning + versioning (non-scary)
##Keep models small; checkpoint nightly; never overwrite—version everything.

--- /models/enc_fusion_YYYYMMDD.pt
--- /models/policy_bc_YYYYMMDD.pt
--- /models/metrics.jsonl
--- Keep LATEST symlinks for prod use.
--- Add a gate: new model only “promotes” if it beats last week’s metrics by X%.