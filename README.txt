SDPA causal-diagonal fixes for three dnn-benchmarking DVC tarballs
==================================================================

Tracking issue and check: see the dnn-benchmarking issue and PR that link here.
Base: dnn-benchmarking main 1eeb39e. The three .dvc pointers are unchanged
since 73fff8a:
  Workloads/microbench/cudnn_attention_inference.tar.gz  md5 45d692f85cda75a959c01d2a2a304cfc
  Workloads/microbench/aiter.tar.gz                      md5 d7c2ccc24668edf3e86135baaaf5f632
  Workloads/headline/attn.tar.gz                         md5 506430dcfe2729675a169b825ce87793

Problem
-------
The cudnn_attention_inference extraction wrote every decode graph as
causal_mask=true with diagonal_alignment TOP_LEFT. With Sq=1 that attends only
to key 0 of a 4K to 131K cache. With 1 < Sq < Skv the upstream cuDNN benchmark
means bottom-right (the last query sees the whole cache). The same Sq=1 shape
also occurs in 22 aiter varlen graphs and 1 attn graph.

Changes (251 files; each diff is 2 or 5 lines; original indentation kept)
-------------------------------------------------------------------------
- Sq=1, no window (71 files: 49 cudnn_attention_inference, 22 aiter):
  causal_mask true -> false. Unmasked attention is what an Sq=1 decode step
  computes, and is the same as bottom-right causal at Sq=1.
- Sq=1 with a sliding window (8 cudnn_attention_inference gpt_oss SWA,
  1 attn gpt-oss swa_sink_decode): diagonal_alignment TOP_LEFT -> BOTTOM_RIGHT,
  window kept, metadata.diagonal_alignment added.
- 1 < Sq < Skv (171 cudnn_attention_inference): diagonal_alignment
  TOP_LEFT -> BOTTOM_RIGHT, metadata.diagonal_alignment added. This matches the
  context_chunked graphs from the same extraction, which are already BOTTOM_RIGHT.

manifest.json lists every replaced member with its tarball, the fixed file in
this folder, and the original and fixed sha256.

Not changed (warnings only): 1248 aiter graphs and 10 attn graphs are top-left
causal with Sq != Skv and Sq > 1, and 4 aotriton graphs. They need a review of
what the source meant.

To publish (needs write access to s3://therock-dvc/dnn-benchmarking)
--------------------------------------------------------------------
  cd <dnn-benchmarking>
  dvc pull Workloads/microbench/cudnn_attention_inference.tar.gz.dvc \
           Workloads/microbench/aiter.tar.gz.dvc Workloads/headline/attn.tar.gz.dvc
  python <this folder>/apply_fixes.py .
  dvc add Workloads/microbench/cudnn_attention_inference.tar.gz \
          Workloads/microbench/aiter.tar.gz Workloads/headline/attn.tar.gz
  dvc push
  git add Workloads/microbench/cudnn_attention_inference.tar.gz.dvc \
          Workloads/microbench/aiter.tar.gz.dvc Workloads/headline/attn.tar.gz.dvc

apply_fixes.py checks the sha256 of every member it replaces and stops if one
changed upstream. It keeps all other members, their order and metadata. A
second run refuses, because the replaced members no longer match.

Checked: on the three tarballs pulled at the md5s above, apply_fixes.py replaced
228 + 22 + 1 members. check_deserialize --level json with the new checks over
all 31 workload tarballs (8868 graphs): before fail=80 warn=1433, after fail=0
warn=1262.
