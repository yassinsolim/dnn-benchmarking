# dnn-benchmarking

Benchmarking and validation tool for hipDNN graphs.

## Overview

> **Caution**: This tool is in early development and subject to change.
> Do not use it in build workflows or CI pipelines.

This tool loads serialized hipDNN graphs, executes them via installed hipDNN
engine plugins, and captures performance metrics. On AMD GPUs, hipDNN kernel
timing and synchronized E2E timing use direct HIP runtime events exposed by
`hipdnn_frontend`; PyTorch is only needed for the optional PyTorch executor and
reference validation.

On supported AMD GPUs, benchmark comparisons use a stall gate to exclude host
submission gaps from HIP event timing. If the watchdog releases that gate, the
tool discards the partial comparison and reruns every engine without stalling.
This avoids comparing device-only and unstalled samples.

The `--backend pytorch` executor also runs on NVIDIA GPUs with a CUDA PyTorch
build, where it times kernels with `torch.cuda` events. Because the hipDNN and
PyTorch backends share one suite execution path and emit the same
`SuiteResult` JSON schema, the same `--graph ... --backend pytorch -o out.json`
command can be run on a ROCm host and a CUDA host and the two JSON artifacts
compared offline. See [Cross-Machine Comparison](#cross-machine-comparison-rocm-vs-cuda).

## Requirements

- Python 3.12+
- numpy
- hipdnn_frontend (installed hipDNN Python bindings) — required for the hipDNN
  backend; **not** required for the `--backend pytorch` executor
- A GPU for execution:
  - AMD GPU with ROCm + hipDNN provider plugins — for the hipDNN backend and the
    ROCm PyTorch executor
  - NVIDIA GPU with a CUDA PyTorch build — for the `--backend pytorch` executor
- PyTorch *(optional for the hipDNN backend)* — any usable PyTorch build enables
  the `--validate pytorch` reference provider. A ROCm or CUDA PyTorch build
  enables the `--backend pytorch` GPU executor. Not listed in `pyproject.toml`
  because torch package selection depends on the target environment.

## Installation

### Quick Setup (ROCm/AMD GPUs)

Run the provided setup script from the `dnn-benchmarking` directory:

`rocm-libraries` (hipDNN sources and provider plugins) is a git submodule
tracking its `develop` branch by default. `setup_env.py` fetches it
automatically on first run via a sparse, blobless clone limited to the two
subtrees this tool builds (`projects/hipdnn`, `dnn-providers`) rather than
the full ~9GB monorepo, and build hipDNN plus the provider plugins from it
by default (pass `--reuse-artifacts` to skip and reuse whatever is already
installed instead). A submodule already populated by
`git submodule update --init` (a full, non-sparse checkout) is left as-is.
To build against a different rocm-libraries ref, check it out directly,
e.g. `git -C rocm-libraries fetch --depth 1 origin <ref> && git -C
rocm-libraries checkout FETCH_HEAD`.

Requires Python 3.12 or newer.

```bash
python3 setup_env.py --workspace .workspace
source .workspace/.venv/bin/activate
```

By default, setup uses `/workspace` when it already exists and is writable;
otherwise it uses `.workspace` under the `dnn-benchmarking` directory. Use
`--workspace <path>` to place the virtual environment, Python bytecode cache,
and runtime benchmark caches somewhere else:

```bash
python3 setup_env.py --workspace /tmp/dnn-bench
source /tmp/dnn-bench/.venv/bin/activate
```

The default `--torch-mode rocm` flow assumes no system ROCm installation:
1. Creates a virtual environment under the selected workspace (`--workspace`,
   `$DNN_BENCH_WORKSPACE`, writable `/workspace`, or local `.workspace`)
2. Detects the GPU architecture and installs the matching ROCm PyTorch nightly wheel
3. Discovers ROCm libraries from the torch wheel's bundled ROCm SDK libraries
4. Builds local hipDNN when CMake configs are absent from the selected prefix
5. Builds the local MIOpen, hipBLASLt, and hip-kernel providers when their
   installed artifacts are missing, using the bundled ROCm SDK devel wheel for
   compiler/toolchain discovery when needed
6. Installs the hipDNN Python bindings against the selected ROCm SDK libraries

The selected prefix is printed as `Using hipDNN/ROCm prefix: ...`; activation
sets `ROCM_PATH` to that prefix and prepends its `lib` directory to
`LD_LIBRARY_PATH`. dnn-benchmarking infers plugins from
`$ROCM_PATH/lib/hipdnn_plugins/engines`.
If GPU architecture detection is unavailable on the setup host, pass
`--gpu-arch gfx90a`, `--gpu-arch gfx942`, or `--gpu-arch gfx950`.

### Extra CMake Defines (`--cmake-arg`)

`setup_env.py` configures hipDNN and the provider plugins with a fixed set of
defaults. `--cmake-arg NAME=VALUE` appends an extra define to that configure.
It is repeatable, and the extra defines are appended *after* the defaults, so
one can override a default rather than be silently overridden by it.

Both `NAME=VALUE` and `-DNAME=VALUE` are accepted; the leading `-D` is added
when absent. Write the `-D` spelling with an `=` (`--cmake-arg=-DFOO=ON`) —
with a space, argparse reads `-DFOO=ON` as an option rather than a value.

```bash
# Skip rocKE, including its descriptor-backed engines
python3 setup_env.py --workspace .workspace \
  --cmake-arg HIPKERNELPROVIDER_ENABLE_ROCKE=OFF \
  --cmake-arg HIPDNN_ENABLE_KERNEL_INGESTOR=OFF

# Repeatable; the -D spelling needs the '=' form
python3 setup_env.py --cmake-arg HIPDNN_ENABLE_SDPA=OFF --cmake-arg=-DCMAKE_BUILD_TYPE=Debug
```

On Linux, setup builds rocKE and its descriptor-backed engines by default
(`hipkernel:Gfx950AttentionDense` on gfx950); Windows keeps them off. Disabling
an engine's option still installs the plugin `.so`, so graphs that only that
engine supports report `no engines applicable` rather than a missing-engine error.

### Testing/CI Setup with CPU-Only PyTorch

When ROCm/hipDNN artifacts are installed by CI, install CPU-only PyTorch on top
so Python reference validation can use torch without pulling conflicting ROCm
torch wheels:

```bash
python3 setup_env.py --torch-mode cpu --rocm-prefix /opt/rocm
source /workspace/.venv/bin/activate
```

CPU-only torch never enables the PyTorch execution backend; it is only for
Python reference validation. The `--backend pytorch` executor needs a GPU torch
build: ROCm torch (HIP-event timing) or CUDA torch (torch.cuda-event timing,
see below).

Use `--torch-mode existing` to reuse torch already installed in the target
virtual environment. Existing ROCm torch uses its bundled ROCm SDK libraries;
existing CUDA torch takes the CUDA skip path (no hipDNN bindings); existing
CPU-only torch builds the hipDNN bindings against `--rocm-prefix`, `$ROCM_PATH`,
or `/opt/rocm` for the hipDNN backend.

### CUDA PyTorch (NVIDIA GPUs, `--backend pytorch` only)

To benchmark graph shapes through PyTorch on an NVIDIA GPU — for offline
comparison against ROCm results — install a CUDA PyTorch build with
`--torch-mode cuda`:

```bash
python3 setup_env.py --torch-mode cuda --workspace .workspace
source .workspace/.venv/bin/activate
```

`--torch-mode cuda` installs torch from PyPI (override the index with
`--torch-index-url <url>`, e.g. a specific CUDA wheel channel) and **skips all
ROCm setup**: no hipDNN build, engine plugins, hipDNN Python bindings, amdsmi,
or `ROCM_PATH`/`LD_LIBRARY_PATH` wiring. Only the `--backend pytorch` executor
is available in this mode — the hipDNN backend requires `hipdnn_frontend` and
ROCm.

```bash
# Benchmark a graph suite through PyTorch CUDA and emit comparable JSON
python -m dnn_benchmarking --graph 'graphs/*.json' --backend pytorch -o cuda_results.json
```

On CUDA, kernel timing uses `torch.cuda` events and the ROCm-specific metadata
fields (`rocm_version`, amdsmi GPU snapshot) are `None` in the JSON, while
`gpu_arch` is the sentinel `"unknown"` (no ROCm gfx target is detectable); the
timing statistics and graph structure are identical to a ROCm run.


## Usage

### Basic Benchmarking

A single graph, a glob of graphs, and a tarball of graphs all share the same
execution path. By default results are printed as a summary table. Use `-v` for
the rich per-engine block (useful for debugging a single graph or comparing
engines).

```bash
# Single graph (default summary output)
dnn-benchmark --graph ./graphs/sample_conv_fwd.json --warmup 10 --iters 100

# Single graph, verbose: rich per-engine block
dnn-benchmark --graph ./graphs/sample_conv_fwd.json -v

# Filter to specific engine(s) — comma-separated
dnn-benchmark --graph ./graphs/sample_conv_fwd.json --engine 1
dnn-benchmark --graph ./graphs/sample_conv_fwd.json --engine 1,2

# Multiple graphs (glob): same path, default summary table
dnn-benchmark --graph 'graphs/*.json' --warmup 10 --iters 100

# With reproducible random seed
dnn-benchmark --graph ./graphs/sample_conv_fwd.json --seed 42
```

### Running from a Tarball

Pass a tarball directly to `--graph` and all `.json` files inside are extracted
to a temporary directory and run as a suite. The archive is cleaned up
automatically when the run finishes.

Supported formats: `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`

```bash
# Run every graph in a tarball (summary table)
dnn-benchmark --graph ./Workloads/conv_workloads.tar.gz

# Tarball + JSON output
dnn-benchmark --graph ./Workloads/conv_workloads.tar.gz --output results.json

# Tarball + verbose per-engine blocks
dnn-benchmark --graph ./Workloads/conv_workloads.tar.gz -v

# Glob that mixes tarballs and plain JSON files
dnn-benchmark --graph 'Workloads/*.tar.gz'
```

The extraction progress is reported on stderr:

```
Extracted 42 graph(s) from ./Workloads/conv_workloads.tar.gz
```

### Engine Comparison

Run multiple engines by passing comma-separated engine IDs. By default,
dnn-benchmarking infers the plugin directory from
`$ROCM_PATH/lib/hipdnn_plugins/engines` when `ROCM_PATH` is set by `setup_env.py`
activation. Plugin paths may also be a single shared directory or a
comma-separated list matching `--engine` order.

```bash
# Compare two engines using ROCM_PATH from the activated setup environment
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json \
  --engine 1,2

# Compare two plugin directories with specific engine IDs
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json \
  --engine 1,2 \
  --plugin-path /path/to/pluginA,/path/to/pluginB
```

### PyTorch Backend

`--backend pytorch` runs each graph through the PyTorch executor instead of
hipDNN engine plugins, producing one `provider="pytorch"` row per graph. It
shares the suite execution path with the hipDNN backend, so single-graph,
glob, and tarball inputs and `--output` JSON all work identically.

```bash
# Single graph through PyTorch
dnn-benchmark --graph ./graphs/sample_conv_fwd.json --backend pytorch

# A whole suite through PyTorch, with JSON output
dnn-benchmark --graph 'graphs/*.json' --backend pytorch -o pytorch_results.json

# Select the strict Flash category and prefer AOTriton within it on ROCm
dnn-benchmark --graph ./graphs/sample_sdpa.json --backend pytorch --pytorch-sdpa-backend flash --pytorch-rocm-fa-library aotriton -o pytorch_flash_aotriton.json
```

`--pytorch-sdpa-backend default` preserves normal PyTorch SDPA dispatch.
`flash`, `math`, `efficient`, `cudnn`, and `overrideable` are strict selectors:
the graph must execute native forward SDPA through the selected public category
or the benchmark errors. No non-default selection falls back to normal dispatch
or a CPU PyTorch reference.

`--pytorch-rocm-fa-library LIBRARY` is ROCm-only and requires
`--pytorch-sdpa-backend flash`. It forwards `LIBRARY` unchanged to PyTorch's
`preferred_rocm_fa_library`; for example, use `aotriton`. PyTorch rejects an
unknown library and may use another Flash implementation when the preferred one
cannot serve an input.

The PyTorch backend ignores hipDNN-specific selection and profiling options;
the following are rejected with `--backend pytorch`:

- `--engine` / `--plugin-path` (no hipDNN engine plugins are loaded)
- `--validate pytorch` (the backend would validate against itself)
- `--pmc` / `--emit-trace` / `--perf` / `--roofline` (rocprofv3-based passes)
- `--oracle-mode` (auto-tuning is a hipDNN engine feature)

### Oracle (Auto-Tuned) Comparison

Use `--oracle-mode` to compare the normal out-of-the-box (OOTB) plan with
hipDNN's tuned plan for each engine:

| Mode | Behavior |
|---|---|
| `off` | Run only the OOTB plan. This is the default. |
| `plan` | Benchmark every backend-generated plan for the engine. |
| `exhaustive` | Run `plan` mode and enable provider-managed kernel selection where supported. |

The exhaustive path keeps every plan returned by the backend. It does not
generate a Cartesian product of public knob values. The kernel ingestor and
MIOpen currently support provider-level selection; other engines, including
hipBLASLt, remain at plan-level tuning. Providers can reuse cached selections,
so exhaustive mode does not prove that every variant was measured during the
current invocation.

Both modes are slower than a normal run. Use `--engine` to limit the work.
`exhaustive` requires `--warmup >= 1`.

The summary table shows:

- `ootb_kernel_mean_ms`: the original OOTB measurement.
- `warm_ootb_kernel_mean_ms`: the same OOTB plan re-measured after tuning.
- `oracle_kernel_mean_ms`: the selected plan measured after tuning.
- `oracle_speedup`: `warm_ootb_kernel_mean_ms / oracle_kernel_mean_ms`.

The warm OOTB measurement is the comparison baseline because it has comparable
device warmup. The oracle can be slower; `0.99x` is a valid measured result.
An unsupported single-plan row prints `no-search` and stays outside the suite
geometric mean.

JSON records the candidate counts and the provider-level state:

| Field | Meaning |
|---|---|
| `compiled_plans_benchmarked` | Compiled plans measured successfully. |
| `compiled_plans_total` | Eligible compiled plans, including failures. |
| `exhaustive_requested` | The run requested provider-level tuning. |
| `exhaustive_supported` | The engine advertises `global.benchmarking`. |
| `exhaustive_enabled` | Provider-level tuning was requested and supported. |
| `tuning_available` | Multiple plans competed or provider-level tuning was enabled. |

These counts do not include provider-internal kernel variants; hipDNN exposes
no count for them. An empty `knob_settings` list means that no public plan knob
was set explicitly.

With `--validate`, the tool validates the OOTB and tuned plans independently.
It reports no speedup if either plan fails.

Cache state can affect selection. Set
`HIPDNN_DISABLE_EXACT_ENGINE_CACHE=1` for a cold heuristic baseline.
Exhaustive mode disables hipDNN's provider caches during its oracle pass, but
MIOpen FindDb and performance-database entries can still supply existing tuned
selections. The output records the relevant cache and MIOpen database paths in
`metadata.hipdnn_selection_env`.

```bash
HIPDNN_DISABLE_EXACT_ENGINE_CACHE=1 python -m dnn_benchmarking \
  --graph ./graphs/sample_conv_fwd.json \
  --oracle-mode exhaustive -v -o oracle.json
```

### Cross-Machine Comparison (ROCm vs CUDA)

Because both backends emit the same `SuiteResult` JSON schema, you can benchmark
the same graphs on an AMD machine and an NVIDIA machine and compare offline:

```bash
# On the AMD/ROCm host (hipDNN engines):
dnn-benchmark --graph 'graphs/*.json' -o rocm_results.json

# On the AMD/ROCm host (PyTorch executor, for an apples-to-apples PyTorch row):
dnn-benchmark --graph 'graphs/*.json' --backend pytorch -o rocm_pytorch_results.json

# On the NVIDIA/CUDA host (PyTorch executor):
dnn-benchmark --graph 'graphs/*.json' --backend pytorch -o cuda_pytorch_results.json
```

Each JSON file is a full `SuiteResult`: `graphs` is a list of graph entries,
each carrying its `graph_name` plus result rows with E2E and kernel timing
statistics and whatever machine metadata the host could provide
(`rocm_version` and the amdsmi snapshot are `None` on CUDA, and `gpu_arch` is
`"unknown"`). Graphs match across files by `graph_name`, so the artifacts can
be diffed offline. (An offline comparison helper is planned but not yet
included.)

### Kernel Selection (`--autotune`, `--cache-dir`)

An engine has two kernel-selection paths, and they answer different questions.
By default it serves its cold heuristic's rank-0 pick, so a table measures the
*heuristic*. `--autotune` sets `HIPDNN_FORCE_BENCHMARKING=1`, which samples
every knob-filtered candidate on each plan's first execute and caches the
winner — that measures what the shipped *kernel set* can deliver. Use it for
any best-vs-best comparison: without it, an engine that gains good variants can
measure slower when the heuristic tie-break is a coin flip.

The selected path is always printed, whether or not it was requested, so a
table is never ambiguous about which question it answers.

`--cache-dir` sets `HIPDNN_CACHE_DIR` for the run. The winner cache is on disk
and is keyed by graph content and device — not by checkout, engine, or session
— and reads are not gated on benchmarking while writes are. Two runs over the
same graphs therefore share rankings, and an untuned run can silently report a
ranking some other run tuned. Give each phase its own empty root:

```bash
# Measure the heuristic (default) — path is announced on stderr
dnn-benchmark --graph 'graphs/*.json' --cache-dir /tmp/cache-cold

# Measure the kernel set: benchmark every candidate, into an isolated cache
dnn-benchmark --graph 'graphs/*.json' --autotune --cache-dir /tmp/cache-tuned
```

### Config Files

Use `--config` for repeatable benchmark recipes. CLI flags override config
values, so a recipe can be reused with per-run workload or iteration changes.
Relative paths in a config file are resolved from that config file's directory.

```bash
python -m dnn_benchmarking --config sample_configs/basic.toml.example --graph ./graphs/sample_conv_fwd.json
python -m dnn_benchmarking --config sample_configs/config.toml.example --iters 500
```

### Command-line reference

Run `dnn-benchmark --help` for the authoritative option list and defaults.

## Output

### Default Output (summary table)

The default console output is a compact, suite-style summary. One line per
graph reports the per-engine pass/fail counts, followed by a final summary
block. JSON output (`--output`) always contains the full per-engine
`SuiteResult` regardless of console verbosity.

```
================================================================================
hipDNN Benchmark Suite: 3 graph(s)
================================================================================

[1/3] sample_conv_fwd...
  -> 2 passed, 0 failed, 0 skipped, 0 errored
[2/3] sample_matmul...
  -> 2 passed, 0 failed, 0 skipped, 0 errored
[3/3] sample_relu...
  -> 1 passed, 1 failed, 0 skipped, 0 errored

--------------------------------------------------------------------------------
Suite Summary:
  Graphs:       3
  Combinations: 6
  Passed:       5
  Failed:       1
  Skipped:      0
  Errors:       0
================================================================================
```

### Verbose Output (`-v`)

`-v` switches to a detailed per-engine block per graph. Use it when debugging a
single graph or comparing engines side-by-side.

## Related Tools

For the MIOpen shape conversion tool, see [`dnn-convert-shapes`](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/tools/dnn-convert-shapes).

## Workload Files

The `Workloads/` directory contains benchmark workload tarballs tracked with
[DVC](https://dvc.org/). The public DVC remote is configured in `.dvc/config`
for anonymous reads; archive contents are not stored directly in Git.

- `Workloads/headline/` — the small, curated set currently tracked for
  regression monitoring: `conv.tar.gz`, `bnorm.tar.gz`, `attn.tar.gz`,
  `hipblaslt.tar.gz`, `moe.tar.gz`, `norm.tar.gz`. Expected to grow over
  time. Note: `norm.tar.gz` currently has 0% applicability (no ROCm engine
  implements RMSNorm/LayerNorm yet) -- tracked here ahead of engine support
  landing, not because it has a signal today.
- `Workloads/microbench/` — broader backing/brute-force sets (raw shape
  sweeps, per-source SDPA/norm/GEMM/MoE collections) not part of the
  frequent-cadence headline signal.
- `Workloads/models/` — per-model workload collections.

### Download workload archives

Install DVC with S3 support, then pull every tracked workload:

```bash
python -m pip install "dvc[s3]"
dvc pull
```

To fetch one workload instead:

This downloads the tar files tracked by `.dvc` pointer files in `Workloads/`. If the file is already cached locally, DVC restores it without re-downloading.

```bash
dvc pull Workloads/headline/conv.tar.gz.dvc
```

Keep credentials in DVC's ignored local configuration (`.dvc/config.local`);
the committed configuration supports anonymous access and contains no secrets.

### Validating graphs

`tools/check_deserialize.py` checks that graph JSON files deserialize and validate
without building a plan or running a kernel. It has three escalating levels:

```bash
# Pure-Python loader only (no hipDNN build required)
python tools/check_deserialize.py --level json --src src 'Workloads/**/*.json'

# Full deserialize + build/finalize the backend operation graph (needs a built hipDNN)
python tools/check_deserialize.py --level opgraph 'Workloads/**/*.json'
```

Run this after adding new workload graphs to confirm hipDNN can load them. Paths may
be globs, directories, or tarball-extracted trees; the script exits non-zero on any
failure and prints the first failures with their error messages.

## Running Tests

### Quick Start

```bash
# Activate venv
source /workspace/.venv/bin/activate  # or $DNN_BENCH_WORKSPACE/.venv/bin/activate

# All non-GPU tests (no hipDNN required)
pytest -m "not gpu"

# All tests including GPU (activation sets LD_LIBRARY_PATH for setup workspaces)
pytest

# Only GPU tests
pytest -m gpu

# GPU tests with explicit hipDNN engine plugin directories
pytest -m gpu --dnn-plugin-paths /path/to/hipdnn_plugins/engines
```

### GPU Tests

GPU tests require hipDNN Python bindings and ROCm libraries. After activating
the setup environment from Quick Start, run:

```bash
pytest -m gpu
```

GPU tests auto-discover provider build-tree, active-venv ROCm SDK, and
`/opt/rocm` plugin installs. Use `--dnn-plugin-paths` with a comma-separated
directory list when testing custom engine plugin builds.

GPU tests are tiered by marker: `gpu` (any GPU), `rocm` (AMD ROCm only),
`cuda` (NVIDIA CUDA only). GPU-generic tests run on either platform and
adapt their timing-backend assertion automatically (HIP on ROCm,
torch.cuda on CUDA); platform-specific tests assert one backend's unique
behavior. Every GPU test self-skips on the wrong platform, so bare
`pytest` is safe on any host. Use `-m` for explicit, additive selection:

```bash
# On a ROCm host: unit + generic + rocm (drops cuda-only tests)
pytest -m "not cuda"

# On a CUDA host: unit + generic + cuda (drops rocm-only tests)
pytest -m "not rocm"

# Only one platform's dedicated tests
pytest -m rocm
pytest -m cuda
```

Strict profiling tests that require real profiler artifacts are skipped by
default. Run them explicitly on a known-good profiling host:

```bash
LD_LIBRARY_PATH=$HIPDNN_PREFIX/lib:$LD_LIBRARY_PATH pytest --profiling-strict -m profiling_strict
```

## Limitations

- Engine comparison and timed validation-provider rows are reported side by side. Reference rows are timing baselines and are not counted as hipDNN engine pass/fail combinations; use `--validate` for reference-output correctness checks.
