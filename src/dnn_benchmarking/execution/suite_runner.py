# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite runner for per-graph engine iteration with granular timing.

Uses explicit ``--engine`` IDs in caller order when provided; otherwise
discovers ranked engine IDs for the graph via ``Graph.get_ranked_engine_ids``.
For each engine, captures separated CPU build time, GPU kernel time,
and E2E wall-clock time. Performs correctness validation by comparing GPU
output against a reference provider via ArrayComparator.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from ..common.exceptions import ExecutionError, UnsupportedGraphError
from ..config.benchmark_config import (
    BenchmarkConfig,
    PyTorchSdpaBackendName,
    ReferenceProviderName,
    SuiteConfig,
)
from ..execution.buffer_manager import BufferManager, generate_input_data
from ..execution.executor import Executor
from ..execution.timing import StallFallbackError, Timer
from ..metrics import (
    CpuTimeProbe,
    GpuSmiProbe,
    compute_flops,
    compute_io_bytes,
    derive_throughputs,
)
from ..metrics._diagnostic import warn_once
from ..reporting.reporter import Reporter
from ..reporting.statistics import BenchmarkStats
from ..reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    OracleResult,
    ProviderEngineResult,
    build_oracle_delta,
)
from ..validation.reference_provider import (
    ReferenceOutput,
    ReferenceProvider,
    ReferenceProviderRegistry,
)
from ..validation.validator import Validator


# bf16 has a 7-bit mantissa: 1 ULP ~= 2^-7 = 0.78% relative. Backward
# convolutions (wgrad/dgrad) accumulate over large reductions, and the MIOpen
# kernels hipDNN and PyTorch select round 2-3 ULP apart even when they pick the
# same solver. A 1% (~1.3 ULP) rtol flags that legitimate bf16 drift as a
# failure; 3% (~3.8 ULP) keeps validation meaningful while tolerating it.
_BFLOAT16_RTOL = 3e-2
_BFLOAT16_ATOL = 1e-3
_HALF_RTOL = 1e-3
_HALF_ATOL = 1e-3
_DEFAULT_RTOL = 1e-5
_DEFAULT_ATOL = 1e-6


@dataclass
class _TimedPytorchRow:
    """Timed PyTorch row plus reference outputs (reference role only)."""

    result: ProviderEngineResult
    outputs: Optional[Dict[int, ReferenceOutput]]


_STALL_FALLBACK_WARNING = (
    "Stall-gated timing failed; every engine was remeasured without stalling."
)


def _mark_stall_fallback(result: GraphResult) -> GraphResult:
    for row in result.results:
        row.warnings = [*(row.warnings or []), _STALL_FALLBACK_WARNING]
    return result


def _output_node_types(graph_json: Dict[str, Any]) -> Dict[int, str]:
    output_to_node: Dict[int, str] = {}
    for node in graph_json.get("nodes", []):
        node_type = str(node.get("type", ""))
        for uid in (node.get("outputs") or {}).values():
            if uid is not None:
                output_to_node[int(uid)] = node_type
    return output_to_node


def _default_tolerance_for_output(
    tensor_info: Any,
    output_node_type: Optional[str],
) -> tuple[float, float]:
    dtype = str(getattr(tensor_info, "data_type", "float")).lower()
    if dtype == "bfloat16":
        return _BFLOAT16_RTOL, _BFLOAT16_ATOL
    if dtype == "half":
        return _HALF_RTOL, _HALF_ATOL
    return _DEFAULT_RTOL, _DEFAULT_ATOL


def _fallback_tolerance_for_config(config: SuiteConfig) -> tuple[float, float]:
    """Return explicit tolerances, or the default float tolerance for reporting."""
    return config.validation.tolerance_override or (_DEFAULT_RTOL, _DEFAULT_ATOL)


def _tolerance_for_output(
    config: SuiteConfig,
    tensor_info: Any,
    output_node_type: Optional[str],
) -> tuple[float, float]:
    override = config.validation.tolerance_override
    if override is not None:
        return override
    return _default_tolerance_for_output(tensor_info, output_node_type)


def _hipdnn_buffer_device(
    reference_outputs: Optional[Dict[int, ReferenceOutput]],
) -> Optional[str]:
    """Torch device for hipDNN I/O buffers, or None for hipdnn.DeviceBuffer.

    Torch storage is used only when a device reference exists, so the
    outputs can be compared on the GPU. Timing-only runs, and runs whose
    reference is host-only, keep the DeviceBuffer baseline. ``"cuda"`` is
    torch's current device, which torch reads with ``hipGetDevice``. The
    hipDNN handle uses the same in-process HIP device, so the pointers
    belong to the handle's device.
    """
    if reference_outputs and any(
        ref.device_data is not None for ref in reference_outputs.values()
    ):
        return "cuda"
    return None


def _resolve_engine_name(engine_id: int) -> str:
    """Resolve an engine ID to its registered name.

    Looks up the name via ``hipdnn_frontend.engine_id_to_name``. If the ID
    isn't registered (returns empty string), falls back to a hex display
    string so callers always have something printable.

    Unexpected exceptions (plugin import error, registry corruption)
    emit a one-shot warning to stderr so a silent fallback doesn't hide
    a real plugin-init bug — the hex fallback only changes the artifact
    path and reporting label, but the underlying error usually indicates
    a broader registry problem worth surfacing.

    Args:
        engine_id: int engine ID.

    Returns:
        Registered engine name or ``f"engine_0x..."`` fallback.
    """
    try:
        import hipdnn_frontend as hipdnn

        name = hipdnn.engine_id_to_name(engine_id)
        if name:
            return name
    except Exception as e:
        from ..metrics._diagnostic import warn_once

        warn_once(
            "suite_runner",
            f"engine_id_to_name failed for {engine_id:#x}: {e}; "
            "falling back to hex display string",
        )
    return f"engine_{engine_id:#x}"


def _resolve_engine_version(handle: Any, engine_id: int) -> str:
    """Return the loaded provider plugin version for an engine ID."""
    try:
        info = handle.get_engine_info(engine_id)
        if info.version:
            return str(info.version)
    except Exception as e:
        warn_once(
            "suite_runner",
            f"engine version lookup failed for {engine_id:#x}: {e}",
        )
    return "unavailable"


def set_plugin_path(
    hipdnn: Any, plugin_path: Optional[Path], loading_mode: Optional[Any] = None
) -> None:
    """Set the process-wide hipDNN plugin search path for the next handle."""
    if plugin_path is None:
        return
    paths = [str(plugin_path)]
    if loading_mode is None:
        loading_mode = hipdnn.PluginLoadingMode.ABSOLUTE
    hipdnn.set_engine_plugin_paths(paths, loading_mode)


def _engine_setup_error_result(
    provider: str,
    engine_id: int,
    plugin_path: Optional[Path],
    config: SuiteConfig,
    error_message: str,
) -> ProviderEngineResult:
    """Build a per-engine error row for plugin-path/handle setup failures."""
    rtol, atol = _fallback_tolerance_for_config(config)
    return ProviderEngineResult(
        provider=provider,
        engine_id=engine_id,
        status="error",
        plugin_path=str(plugin_path) if plugin_path is not None else None,
        error_message=error_message,
        correctness=CorrectnessResult.failed(
            rtol=rtol, atol=atol, error_message=error_message
        ),
    )


def _get_reference_provider(
    config: SuiteConfig, graph_json: Dict[str, Any]
) -> Optional[ReferenceProvider]:
    """Attempt to get and validate a reference provider for this graph.

    Args:
        config: Suite configuration with validation settings.
        graph_json: Parsed graph JSON dictionary.

    Returns:
        ReferenceProvider instance if available and supports the graph,
        None if validation was not requested (``config.validation`` is
        disabled) or if the provider is unavailable/unsupported.
    """
    if not config.validation.enabled:
        return None

    try:
        provider = ReferenceProviderRegistry.get_provider(
            config.validation.provider.value
        )
    except ValueError:
        print(
            f"Reference provider '{config.validation.provider.value}' not registered",
            file=sys.stderr,
        )
        return None

    if not provider.is_available():
        print(
            f"Reference provider '{config.validation.provider.value}' not available",
            file=sys.stderr,
        )
        return None

    if not provider.supports_graph(graph_json):
        print(
            f"Reference provider '{config.validation.provider.value}' "
            "does not support this graph",
            file=sys.stderr,
        )
        return None

    return provider


def _check_correctness(
    buffer_manager: BufferManager,
    tensor_infos: list,
    graph_json: Dict[str, Any],
    ref_outputs: Dict[int, ReferenceOutput],
    reference_provider_name: str,
    config: SuiteConfig,
) -> CorrectnessResult:
    """Compare GPU outputs against precomputed reference outputs.

    Reference providers are executed once per graph by the suite runner. This
    helper only compares a single hipDNN engine's outputs against that cached
    reference output map.
    """
    try:
        output_node_types = _output_node_types(graph_json)
        all_passed = True
        worst_abs_diff = 0.0
        worst_rel_diff = 0.0
        used_rtol = 0.0
        used_atol = 0.0

        output_count = 0
        for ti in tensor_infos:
            if not ti.is_output:
                continue

            rtol, atol = _tolerance_for_output(
                config, ti, output_node_types.get(ti.uid)
            )
            validator = Validator(rtol=rtol, atol=atol)
            ref = ref_outputs.get(ti.uid)

            # Compare on the GPU when both sides are device tensors.
            result = None
            if ref is not None and ref.device_data is not None:
                actual_tensor = buffer_manager.get_output_tensor(ti.uid)
                if actual_tensor is not None:
                    import torch

                    try:
                        result = validator.validate_tensors(
                            actual_tensor, ti, ref.device_data
                        )
                    except torch.cuda.OutOfMemoryError:
                        result = None  # Fall back to the host comparison.

            if result is None:
                actual = buffer_manager.get_output_data(ti.uid)
                if actual is None:
                    continue
                if ref is None:
                    return CorrectnessResult(
                        execution_success=True,
                        tolerance_match=False,
                        rtol=rtol,
                        atol=atol,
                        error_message=(
                            f"Reference provider '{reference_provider_name}' did not "
                            f"produce output tensor UID {ti.uid}"
                        ),
                    )
                result = validator.validate(actual, ti, reference_data=ref.data)
            output_count += 1
            used_rtol = max(used_rtol, rtol)
            used_atol = max(used_atol, atol)

            if not result.passed:
                all_passed = False

            if result.max_abs_diff > worst_abs_diff:
                worst_abs_diff = result.max_abs_diff
            if result.max_rel_diff > worst_rel_diff:
                worst_rel_diff = result.max_rel_diff

        if output_count == 0:
            rtol, atol = _fallback_tolerance_for_config(config)
            return CorrectnessResult(
                execution_success=True,
                tolerance_match=False,
                rtol=rtol,
                atol=atol,
                error_message="No output tensors to compare",
            )

        return CorrectnessResult(
            execution_success=True,
            tolerance_match=all_passed,
            rtol=used_rtol,
            atol=used_atol,
            max_abs_diff=worst_abs_diff,
            max_rel_diff=worst_rel_diff,
        )

    except (ValueError, RuntimeError, ExecutionError) as e:
        rtol, atol = _fallback_tolerance_for_config(config)
        return CorrectnessResult(
            execution_success=True,
            tolerance_match=False,
            rtol=rtol,
            atol=atol,
            error_message=str(e),
        )


def _reference_unavailable_correctness(
    config: SuiteConfig, error_message: str
) -> CorrectnessResult:
    rtol, atol = _fallback_tolerance_for_config(config)
    return CorrectnessResult(
        execution_success=True,
        tolerance_match=False,
        rtol=rtol,
        atol=atol,
        error_message=error_message,
    )


def _reference_row_correctness(config: SuiteConfig) -> CorrectnessResult:
    rtol, atol = _fallback_tolerance_for_config(config)
    return CorrectnessResult(
        execution_success=True,
        tolerance_match=None,
        rtol=rtol,
        atol=atol,
        error_message="Reference provider timing row; no comparison performed",
    )


def _compute_reference_outputs_once(
    ref_provider: ReferenceProvider,
    graph_json: Dict[str, Any],
    input_data: Dict[int, Any],
    config: SuiteConfig,
) -> tuple[Optional[Dict[int, ReferenceOutput]], Optional[str]]:
    try:
        from ..validation.providers.pytorch_provider import PyTorchReferenceProvider

        if isinstance(ref_provider, PyTorchReferenceProvider):
            from .pytorch_ops import (
                PyTorchSdpaBackendState,
                use_pytorch_sdpa_backend,
            )

            state = PyTorchSdpaBackendState(
                config.pytorch_sdpa_backend,
                config.pytorch_rocm_fa_library,
            )
            with use_pytorch_sdpa_backend(state):
                outputs = ref_provider.compute_reference(graph_json, input_data)
            return outputs, None
        return ref_provider.compute_reference(graph_json, input_data), None
    except (ImportError, ValueError, RuntimeError, ExecutionError) as e:
        return None, str(e)


def _pytorch_reference_outputs_from_buffer(
    buffer_manager: Any,
    keep_device: bool = True,
) -> Dict[int, ReferenceOutput]:
    """Collect reference outputs from a PyTorch buffer manager.

    ``data`` is always a host copy (one per output per graph); the host
    fallback comparison needs it. With ``keep_device``, a device clone is
    also kept so each engine can compare on the GPU without a host copy of
    its own output.
    """
    outputs: Dict[int, ReferenceOutput] = {}
    tensors = buffer_manager.get_tensors()
    for tensor_info in buffer_manager.get_output_tensors():
        data = buffer_manager.get_output_data(tensor_info.uid)
        if data is not None:
            tensor = tensors.get(tensor_info.uid)
            outputs[tensor_info.uid] = ReferenceOutput(
                data=data,
                tensor_uid=tensor_info.uid,
                # Clone: the buffer manager frees its tensors on exit.
                device_data=(
                    tensor.detach().clone()
                    if keep_device and tensor is not None and tensor.is_cuda
                    else None
                ),
            )
    return outputs


def _compute_graph_analytical_metrics(
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    graph_name: str,
) -> tuple[Optional[int], bool, Optional[int]]:
    """Compute per-graph analytical FLOPs/IO once.

    These are a function of the graph shape only and don't change across
    engines, so they're computed once per graph and propagated onto every
    row. Failures route through warn_once and yield None.

    Returns:
        (analytical_flops, analytical_flops_partial, analytical_io_bytes)
    """
    analytical_flops: Optional[int] = None
    analytical_flops_partial = False
    analytical_io_bytes: Optional[int] = None
    if config.metrics.basic_enabled:
        try:
            analytical_flops, analytical_flops_partial = compute_flops(graph_json)
        except Exception as e:
            warn_once("analytical", f"compute_flops failed for {graph_name}: {e}")
        try:
            analytical_io_bytes = compute_io_bytes(tensor_infos)
        except Exception as e:
            warn_once("analytical", f"compute_io_bytes failed for {graph_name}: {e}")
    return analytical_flops, analytical_flops_partial, analytical_io_bytes


def _run_timed_pytorch_row(
    graph_path: Path,
    graph_json: Dict[str, Any],
    graph_name: str,
    tensor_infos: list,
    config: SuiteConfig,
    input_data: Dict[int, Any],
    analytical_flops: Optional[int],
    analytical_flops_partial: bool,
    analytical_io_bytes: Optional[int],
    role: Literal["engine", "reference"] = "reference",
    allow_staging: bool = True,
) -> _TimedPytorchRow:
    """Run PyTorch once as a timed suite row.

    With ``role="reference"`` this is the timed validation-provider row: it
    additionally extracts reference outputs for engine comparison. Failed
    default-dispatch references are skipped; failed strict selections are
    errors because their requested native SDPA path was not fulfilled. With
    ``role="engine"`` (the ``--backend pytorch`` path), failures are errors.
    """
    from . import pytorch_ops

    try:
        from importlib.metadata import version

        engine_version = version("torch")
    except Exception:
        engine_version = "unavailable"
    result = ProviderEngineResult(
        provider=ReferenceProviderName.PYTORCH.value,
        engine_id=0,
        status="skipped",
        engine_version=engine_version,
    )
    outputs: Optional[Dict[int, ReferenceOutput]] = None
    strict_selection = config.pytorch_sdpa_backend is not PyTorchSdpaBackendName.DEFAULT

    with Timer() as elapsed_timer:
        try:
            from ..execution.pytorch_buffer_manager import PyTorchCudaBufferManager
            from ..execution.pytorch_executor import PyTorchCudaExecutor

            bench_config = BenchmarkConfig(
                graph_path=graph_path,
                warmup_iters=config.warmup_iters,
                benchmark_iters=config.benchmark_iters,
                engine_id=0,
                pytorch_sdpa_backend=config.pytorch_sdpa_backend,
                pytorch_rocm_fa_library=config.pytorch_rocm_fa_library,
            )
            executor = PyTorchCudaExecutor(graph_json, bench_config)
            executor.prepare()
            result.cpu_build_time_ms = executor.init_time_ms

            with PyTorchCudaBufferManager(tensor_infos) as buffer_manager:
                buffer_manager.allocate_all()
                buffer_manager.load_input_data(input_data)
                buffer_manager.zero_outputs()

                tensors = buffer_manager.get_tensors()
                executor.warmup(tensors)

                cpu_time_probe = (
                    CpuTimeProbe() if config.metrics.basic_enabled else None
                )
                if cpu_time_probe is not None:
                    cpu_time_probe.__enter__()
                try:
                    bench_result = executor.benchmark(
                        tensors, graph_name=graph_name, allow_staging=allow_staging
                    )
                finally:
                    if cpu_time_probe is not None:
                        cpu_time_probe.__exit__(None, None, None)

                result.host_stats = BenchmarkStats.from_timings(
                    bench_result.host_timings
                )
                if bench_result.has_kernel_timings:
                    result.gpu_kernel_stats = BenchmarkStats.from_timings(
                        bench_result.kernel_timings
                    )
                assert bench_result.metadata is not None

                if config.metrics.basic_enabled:
                    _collect_basic_metrics_post_loop(
                        result=result,
                        cpu_time_probe=cpu_time_probe,
                        benchmark_iters=config.benchmark_iters,
                        analytical_flops=analytical_flops,
                        analytical_flops_partial=analytical_flops_partial,
                        analytical_io_bytes=analytical_io_bytes,
                    )

                if role == "reference":
                    buffer_manager.zero_outputs()
                    executor.execute_once(tensors)
                    # A profiling child allocates its own VRAM, so profiled
                    # runs keep host-only references (and DeviceBuffer I/O).
                    outputs = _pytorch_reference_outputs_from_buffer(
                        buffer_manager,
                        keep_device=not config.metrics.opt_in_pass_requested,
                    )

            if role == "reference":
                result.correctness = _reference_row_correctness(config)
                warnings = pytorch_ops.get_reference_warnings(graph_json)
                if warnings:
                    result.warnings = warnings
            else:
                rtol, atol = _fallback_tolerance_for_config(config)
                result.correctness = CorrectnessResult(
                    execution_success=True,
                    tolerance_match=None,
                    rtol=rtol,
                    atol=atol,
                    error_message="No reference provider requested",
                )
            result.status = "success"

        except StallFallbackError:
            raise
        except UnsupportedGraphError as e:
            msg = str(e)
            if strict_selection:
                result.status = "error"
                result.error_message = msg
            else:
                result.status = "skipped"
                result.skip_reason = msg
            if role == "engine":
                rtol, atol = _fallback_tolerance_for_config(config)
                result.correctness = CorrectnessResult.failed(
                    rtol=rtol, atol=atol, error_message=msg
                )
        except Exception as e:
            msg = str(e)
            if role == "engine" or strict_selection:
                result.status = "error"
                result.error_message = msg
                if role == "engine":
                    rtol, atol = _fallback_tolerance_for_config(config)
                    result.correctness = CorrectnessResult.failed(
                        rtol=rtol, atol=atol, error_message=msg
                    )
            else:
                result.skip_reason = msg
    result.elapsed_time_ms = elapsed_timer.elapsed_ms
    return _TimedPytorchRow(result=result, outputs=outputs)


def run_graph_all_providers(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    handle: Any,
    reporter: Optional[Reporter] = None,
    _allow_staging: bool = True,
    _engine_ids: Optional[List[int]] = None,
) -> GraphResult:
    """Run a single graph against every engine the backend ranks for it.

    Discovers engine IDs via ``Graph.get_ranked_engine_ids``, applies any
    user engine filter, and runs the benchmark for each remaining ID.
    Captures separated CPU build time, GPU kernel time, and E2E wall-clock
    time per engine. Performs correctness checking against a reference
    provider when ``--validate`` was requested.

    Args:
        graph_path: Path to the graph JSON file.
        graph_json: Parsed graph JSON dictionary.
        tensor_infos: List of TensorInfo objects for the graph.
        config: Suite configuration.
        handle: hipdnn.Handle instance.

    Returns:
        GraphResult with one ProviderEngineResult per engine.
    """
    graph_name = graph_json.get("name", graph_path.stem)
    graph_json_str = json.dumps(graph_json)

    validation_requested = config.validation.enabled

    if _engine_ids is not None:
        engine_ids = _engine_ids
    elif config.engine_filter is not None:
        # Explicit --engine is a selection, not a post-discovery filter. Keep the
        # caller's order so per-engine plugin paths are deterministic.
        engine_ids = list(config.engine_filter)
    else:
        # Discover engines via real backend heuristics. A discovery failure is a
        # graph-level error (record it and stop iterating engines), but "no
        # engine configurations available" / "not supported" messages are
        # really an unsupported-graph signal.
        discovery_config = BenchmarkConfig(
            graph_path=graph_path,
            warmup_iters=config.warmup_iters,
            benchmark_iters=config.benchmark_iters,
        )
        try:
            if handle is None:
                import hipdnn_frontend as hipdnn

                handle = hipdnn.Handle()
            discovery_executor = Executor(
                graph_json_str=graph_json_str,
                config=discovery_config,
            )
            engine_ids = discovery_executor.discover_engines(handle)
        except UnsupportedGraphError as e:
            rtol, atol = _fallback_tolerance_for_config(config)
            return GraphResult(
                graph_name=graph_name,
                graph_path=str(graph_path),
                results=[
                    ProviderEngineResult(
                        provider="unknown",
                        engine_id=0,
                        status="skipped",
                        skip_reason=str(e),
                        correctness=CorrectnessResult.failed(
                            rtol=rtol, atol=atol, error_message=str(e)
                        ),
                    )
                ],
            )
        except (ExecutionError, RuntimeError) as e:
            msg = str(e)
            rtol, atol = _fallback_tolerance_for_config(config)
            return GraphResult(
                graph_name=graph_name,
                graph_path=str(graph_path),
                results=[
                    ProviderEngineResult(
                        provider="unknown",
                        engine_id=0,
                        status="error",
                        error_message=f"Engine discovery failed: {msg}",
                        correctness=CorrectnessResult.failed(
                            rtol=rtol, atol=atol, error_message=msg
                        ),
                    )
                ],
            )

    if not engine_ids:
        return GraphResult(
            graph_name=graph_name,
            graph_path=str(graph_path),
            results=[
                ProviderEngineResult(
                    provider="unknown",
                    engine_id=0,
                    status="error",
                    error_message=(
                        "No engines discovered for graph"
                        if config.engine_filter is None
                        else "No engines selected for graph"
                    ),
                )
            ],
        )
    engine_selections = config.engine_selections_for(engine_ids)
    ref_provider = _get_reference_provider(config, graph_json)
    try:
        graph_input_data = generate_input_data(tensor_infos, config.seed, graph_json)
    except (ValueError, RuntimeError, OSError, TypeError, OverflowError) as e:
        msg = f"Input data generation failed: {e}"
        rtol, atol = _fallback_tolerance_for_config(config)
        return GraphResult(
            graph_name=graph_name,
            graph_path=str(graph_path),
            results=[
                ProviderEngineResult(
                    provider="unknown",
                    engine_id=0,
                    status="error",
                    error_message=msg,
                    correctness=CorrectnessResult.failed(
                        rtol=rtol, atol=atol, error_message=msg
                    ),
                )
            ],
            engine_ids=engine_ids,
        )
    reference_outputs: Optional[Dict[int, ReferenceOutput]] = None
    reference_error: Optional[str] = None
    timed_reference: Optional[_TimedPytorchRow] = None

    (
        analytical_flops,
        analytical_flops_partial,
        analytical_io_bytes,
    ) = _compute_graph_analytical_metrics(graph_json, tensor_infos, config, graph_name)

    pe_results: List[ProviderEngineResult] = []
    if (
        ref_provider is not None
        and config.validation.provider is ReferenceProviderName.PYTORCH
    ):
        if reporter is not None:
            reporter.print_engine_start("pytorch reference")
        try:
            timed_reference = _run_timed_pytorch_row(
                graph_path=graph_path,
                graph_json=graph_json,
                graph_name=graph_name,
                tensor_infos=tensor_infos,
                config=config,
                input_data=graph_input_data,
                analytical_flops=analytical_flops,
                analytical_flops_partial=analytical_flops_partial,
                analytical_io_bytes=analytical_io_bytes,
                allow_staging=_allow_staging,
            )
        except StallFallbackError:
            return _mark_stall_fallback(
                run_graph_all_providers(
                    graph_path,
                    graph_json,
                    tensor_infos,
                    config,
                    handle,
                    reporter,
                    _allow_staging=False,
                    _engine_ids=engine_ids,
                )
            )
        reference_outputs = timed_reference.outputs
        if reporter is not None:
            reporter.print_engine_result(timed_reference.result)
        pe_results.append(timed_reference.result)

    # A strict PyTorch selection must run through the timed native path. It
    # never falls back to CPU reference execution after that path fails.
    if ref_provider is not None and reference_outputs is None:
        if (
            config.validation.provider is ReferenceProviderName.PYTORCH
            and config.pytorch_sdpa_backend is not PyTorchSdpaBackendName.DEFAULT
        ):
            assert timed_reference is not None
            reference_error = (
                timed_reference.result.error_message
                or timed_reference.result.skip_reason
                or (
                    f"Requested PyTorch SDPA backend "
                    f"'{config.pytorch_sdpa_backend.value}' is unavailable; "
                    "no fallback is used. The timed PyTorch reference did not "
                    "produce outputs."
                )
            )
        else:
            reference_outputs, reference_error = _compute_reference_outputs_once(
                ref_provider,
                graph_json,
                graph_input_data,
                config,
            )

    for selection in engine_selections:
        engine_id = selection.engine_id
        engine_plugin_path = selection.plugin_path
        engine_name = _resolve_engine_name(engine_id)
        engine_handle = handle
        with Timer() as t:
            if engine_handle is None:
                try:
                    import hipdnn_frontend as hipdnn

                    set_plugin_path(
                        hipdnn,
                        engine_plugin_path,
                        hipdnn.PluginLoadingMode.ABSOLUTE,
                    )
                    engine_handle = hipdnn.Handle()
                except (ImportError, RuntimeError, ValueError, OSError) as e:
                    pe_result = _engine_setup_error_result(
                        provider=engine_name,
                        engine_id=engine_id,
                        plugin_path=engine_plugin_path,
                        config=config,
                        error_message=str(e),
                    )
                    pe_result.elapsed_time_ms = t.elapsed_ms
                    if reporter is not None:
                        reporter.print_engine_start(engine_name)
                        reporter.print_engine_result(pe_result)
                    pe_results.append(pe_result)
                    continue

            if reporter is not None:
                reporter.print_engine_start(engine_name)

            try:
                pe_result = run_single_provider_engine(
                    graph_path=graph_path,
                    graph_json_str=graph_json_str,
                    graph_name=graph_name,
                    tensor_infos=tensor_infos,
                    config=config,
                    handle=engine_handle,
                    provider=engine_name,
                    engine_id=engine_id,
                    plugin_path=engine_plugin_path,
                    reference_outputs=reference_outputs,
                    reference_error=reference_error,
                    input_data=graph_input_data,
                    validation_requested=validation_requested,
                    graph_json=graph_json,
                    analytical_flops=analytical_flops,
                    analytical_flops_partial=analytical_flops_partial,
                    analytical_io_bytes=analytical_io_bytes,
                    allow_staging=_allow_staging,
                )
            except StallFallbackError:
                return _mark_stall_fallback(
                    run_graph_all_providers(
                        graph_path,
                        graph_json,
                        tensor_infos,
                        config,
                        handle,
                        reporter,
                        _allow_staging=False,
                        _engine_ids=engine_ids,
                    )
                )
        pe_result.elapsed_time_ms = t.elapsed_ms
        if engine_plugin_path is not None:
            pe_result.plugin_path = str(engine_plugin_path)
        if reporter is not None:
            reporter.print_engine_result(pe_result)
        pe_results.append(pe_result)

    return GraphResult(
        graph_name=graph_name,
        graph_path=str(graph_path),
        results=pe_results,
        engine_ids=engine_ids,
    )


def run_graph_pytorch_backend(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    reporter: Optional[Reporter] = None,
) -> GraphResult:
    """Run a single graph through the PyTorch executor as the sole engine row.

    The ``--backend pytorch`` counterpart of :func:`run_graph_all_providers`:
    no hipDNN engine discovery, plugins, or reference validation — one
    ``provider="pytorch"`` engine row per graph, so suite console output and
    JSON artifacts share one schema across execution backends and hosts
    (ROCm and CUDA).
    """
    graph_name = graph_json.get("name", graph_path.stem)
    provider = ReferenceProviderName.PYTORCH.value
    rtol, atol = _fallback_tolerance_for_config(config)

    # Unsupported operations are an unsupported-graph signal (mirrors the
    # hipDNN UnsupportedGraphError path), not an execution error. Checked
    # before input generation: the check is a static graph inspection, so
    # unsupported graphs are skipped without allocating inputs (and without
    # misreporting an input-generation failure as an error). A torch import
    # failure here is deliberately ignored: the timed row below will surface
    # it as a proper engine error.
    unsupported: List[str] = []
    try:
        from ..execution import pytorch_ops

        unsupported = sorted(pytorch_ops.get_unsupported_operations(graph_json))
    except Exception:
        pass
    if unsupported:
        msg = f"Graph contains unsupported operations: {unsupported}"
        strict_selection = (
            config.pytorch_sdpa_backend is not PyTorchSdpaBackendName.DEFAULT
        )
        row = ProviderEngineResult(
            provider=provider,
            engine_id=0,
            status="error" if strict_selection else "skipped",
            error_message=msg if strict_selection else None,
            skip_reason=None if strict_selection else msg,
            correctness=CorrectnessResult.failed(
                rtol=rtol, atol=atol, error_message=msg
            ),
        )
        if reporter is not None:
            reporter.print_engine_start(provider)
            reporter.print_engine_result(row)
        return GraphResult(
            graph_name=graph_name,
            graph_path=str(graph_path),
            results=[row],
            engine_ids=[0],
        )

    try:
        graph_input_data = generate_input_data(tensor_infos, config.seed, graph_json)
    except (ValueError, RuntimeError, OSError, TypeError, OverflowError) as e:
        msg = f"Input data generation failed: {e}"
        return GraphResult(
            graph_name=graph_name,
            graph_path=str(graph_path),
            results=[
                ProviderEngineResult(
                    provider=provider,
                    engine_id=0,
                    status="error",
                    error_message=msg,
                    correctness=CorrectnessResult.failed(
                        rtol=rtol, atol=atol, error_message=msg
                    ),
                )
            ],
            engine_ids=[0],
        )

    (
        analytical_flops,
        analytical_flops_partial,
        analytical_io_bytes,
    ) = _compute_graph_analytical_metrics(graph_json, tensor_infos, config, graph_name)

    if reporter is not None:
        reporter.print_engine_start(provider)
    try:
        row = _run_timed_pytorch_row(
            graph_path=graph_path,
            graph_json=graph_json,
            graph_name=graph_name,
            tensor_infos=tensor_infos,
            config=config,
            input_data=graph_input_data,
            analytical_flops=analytical_flops,
            analytical_flops_partial=analytical_flops_partial,
            analytical_io_bytes=analytical_io_bytes,
            role="engine",
        ).result
    except StallFallbackError:
        row = _run_timed_pytorch_row(
            graph_path=graph_path,
            graph_json=graph_json,
            graph_name=graph_name,
            tensor_infos=tensor_infos,
            config=config,
            input_data=graph_input_data,
            analytical_flops=analytical_flops,
            analytical_flops_partial=analytical_flops_partial,
            analytical_io_bytes=analytical_io_bytes,
            role="engine",
            allow_staging=False,
        ).result
        row.warnings = [*(row.warnings or []), _STALL_FALLBACK_WARNING]
    if reporter is not None:
        reporter.print_engine_result(row)

    return GraphResult(
        graph_name=graph_name,
        graph_path=str(graph_path),
        results=[row],
        engine_ids=[0],
    )


def _collect_basic_metrics_post_loop(
    result: ProviderEngineResult,
    cpu_time_probe: Optional[CpuTimeProbe],
    benchmark_iters: int,
    analytical_flops: Optional[int],
    analytical_flops_partial: bool,
    analytical_io_bytes: Optional[int],
) -> None:
    """Populate the basic always-on metric fields on ``result``.

    Called once after the timed loop when ``metrics.tier == "basic"``.
    Pulled out of :func:`run_single_provider_engine` to keep that
    function focused on the timed loop itself; the basic-tier book
    keeping is otherwise just a long sequence of conditionals on
    intermediate results.
    """
    if cpu_time_probe is not None and cpu_time_probe.delta is not None:
        # Per-iter microseconds is the interpretable unit: the loop
        # total is dominated by Python dispatch cost, and per-iter
        # lets users compare directly against the kernel mean (also
        # reported per-iter).
        iters = max(benchmark_iters, 1)
        result.cpu_user_time_per_iter_us = (
            cpu_time_probe.delta.user_time_ms * 1000.0 / iters
        )
        result.cpu_kernel_time_per_iter_us = (
            cpu_time_probe.delta.kernel_time_ms * 1000.0 / iters
        )

    # Analytical totals were computed once at the graph level; propagate
    # them onto every engine's result so JSON consumers don't have to
    # look up across structures.
    result.analytical_flops = analytical_flops
    result.analytical_flops_partial = analytical_flops_partial
    result.analytical_io_bytes = analytical_io_bytes

    # Derived throughputs use the *arithmetic mean* of post-warmup kernel
    # timings — no trimming, no outlier rejection. A single noisy iter
    # (context switch, thermal throttle) skews the headline number; for
    # tighter signal use gpu_kernel_stats.min_ms or p95_ms.
    kernel_mean = (
        result.gpu_kernel_stats.mean_ms if result.gpu_kernel_stats is not None else None
    )
    tflops, gbytes = derive_throughputs(
        analytical_flops, analytical_io_bytes, kernel_mean
    )
    result.derived_tflops_per_s = tflops
    result.derived_gbytes_per_s = gbytes

    # VRAM is sampled here (still inside the BufferManager context, so
    # workspace + I/O buffers are still allocated). This is the only
    # amdsmi field we expose per-engine — see ProviderEngineResult
    # docstring.
    try:
        snap = GpuSmiProbe().snapshot()
        result.vram_used_mb = snap.get("vram_used_mb")
    except Exception as e:
        warn_once("gpu_smi", f"vram snapshot failed: {e}")


# Benchmarking is latched when oracle plans are built. Disable hipDNN disk
# caches so the selected provider variant cannot affect later runs.
# ponytail: process-global guard; concurrent execution needs process isolation.
_EXHAUSTIVE_ENV = {
    "HIPDNN_FORCE_BENCHMARKING": "1",
    "HIPDNN_DISABLE_CACHE": "1",
}


@contextmanager
def _exhaustive_env(enabled: bool):
    """Set provider benchmarking controls and restore the prior environment."""
    if not enabled:
        yield
        return
    previous = {name: os.environ.get(name) for name in _EXHAUSTIVE_ENV}
    os.environ.update(_EXHAUSTIVE_ENV)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_oracle_pass(
    *,
    result: ProviderEngineResult,
    graph_path: Path,
    graph_json_str: str,
    graph_name: str,
    config: SuiteConfig,
    handle: Any,
    engine_id: int,
    bm: Any,
    ootb_executor: Any,
    tensor_infos: list,
    graph_json: Dict[str, Any],
    reference_outputs: Optional[Dict[int, Any]],
) -> None:
    """Tune one engine and attach its result without failing the OOTB row."""
    try:
        with _exhaustive_env(config.oracle_exhaustive):
            # Isolate MIOpen's mutable per-handle solver map from the baseline.
            oracle_handle = type(handle)()
            get_stream = getattr(handle, "get_stream", None)
            set_stream = getattr(oracle_handle, "set_stream", None)
            if callable(get_stream) and callable(set_stream):
                set_stream(get_stream())
            bench_config = BenchmarkConfig(
                graph_path=graph_path,
                warmup_iters=config.warmup_iters,
                benchmark_iters=config.benchmark_iters,
                engine_id=engine_id,
            )
            executor = Executor(
                graph_json_str=graph_json_str,
                config=bench_config,
            )
            executor.prepare(oracle_handle, engine_id=engine_id, for_autotune=True)

            # Restore outputs; inputs remain valid across executions.
            bm.zero_outputs()
            variant_pack = bm.create_variant_pack()

            candidates = executor.autotune(oracle_handle, variant_pack, engine_id)
            eligible_candidates = [
                candidate
                for candidate in candidates
                if not getattr(candidate, "excluded_by_caller", False)
            ]
            successful_candidates = [
                candidate for candidate in eligible_candidates if candidate.succeeded
            ]
            winner = successful_candidates[0]

            # Re-time OOTB after the sweep so both operands share device warmth.
            bm.zero_outputs()
            ootb_executor.warmup(handle, variant_pack)
            baseline_result = ootb_executor.benchmark(
                handle, variant_pack, graph_name=graph_name
            )

            bm.zero_outputs()
            executor.warmup(oracle_handle, variant_pack)
            bench_result = executor.benchmark(
                oracle_handle, variant_pack, graph_name=graph_name
            )

            oracle = OracleResult(
                plan_name=executor.plan_name(oracle_handle) or "",
                compiled_plan_index=int(winner.compiled_plan_index),
                rank=int(winner.rank),
                sweep_min_time_ms=float(winner.min_time_ms),
                compiled_plans_benchmarked=len(successful_candidates),
                compiled_plans_total=len(eligible_candidates),
                compiled_plans_failed=(
                    len(eligible_candidates) - len(successful_candidates)
                ),
                knob_settings=[
                    {"knob_id": str(k.knob_id), "value": k.value}
                    for k in winner.knob_settings
                ],
                cpu_build_time_ms=executor.init_time_ms,
                host_stats=BenchmarkStats.from_timings(bench_result.host_timings),
                gpu_kernel_stats=(
                    BenchmarkStats.from_timings(bench_result.kernel_timings)
                    if bench_result.has_kernel_timings
                    else None
                ),
                warm_baseline_host_stats=BenchmarkStats.from_timings(
                    baseline_result.host_timings
                ),
                warm_baseline_gpu_kernel_stats=(
                    BenchmarkStats.from_timings(baseline_result.kernel_timings)
                    if baseline_result.has_kernel_timings
                    else None
                ),
                exhaustive_requested=config.oracle_exhaustive,
                exhaustive_supported=bool(
                    getattr(winner, "supports_exhaustive", False)
                ),
            )

            # Validate after timing; keep OOTB and tuned verdicts separate.
            if reference_outputs is not None:
                bm.zero_outputs()
                executor.execute_once(oracle_handle, variant_pack)
                oracle.correctness = _check_correctness(
                    bm,
                    tensor_infos,
                    graph_json,
                    reference_outputs,
                    config.validation.provider.value,
                    config,
                )

            result.oracle = oracle
            # A speedup requires two valid operands.
            invalid = [
                side
                for side, verdict in (
                    ("baseline", result.correctness),
                    ("tuned plan", oracle.correctness),
                )
                if verdict is not None and verdict.explicitly_failed
            ]
            if invalid:
                # Preserve timing and verdict evidence, but refuse the ratio.
                result.oracle_delta = None
                warn_once(
                    "oracle_correctness",
                    f"oracle comparison for {graph_name} engine {engine_id} "
                    f"suppressed: {' and '.join(invalid)} failed validation",
                )
            else:
                result.oracle_delta = build_oracle_delta(oracle)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        result.oracle_error = error
        warn_once(
            "oracle",
            f"oracle tuning failed for {graph_name} engine {engine_id}: {error}",
        )


def run_single_provider_engine(
    graph_path: Path,
    graph_json_str: str,
    graph_name: str,
    tensor_infos: list,
    config: SuiteConfig,
    handle: Any,
    provider: str,
    engine_id: int,
    plugin_path: Optional[Path],
    reference_outputs: Optional[Dict[int, ReferenceOutput]],
    reference_error: Optional[str],
    input_data: Dict[int, Any],
    validation_requested: bool,
    graph_json: Dict[str, Any],
    analytical_flops: Optional[int] = None,
    analytical_flops_partial: bool = False,
    analytical_io_bytes: Optional[int] = None,
    allow_staging: bool = True,
) -> ProviderEngineResult:
    """Execute a single engine for a graph (single attempt)."""
    # Initialise the result conservatively as an error and mutate fields as
    # the run progresses; on success, status flips to "success" at the end.
    result = ProviderEngineResult(
        provider=provider,
        engine_id=engine_id,
        status="error",
        engine_version=_resolve_engine_version(handle, engine_id),
    )

    metrics_basic = config.metrics.basic_enabled

    try:
        bench_config = BenchmarkConfig(
            graph_path=graph_path,
            warmup_iters=config.warmup_iters,
            benchmark_iters=config.benchmark_iters,
            engine_id=engine_id,
        )

        executor = Executor(
            graph_json_str=graph_json_str,
            config=bench_config,
        )
        executor.prepare(handle, engine_id=engine_id)
        result.cpu_build_time_ms = executor.init_time_ms
        if metrics_basic:
            result.workspace_bytes = executor.workspace_size

        with BufferManager(
            tensor_infos, device=_hipdnn_buffer_device(reference_outputs)
        ) as bm:
            bm.allocate_all()
            bm.load_input_data(input_data)
            bm.zero_outputs()

            variant_pack = bm.create_variant_pack()
            executor.warmup(handle, variant_pack)

            # Wrap the timed loop with always-on host probes when basic
            # tier is enabled. The probes are designed to be no-cost on
            # the GPU side: CPU-time sampling is two read-only kernel
            # calls before/after the loop, and the amdsmi snapshot fires
            # once after the benchmark returns. Failures inside probes
            # are swallowed and surface as None fields on the result.
            cpu_time_probe = CpuTimeProbe() if metrics_basic else None
            if cpu_time_probe is not None:
                cpu_time_probe.__enter__()
            try:
                bench_result = executor.benchmark(
                    handle,
                    variant_pack,
                    graph_name=graph_name,
                    allow_staging=allow_staging,
                )
            finally:
                if cpu_time_probe is not None:
                    cpu_time_probe.__exit__(None, None, None)

            result.host_stats = BenchmarkStats.from_timings(bench_result.host_timings)
            if bench_result.has_kernel_timings:
                result.gpu_kernel_stats = BenchmarkStats.from_timings(
                    bench_result.kernel_timings
                )

            if metrics_basic:
                _collect_basic_metrics_post_loop(
                    result=result,
                    cpu_time_probe=cpu_time_probe,
                    benchmark_iters=config.benchmark_iters,
                    analytical_flops=analytical_flops,
                    analytical_flops_partial=analytical_flops_partial,
                    analytical_io_bytes=analytical_io_bytes,
                )

            if reference_outputs is not None:
                bm.zero_outputs()
                executor.execute_once(handle, variant_pack)
                result.correctness = _check_correctness(
                    bm,
                    tensor_infos,
                    graph_json,
                    reference_outputs,
                    config.validation.provider.value,
                    config,
                )
            elif validation_requested:
                error_message = reference_error or (
                    f"Reference provider '{config.validation.provider.value}' "
                    f"does not support this graph"
                )
                # User asked for validation but no reference output was usable.
                # Treat as a correctness failure so --validate stays a hard gate.
                result.correctness = _reference_unavailable_correctness(
                    config, error_message
                )
            else:
                rtol, atol = _fallback_tolerance_for_config(config)
                result.correctness = CorrectnessResult(
                    execution_success=True,
                    tolerance_match=None,
                    rtol=rtol,
                    atol=atol,
                    error_message="No reference provider requested",
                )

            # Reaching here means the OOTB timed pass succeeded (any
            # failure raised out of this try block already), so the row
            # is eligible for an oracle comparison.
            if config.oracle_enabled:
                _run_oracle_pass(
                    result=result,
                    graph_path=graph_path,
                    graph_json_str=graph_json_str,
                    graph_name=graph_name,
                    config=config,
                    handle=handle,
                    engine_id=engine_id,
                    bm=bm,
                    ootb_executor=executor,
                    tensor_infos=tensor_infos,
                    graph_json=graph_json,
                    reference_outputs=reference_outputs,
                )

        # BufferManager context has exited — I/O buffers are freed.
        # Drop the executor reference too so its workspace allocation
        # is released before the profiling subprocess fires. Without
        # this, the inner profiling process allocates its own VRAM on
        # top of the parent's still-pinned workspace, which roughly
        # doubles peak VRAM and can OOM on large graphs that fit fine
        # on the headline run.
        del executor

        # Opt-in profiling pass — runs *after* the timed pass and
        # always-on probes (so profiler overhead can't pollute the
        # headline numbers) and *after* the bm/executor teardown above
        # (so the subprocess gets the full VRAM headroom the parent
        # had). The orchestrator handles tool-missing / paranoid /
        # parse failures internally and never raises; we still wrap
        # defensively because anything bubbling out would mark a
        # successful timed run as failed.
        if config.metrics.opt_in_pass_requested:
            try:
                from ..metrics.profiling_orchestrator import (
                    run_profiling_passes,
                )

                extra = run_profiling_passes(
                    graph_path=graph_path,
                    engine_id=engine_id,
                    # `provider` is the human-readable engine name set
                    # by run_graph_all_providers (see
                    # ``_resolve_engine_name``) — use it for the
                    # per-engine output subdirectory so the artifact
                    # path doesn't carry the 19-digit engine ID hash.
                    engine_name=provider,
                    seed=config.seed,
                    warmup_iters=config.warmup_iters,
                    benchmark_iters=config.benchmark_iters,
                    metrics_config=config.metrics,
                    plugin_path=plugin_path,
                )
                if extra:
                    result.extra_metrics = extra
            except Exception as e:
                warn_once(
                    "profiling_orchestrator",
                    f"profiling pass failed: {type(e).__name__}: {e}",
                )

        result.status = "success"
        return result

    except StallFallbackError:
        raise
    except UnsupportedGraphError as e:
        result.cpu_build_time_ms = None
        result.gpu_kernel_stats = None
        result.host_stats = None
        result.status = "skipped"
        result.skip_reason = str(e)
        rtol, atol = _fallback_tolerance_for_config(config)
        result.correctness = CorrectnessResult.failed(
            rtol=rtol, atol=atol, error_message=str(e)
        )
        return result

    except ExecutionError as e:
        error_msg = str(e)
        result.cpu_build_time_ms = None
        result.gpu_kernel_stats = None
        result.host_stats = None
        result.status = "error"
        result.error_message = error_msg
        rtol, atol = _fallback_tolerance_for_config(config)
        result.correctness = CorrectnessResult.failed(
            rtol=rtol, atol=atol, error_message=error_msg
        )
        return result

    except (ValueError, RuntimeError, OSError) as e:
        error_msg = str(e)
        result.cpu_build_time_ms = None
        result.gpu_kernel_stats = None
        result.host_stats = None
        result.status = "error"
        result.error_message = error_msg
        rtol, atol = _fallback_tolerance_for_config(config)
        result.correctness = CorrectnessResult.failed(
            rtol=rtol, atol=atol, error_message=error_msg
        )
        return result
