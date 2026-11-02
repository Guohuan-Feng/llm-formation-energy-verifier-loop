"""Finite H2 GPU relaxation workers; import with the frozen P1 Python and -B.

Public API: init_worker(), relax_candidate(candidate), runtime_metadata().
Every candidate runs one FIRE trajectory with at most 300 optimizer steps.
Convergence requires BOTH raw Cartesian atomic forces and the complete native
FrechetCellFilter generalized-force matrix to have maximum row norm <=0.1.
Thus the native transformed atomic component is also constrained (a stronger
condition). The cell diagnostic is the last three rows from that SAME filter,
including its original reference cell and default exp_cell_factor=N. It is not
a freshly reconstructed filter and is not a stress in eV/Angstrom**3.

Only cache directories are written here. Scientific records/structure files,
novelty checks and cross-candidate duplicate filtering belong to the caller.
No scientific LLM is used. Old H1/P1 files are imported read-only.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
H2_DATA = ROOT / 'work' / 'h-data' / 'H2'
CACHE = H2_DATA / 'cache'
FMAX = 0.1
MAX_STEPS = 300
TORCH_THREADS = 8
GPU_TOTAL_TORCH_MEMORY_FRACTION = .70
_RELAXER = None
_CAPTURE = None
_REFERENCES = None
_RUNTIME = None


def _safe_destination(path):
    """Resolve all existing Windows directory links before each write."""
    destination = Path(path).resolve()
    root = ROOT.resolve()
    if ('onedrive' in str(destination).casefold()
            or 'onedrive' in str(root).casefold()
            or not destination.is_relative_to(root)):
        raise ValueError(f'Forbidden H2 write destination: {destination}')
    return destination


def _cache_environment():
    import tempfile

    cache = _safe_destination(CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    for key, child in [
        ('TEMP', 'tmp'), ('TMP', 'tmp'), ('TMPDIR', 'tmp'),
        ('TORCH_HOME', 'torch'), ('MPLCONFIGDIR', 'matplotlib'),
        ('XDG_CACHE_HOME', 'xdg'), ('NUMBA_CACHE_DIR', 'numba'),
        ('HF_HOME', 'huggingface'), ('TRITON_CACHE_DIR', 'triton'),
    ]:
        destination = _safe_destination(cache / child)
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(destination)
    tempfile.tempdir = str(_safe_destination(cache / 'tmp'))
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['OMP_NUM_THREADS'] = str(TORCH_THREADS)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # Any otherwise implicit relative writes remain outside OneDrive.
    os.chdir(cache)


def strict_optimizer_class():
    """Create a process-local FIRE subclass without loading a GPU model."""
    import numpy as np
    from ase.filters import FrechetCellFilter
    from ase.optimize.fire import FIRE

    class StrictCapturingFIRE(FIRE):
        last_result = None

        def _force_metrics(self, gradient):
            if not isinstance(self.atoms, FrechetCellFilter):
                raise TypeError('H2 requires the original FrechetCellFilter')
            native = np.asarray(gradient, dtype=float).reshape(-1, 3)
            underlying = self.atoms.atoms
            raw = np.asarray(underlying.get_forces(), dtype=float)
            if native.shape != (len(underlying) + 3, 3):
                raise ValueError(f'Unexpected Frechet force shape: {native.shape}')
            if raw.shape != (len(underlying), 3):
                raise ValueError(f'Unexpected raw atomic force shape: {raw.shape}')
            if not (np.isfinite(native).all() and np.isfinite(raw).all()):
                raise FloatingPointError('Nonfinite raw/native relaxation force')
            return {
                'atomic_fmax_eV_A': float(np.linalg.norm(raw, axis=1).max()),
                'cell_fmax_eV_A': float(np.linalg.norm(native[-3:], axis=1).max()),
                'filtered_fmax_eV_A': float(np.linalg.norm(native, axis=1).max()),
                'filtered_atomic_fmax_eV_A': float(np.linalg.norm(native[:-3], axis=1).max()),
                'cell_exp_factor': float(self.atoms.exp_cell_factor),
            }

        def gradient_converged(self, gradient):
            metrics = self._force_metrics(gradient)
            native_pass = metrics['filtered_fmax_eV_A'] <= self.fmax
            raw_pass = metrics['atomic_fmax_eV_A'] <= self.fmax
            if native_pass and not raw_pass:
                self.native_only_passes = getattr(self, 'native_only_passes', 0) + 1
            # This hook controls FIRE's loop itself, including the initial
            # gradient check. No restart or extra 300-step run is performed.
            return bool(native_pass and raw_pass)

        def run(self, fmax=FMAX, steps=MAX_STEPS):
            self.native_only_passes = 0
            converged = super().run(fmax=fmax, steps=steps)
            metrics = self._force_metrics(self.optimizable.get_gradient())
            metrics.update(
                converged=bool(converged), steps=int(self.nsteps),
                native_only_passes_continued=int(self.native_only_passes),
            )
            StrictCapturingFIRE.last_result = metrics
            return converged

    return StrictCapturingFIRE


def init_worker(worker_index=None):
    """ProcessPoolExecutor spawn initializer; one independent CUDA model each.

    Uses exactly 8 Torch CPU threads for 1-, 2- and 4-worker measurements.
    Initialization/model-load errors intentionally fail the executor instead of
    silently running on CPU. Per-candidate CUDA OOM is caught separately.
    """
    global _RELAXER, _CAPTURE, _REFERENCES, _RUNTIME
    _cache_environment()
    import multiprocessing
    import torch
    import p1_verifier
    from chgnet.model.dynamics import StructOptimizer

    p1_verifier.CACHE = _safe_destination(CACHE)
    gpu_workers = int(os.environ.get('H2_GPU_WORKERS', '1'))
    if gpu_workers not in (1, 2, 4):
        raise ValueError('Expected1/2/4 concurrent H2 GPU workers')
    # Bound the CUDA caching allocator before model allocation. WDDM can
    # otherwise spill an oversized graph into shared RAM without raising OOM,
    # making a single4-atom dense candidate consume25GB and stall calibration.
    # The headroom covers CUDA contexts and the desktop. This is a resource
    # failure guard, not a new material/energy/geometry acceptance criterion.
    torch.cuda.set_per_process_memory_fraction(GPU_TOTAL_TORCH_MEMORY_FRACTION / gpu_workers, device=0)
    started = time.perf_counter()
    model = p1_verifier.load_model(device='cuda')
    torch.set_num_threads(TORCH_THREADS)
    _REFERENCES = p1_verifier.elemental_references()
    _CAPTURE = strict_optimizer_class()
    _RELAXER = StructOptimizer(
        model=model, optimizer_class=_CAPTURE, use_device='cuda',
        on_isolated_atoms='warn',
    )
    torch.cuda.synchronize()
    _RUNTIME = p1_verifier.model_metadata(model)
    _RUNTIME.update(
        worker_index=worker_index,
        worker_name=multiprocessing.current_process().name,
        worker_pid=os.getpid(), torch_num_threads=torch.get_num_threads(),
        model_load_seconds=time.perf_counter() - started,
        cache_dir=str(_safe_destination(CACHE)),
        optimizer='FIRE', ase_filter='FrechetCellFilter', max_steps=MAX_STEPS,
        fmax_eV_A=FMAX,
        convergence_rule='raw atomic max row norm <=0.1 AND full original FrechetCellFilter max row norm <=0.1',
        cell_force_definition='last 3 rows from original relaxation FrechetCellFilter generalized force matrix, default exp_cell_factor=N',
        scientific_llm_calls=0,
        gpu_concurrent_workers=gpu_workers,
        total_torch_memory_fraction=GPU_TOTAL_TORCH_MEMORY_FRACTION,
        per_worker_torch_memory_fraction=GPU_TOTAL_TORCH_MEMORY_FRACTION / gpu_workers,
    )


def runtime_metadata():
    """Return a detached copy of metadata; never initializes a GPU implicitly."""
    if _RUNTIME is None:
        raise RuntimeError('Call init_worker() in this process first')
    return copy.deepcopy(_RUNTIME)


def _distance_check(structure):
    import numpy as np
    from h1_generation import covalent_radius

    radii = np.asarray([covalent_radius(site.specie.symbol) for site in structure], dtype=float)
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('Nonpositive/nonfinite atomic covalent radius')
    cutoff = max(8., 1.4 * float(radii.max()))
    ii, jj, _, distances = structure.get_neighbor_list(cutoff, exclude_self=True)
    if len(distances) == 0:
        return {
            'min_distance_ratio': None, 'min_distance_A': None, 'distance_pass': True,
            'distance_check_note': 'No neighbors within cutoff, which exceeds all violation distances.',
        }
    ratios = np.asarray(distances) / (radii[ii] + radii[jj])
    if not (np.isfinite(distances).all() and np.isfinite(ratios).all()):
        raise ValueError('Nonfinite neighbor distance or radius ratio')
    minimum = float(np.min(ratios))
    return {
        'min_distance_ratio': minimum, 'min_distance_A': float(np.min(distances)),
        'distance_pass': minimum >= .7, 'distance_check_note': '',
    }


def _validate_geometry(structure):
    import numpy as np

    if (not len(structure) or not structure.is_ordered
            or not np.isfinite(structure.lattice.matrix).all()
            or not np.isfinite(structure.frac_coords).all()
            or not math.isfinite(structure.volume) or structure.volume <= 0):
        raise ValueError('Nonfinite or invalid ordered geometry')


def _synchronize_no_raise(torch):
    """Keep error reporting intact if the original GPU operation failed."""
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def relax_candidate(candidate):
    """Relax one candidate and expose all flags without applying novelty/dedup.

    Input metadata is preserved except initial_structure. Real structures have
    no 20-atom restriction. accepted_physics applies the common convergence,
    distance, relative-volume and formation-energy filters to every item; the
    caller can retain a real converged label while reporting other flag failures.
    A rejected but geometrically valid final structure is returned for auditing.
    """
    if _RELAXER is None or _CAPTURE is None or _REFERENCES is None:
        raise RuntimeError('Call init_worker() before relax_candidate()')
    import torch
    from h1_generation import clean_structure
    from p1_verifier import formation_energy_per_atom

    row = {key: value for key, value in candidate.items() if key != 'initial_structure'}
    row.update({key: None for key in [
        'steps', 'atomic_fmax_eV_A', 'cell_fmax_eV_A',
        'filtered_fmax_eV_A', 'filtered_atomic_fmax_eV_A',
        'cell_exp_factor', 'final_volume_A3', 'volume_ratio',
        'template_volume_ratio', 'min_distance_ratio', 'min_distance_A',
        'energy_eV_per_atom', 'formation_energy_eV_per_atom',
        'native_only_passes_continued',
    ]})
    row.update(
        converged=False, accepted_physics=False, failed_checks=[], error='',
        distance_pass=None, volume_pass=None, energy_pass=None,
        distance_check_note='', worker_pid=os.getpid(),
        torch_cuda_peak_allocated_MiB=None, torch_cuda_peak_reserved_MiB=None,
    )
    final = None
    started = time.perf_counter()
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        structure = clean_structure(candidate['initial_structure'])
        _validate_geometry(structure)
        initial_volume = float(candidate.get('initial_volume_A3', structure.volume))
        if not math.isfinite(initial_volume) or initial_volume <= 0:
            raise ValueError('Invalid initial volume reference')
        # No site-count limit applies here: real HOLDOUT/TRAIN can exceed20.
        _CAPTURE.last_result = None
        result = _RELAXER.relax(
            structure, fmax=FMAX, steps=MAX_STEPS, relax_cell=True,
            ase_filter='FrechetCellFilter', verbose=False, assign_magmoms=False,
            logfile=None, trajectory=None,
        )
        torch.cuda.synchronize()
        if _CAPTURE.last_result is None:
            raise RuntimeError('Optimizer did not expose original filtered convergence')
        row.update(_CAPTURE.last_result)
        if not 0 <= row['steps'] <= MAX_STEPS:
            raise RuntimeError('FIRE exceeded the total step budget')
        final = clean_structure(result['final_structure'])
        _validate_geometry(final)
        if len(final) != len(structure) or final.composition != structure.composition:
            raise ValueError('Relaxation changed sites or composition')
        # Trajectory energies are ASE total eV; divide by the final atom count.
        energy_per_atom = float(result['trajectory'].energies[-1]) / len(final)
        if not math.isfinite(energy_per_atom):
            raise ValueError('Nonfinite relaxed energy')
        formation_energy = formation_energy_per_atom(final, energy_per_atom, _REFERENCES)
        if not math.isfinite(formation_energy):
            raise ValueError('Nonfinite formation energy')
        row['energy_eV_per_atom'] = energy_per_atom
        row['formation_energy_eV_per_atom'] = formation_energy
        row['final_volume_A3'] = float(final.volume)
        row['volume_ratio'] = float(final.volume) / initial_volume
        template_volume = candidate.get('template_volume_A3')
        if template_volume is not None:
            template_volume = float(template_volume)
            if not math.isfinite(template_volume) or template_volume <= 0:
                raise ValueError('Invalid template volume reference')
            row['template_volume_ratio'] = float(final.volume) / template_volume
        row.update(_distance_check(final))
        row['volume_pass'] = .5 <= row['volume_ratio'] <= 1.5
        row['energy_pass'] = -5 <= formation_energy <= 3
        strict_pass = (row['atomic_fmax_eV_A'] <= FMAX
                       and row['cell_fmax_eV_A'] <= FMAX
                       and row['filtered_fmax_eV_A'] <= FMAX)
        if row['converged'] != strict_pass:
            raise RuntimeError('Optimizer convergence flag differs from final raw/native forces')
        failures = []
        if not row['converged']:
            failures.append('not_converged')
        if not row['distance_pass']:
            failures.append('distance')
        if not row['volume_pass']:
            failures.append('volume')
        if not row['energy_pass']:
            failures.append('energy')
        row['failed_checks'] = failures
        row['accepted_physics'] = not failures
    except Exception as exc:
        _synchronize_no_raise(torch)
        row['error'] = f'{type(exc).__name__}: {exc}'
        row['accepted_physics'] = False
        row['failed_checks'] = ['relaxation_error']
        final = None
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            # Reclaim failed graph/trajectory references before subsequent work.
            import gc
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    finally:
        _synchronize_no_raise(torch)
        row['relax_seconds'] = time.perf_counter() - started
        try:
            row['torch_cuda_peak_allocated_MiB'] = float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
            row['torch_cuda_peak_reserved_MiB'] = float(torch.cuda.max_memory_reserved()) / (1024 ** 2)
        except Exception:
            pass
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                row[key] = None
    return {'row': row, 'final_structure': final.as_dict() if final is not None else None}
