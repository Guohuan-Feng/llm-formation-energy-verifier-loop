"""Frozen GPU CHGNet verifier for the finite Stage-4 P1 feasibility pilot.

CHGNet predicts corrected absolute energy per atom. Formation energy here uses
published elemental ground-state references, never the model's fitted AtomRef.
This module does not fit, relax structures, or write scientific result files.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path('C:/Users/28908/Codex/fung-research/v2-diagnosis').resolve()
CACHE = ROOT / 'work' / 'p1-cache'
REF_PATH = CACHE / 'mp_elemental_reference_entries.json.gz'
REF_MD5 = '6e93b6f38d6e27d6c811d3cafb23a070'
REF_SHA256 = '9fd6ef848a91c88840a0d1709b5856d0edb4af22095237e59db52f2acd6bbf79'
REF_URL = 'https://ndownloader.figshare.com/files/40387775?download=1'
CHECKPOINT = '0.3.0'
CHECKPOINT_FILENAME = 'chgnet_0.3.0_e29f68s314m37.pth.tar'
SOURCES = {
    'model_readme': 'https://github.com/CederGroupHub/chgnet',
    'model_license': 'https://github.com/CederGroupHub/chgnet/blob/main/LICENSE',
    'benchmark_model_metadata': 'https://github.com/janosh/matbench-discovery/blob/main/models/chgnet/chgnet-0.3.0.yml',
    'formation_energy_definition': 'https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/energy.py',
    'reference_registry': 'https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/data-files.yml',
    'reference_data': REF_URL,
    'torch_cuda_install': 'https://pytorch.org/get-started/previous-versions/',
}


def _cache_environment() -> None:
    # resolve() follows directory links on Windows before any mkdir or model load.
    resolved = CACHE.resolve()
    if 'onedrive' in str(resolved).casefold() or not resolved.is_relative_to(ROOT):
        raise ValueError(f'Forbidden cache destination: {resolved}')
    resolved.mkdir(parents=True, exist_ok=True)
    for key, child in [('TORCH_HOME', 'torch'), ('MPLCONFIGDIR', 'matplotlib'),
                       ('XDG_CACHE_HOME', 'xdg'), ('TMP', 'tmp'), ('TEMP', 'tmp')]:
        dest = (resolved / child).resolve()
        if 'onedrive' in str(dest).casefold() or not dest.is_relative_to(ROOT):
            raise ValueError(f'Forbidden cache destination: {dest}')
        dest.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(dest)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True


def elemental_references(path: str | Path = REF_PATH) -> dict[str, float]:
    """Return MP minimum-energy elemental phase energies in eV/atom.

    Exact published gzip bytes are checked; all entries have zero correction.
    This reproduces ComputedEntry.energy_per_atom for this specific release.
    """
    raw = Path(path).read_bytes()
    if hashlib.md5(raw).hexdigest() != REF_MD5:
        raise ValueError('Elemental-reference file differs from benchmark registry')
    if hashlib.sha256(raw).hexdigest() != REF_SHA256:
        raise ValueError('Elemental-reference SHA256 mismatch')
    entries = json.loads(gzip.decompress(raw))
    result = {}
    for symbol, entry in entries.items():
        if entry.get('correction', 0) != 0 or entry.get('energy_adjustments', []):
            raise ValueError('Unexpected corrected elemental reference')
        comp = entry['composition']
        if set(comp) != {symbol} or sum(comp.values()) <= 0:
            raise ValueError('Reference entry is not a pure elemental phase')
        result[symbol] = float(entry['energy']) / float(sum(comp.values()))
    if len(result) != 89:
        raise ValueError('Expected all 89 published elemental references')
    return result


def formation_energy_per_atom(structure, energy_per_atom_eV: float,
                             references: dict[str, float] | None = None) -> float:
    """e_form = E_model/N - sum_i x_i mu_i (eV/atom)."""
    references = elemental_references() if references is None else references
    comp = structure.composition.element_composition
    missing = {str(el) for el in comp.elements} - set(references)
    if missing:
        raise ValueError(f'Missing elemental ground-state references: {sorted(missing)}')
    baseline = math.fsum(float(amount) * references[str(el)]
                         for el, amount in comp.items()) / float(comp.num_atoms)
    return float(energy_per_atom_eV) - baseline


def load_model(device: str = 'cuda'):
    """Load the shipped 0.3.0 checkpoint; fail rather than silently use CPU."""
    _cache_environment()
    import torch
    import chgnet
    from chgnet.model.model import CHGNet
    expected_env = (ROOT / 'work' / 'p1-env').resolve()
    if Path(sys.prefix).resolve() != expected_env:
        raise RuntimeError(f'Run verifier in the isolated P1 environment: {expected_env}')
    if not all(Path(mod.__file__).resolve().is_relative_to(expected_env)
               for mod in (torch, chgnet)):
        raise RuntimeError('Verifier packages were imported from outside isolated P1 environment')
    if torch.__version__ != '2.6.0+cu124' or importlib.metadata.version('chgnet') != '0.4.0':
        raise RuntimeError('Verifier package pins differ from frozen torch2.6.0+cu124/chgnet0.4.0')
    if device != 'cuda':
        raise ValueError('This feasibility protocol requires the CUDA verifier')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in isolated P1 environment')
    torch.manual_seed(20261016)
    torch.set_num_threads(8)
    model = CHGNet.load(model_name=CHECKPOINT, use_device=device)
    model.eval()
    if str(next(model.parameters()).device).split(':')[0] != 'cuda':
        raise RuntimeError('CHGNet checkpoint was not placed on CUDA')
    return model


def model_metadata(model) -> dict:
    import torch
    import chgnet
    checkpoint_path = (Path(chgnet.__file__).parent / 'pretrained' / CHECKPOINT /
                       CHECKPOINT_FILENAME)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    versions = {}
    for name in ['chgnet', 'torch', 'pymatgen', 'pymatgen-core', 'numpy', 'ase',
                 'scikit-learn', 'matminer', 'scipy', 'pandas']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        'model': 'CHGNet', 'checkpoint_version': CHECKPOINT,
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'checkpoint_path': str(checkpoint_path),
        'model_device': str(next(model.parameters()).device),
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'torch_cuda_runtime': torch.version.cuda,
        'python': platform.python_version(), 'executable': sys.executable,
        'packages': versions, 'n_parameters': int(model.n_params),
        'graph_converter_algorithm': model.graph_converter.algorithm,
        'quantity_model': 'MP2020-corrected absolute energy in eV/atom',
        'quantity_derived': 'formation energy in eV/atom relative to published MP elemental phases',
        'formula': 'Eform_per_atom = E_model_per_atom - sum(element_atomic_fraction * MP_element_reference_energy_per_atom)',
        'double_correction_applied': False,
        'atomref_used_as_thermodynamic_reference': False,
        'reference_count': 89, 'reference_md5': REF_MD5,
        'reference_sha256': REF_SHA256, 'license': 'Modified BSD; benchmark records code and checkpoint as BSD-3-Clause',
        'training_overlap_caveat': 'MPtrj-pretrained CHGNet and Matbench mp_e_form both derive from Materials Project; this pilot does not establish an unseen-from-pretraining test.',
        'compatibility_caveat': 'CHGNet includes MP2020 compatibility corrections; historical Matbench formation-energy labels may use different MP versions or correction conventions.',
        'sources': SOURCES,
    }


def predict_energies(structures, model=None, *, include_formation: bool = True,
                     references: dict[str, float] | None = None,
                     progress_callback=None) -> list[dict]:
    """Serial, timed, energy-only GPU inference; errors are explicit per item.

    Each wall time includes conversion to graph and CUDA synchronization.
    It excludes model initialization. The caller records run-level load time.
    No energy/formation label enters this function.
    """
    import torch
    model = load_model() if model is None else model
    references = elemental_references() if include_formation and references is None else references
    results = []
    for item_index, structure in enumerate(structures):
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            if not structure.is_ordered:
                raise ValueError('CHGNet requires ordered sites; no silent disorder conversion')
            with torch.no_grad():
                prediction = model.predict_structure(structure, task='e', batch_size=1)
            energy = float(prediction['e'])
            if not math.isfinite(energy):
                raise ValueError('Nonfinite CHGNet energy')
            result = {'status': 'ok', 'energy_per_atom_eV': energy,
                      'formation_energy_per_atom_eV': (
                          formation_energy_per_atom(structure, energy, references)
                          if include_formation else None),
                      'n_sites': len(structure), 'error': None}
        except Exception as exc:
            result = {'status': 'failed', 'energy_per_atom_eV': None,
                      'formation_energy_per_atom_eV': None,
                      'n_sites': len(structure),
                      'error': f'{type(exc).__name__}: {exc}'}
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        torch.cuda.synchronize()
        result['elapsed_seconds'] = time.perf_counter() - started
        results.append(result)
        if progress_callback is not None:
            progress_callback(item_index, dict(result))
    return results


def warmup() -> dict:
    """One synthetic Si crystal checks CUDA and quantity conversion, no dataset."""
    from pymatgen.core import Lattice, Structure
    started = time.perf_counter()
    model = load_model()
    load_seconds = time.perf_counter() - started
    structure = Structure(Lattice.cubic(5.43), ['Si'] * 8,
        [[0, 0, 0], [.25, .25, .25], [0, .5, .5], [.25, .75, .75],
         [.5, 0, .5], [.75, .25, .75], [.5, .5, 0], [.75, .75, .25]])
    first = predict_energies([structure], model=model)[0]
    return {'model_load_seconds': load_seconds, 'metadata': model_metadata(model),
            'silicon_warmup': first}


if __name__ == '__main__':
    print(json.dumps(warmup(), indent=2, allow_nan=False))
