"""Pure-CPU H2 resource admission; no material selection or file writes.

group_candidates(candidates, selected_workers) returns nonempty
(actual_workers, candidates) groups in selected-workers, 2-workers, 1-worker
order, with duplicate worker counts removed. Input order is preserved within
each group. The caller must restore its original overall candidate order before
retention or any other scientific selection.

The workload proxy is sum_i degree_i * (degree_i - 1), using periodic directed
neighbors within 3 Angstrom, the CHGNet bond-graph cutoff. More than 160 sites
or 30000 angle pairs admits one worker; more than 80 sites or 12000 angle pairs
admits at most two. Otherwise the measured selected worker count is used.

These rules change compute concurrency only. Every input candidate is returned
exactly once and still receives the same relaxation, convergence and physical
checks. A resource failure must be reported by the relaxation caller; admission
never declares an initial geometry physically rejected or accepted.

At volume/atom <0.5 Angstrom**3, neighbor allocation is skipped and a flagged
large estimate forces one worker. Invalid geometry/counting errors likewise
use conservative single-worker admission so the normal relaxation can expose
the actual candidate error. The CUDA allocator cap belongs to each worker.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
import sys

sys.dont_write_bytecode = True
VERSION = 'H2-resource-admission-v1'
CUTOFF_A = 3.0
ONE_WORKER_SITES = 160
ONE_WORKER_ANGLES = 30000
TWO_WORKER_SITES = 80
TWO_WORKER_ANGLES = 12000
DENSE_VOLUME_PER_ATOM_A3 = 0.5
LARGE_ANGLE_ESTIMATE = 10 ** 18
_CACHE = {}


def _cache_key(candidate):
    """Key by ID AND frozen initial structure, preventing stale-ID reuse."""
    raw = candidate.get('initial_structure')
    encoded = json.dumps(raw, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=True).encode('utf-8')
    structure_hash = hashlib.sha256(encoded).hexdigest()
    return str(candidate.get('id', '')), structure_hash


def admission_metadata(candidate):
    """Return detached, cached CPU workload diagnostics for one candidate.

    angle_pairs is exact when angle_count_exact is True. A dense or invalid
    geometry uses the explicitly labeled LARGE_ANGLE_ESTIMATE sentinel rather
    than allocating a potentially huge periodic neighbor list.
    """
    key = _cache_key(candidate)
    if key in _CACHE:
        return copy.deepcopy(_CACHE[key])
    metadata = {
        'admission_version': VERSION, 'candidate_id': key[0],
        'initial_structure_sha256': key[1], 'neighbor_cutoff_A': CUTOFF_A,
        'n_sites': None, 'initial_volume_A3': None,
        'initial_volume_per_atom_A3': None,
        'directed_neighbors_3A': None, 'max_degree_3A': None,
        'angle_pairs': LARGE_ANGLE_ESTIMATE, 'angle_count_exact': False,
        'dense_geometry_shortcut': False, 'admission_error': '',
        'resource_only': True,
    }
    try:
        import numpy as np

        raw = candidate['initial_structure']
        n_sites = len(raw['sites'])
        metadata['n_sites'] = n_sites
        matrix = np.asarray(raw['lattice']['matrix'], dtype=float)
        coordinates = np.asarray([site['abc'] for site in raw['sites']], dtype=float)
        if (n_sites <= 0 or matrix.shape != (3, 3)
                or coordinates.shape != (n_sites, 3)
                or not np.isfinite(matrix).all()
                or not np.isfinite(coordinates).all()):
            raise ValueError('Invalid initial geometry for resource estimation')
        volume = abs(float(np.linalg.det(matrix)))
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError('Nonpositive/nonfinite initial cell volume')
        per_atom = volume / n_sites
        metadata.update(initial_volume_A3=volume,
                        initial_volume_per_atom_A3=per_atom)
        if per_atom < DENSE_VOLUME_PER_ATOM_A3:
            metadata['dense_geometry_shortcut'] = True
        else:
            from pymatgen.core import Structure

            # Only geometry enters this workload measure. Placeholder H sites
            # avoid reading oxidation states, properties or energy labels.
            geometry = Structure(matrix, ['H'] * n_sites, coordinates,
                                 to_unit_cell=True)
            centers, _, _, distances = geometry.get_neighbor_list(
                CUTOFF_A, exclude_self=True)
            if not np.isfinite(distances).all():
                raise ValueError('Nonfinite periodic neighbor distances')
            degree = Counter(map(int, centers))
            angles = sum(int(d) * (int(d) - 1) for d in degree.values())
            metadata.update(
                directed_neighbors_3A=int(len(centers)),
                max_degree_3A=max(degree.values(), default=0),
                angle_pairs=int(angles), angle_count_exact=True,
            )
    except Exception as exc:
        # This is compute admission, so preserve the candidate and route it
        # conservatively instead of turning this CPU probe into a filter.
        metadata['admission_error'] = f'{type(exc).__name__}: {exc}'
    _CACHE[key] = copy.deepcopy(metadata)
    return metadata


def _actual_workers(metadata, selected_workers):
    n_sites = metadata['n_sites']
    if (metadata['admission_error'] or n_sites is None
            or n_sites > ONE_WORKER_SITES
            or metadata['angle_pairs'] > ONE_WORKER_ANGLES):
        return 1
    if (n_sites > TWO_WORKER_SITES
            or metadata['angle_pairs'] > TWO_WORKER_ANGLES):
        return min(selected_workers, 2)
    return selected_workers


def group_candidates(candidates, selected_workers):
    """Group all candidates by concurrency, preserving order within groups.

    The input objects themselves are returned without copying or mutation.
    The selected setting is a positive integer, normally 1, 2 or 4 from the
    hardware benchmark. No resulting group exceeds that concurrency.
    """
    if (isinstance(selected_workers, bool)
            or not isinstance(selected_workers, int) or selected_workers < 1):
        raise ValueError('selected_workers must be a positive integer')
    worker_order = list(dict.fromkeys([selected_workers,
                                     min(selected_workers, 2), 1]))
    groups = {workers: [] for workers in worker_order}
    for candidate in candidates:
        actual = _actual_workers(admission_metadata(candidate), selected_workers)
        groups[actual].append(candidate)
    return [(workers, groups[workers]) for workers in worker_order if groups[workers]]


def protocol_metadata():
    """Constant protocol metadata for runner manifests and reports."""
    return {
        'version': VERSION, 'resource_only': True,
        'neighbor_cutoff_A': CUTOFF_A,
        'angle_pairs_definition': 'sum_i degree_3A(i) * (degree_3A(i)-1)',
        'one_worker_if_sites_greater_than': ONE_WORKER_SITES,
        'one_worker_if_angle_pairs_greater_than': ONE_WORKER_ANGLES,
        'at_most_two_workers_if_sites_greater_than': TWO_WORKER_SITES,
        'at_most_two_workers_if_angle_pairs_greater_than': TWO_WORKER_ANGLES,
        'skip_neighbor_allocation_if_volume_per_atom_less_than_A3': DENSE_VOLUME_PER_ATOM_A3,
        'dense_geometry_angle_estimate': LARGE_ANGLE_ESTIMATE,
        'cache_key': 'candidate ID plus SHA256 of complete initial_structure JSON',
        'ordering': 'selected concurrency then at most2 then1; preserve within-group order; caller restores original order before retention',
        'material_filter_changes': False,
        'cuda_allocator_cap': 'Provided by worker initializer; not configured by this CPU-only module',
    }
