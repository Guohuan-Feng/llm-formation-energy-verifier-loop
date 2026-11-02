"""CPU-only spawn/process/transport probes; no CHGNet/Torch/GPU imports."""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import threading
import time
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from h2_resilient import ResilientPool, ResilientBudgetStop, _receive_packets


class Budget:
    def __init__(self, seconds=30.):
        self.deadline = time.perf_counter() + seconds

    def remaining(self):
        return 60. + self.deadline - time.perf_counter()


def no_gpu_initializer():
    pass


def cpu_work(candidate, total_attempt):
    sleep_seconds = candidate.get('sleep_by_attempt', {}).get(total_attempt,
                                                            candidate.get('sleep', .01))
    time.sleep(sleep_seconds)
    if total_attempt in candidate.get('exit_attempts', []):
        os._exit(91)
    if candidate.get('fail_always') or total_attempt in candidate.get('fail_attempts', []):
        raise RuntimeError('CUDA error: injected isolated test failure')
    if candidate.get('ordinary_error'):
        raise ValueError('injected ordinary geometry failure')
    return {'row': {
        'id': candidate['id'], 'worker_pid': os.getpid(), 'error': '',
        'converged': True, 'accepted_physics': True, 'steps': 1,
        'relax_seconds': sleep_seconds,
        'energy_eV_per_atom': -1., 'formation_energy_eV_per_atom': -2.,
    }, 'final_structure': {'cpu_test_only': True}}


def candidate(cid, **kwargs):
    return dict(id=cid, initial_structure={'cpu_test_only': cid}, **kwargs)


def malformed_sender(connection):
    # A truncated/malformed pickle frame is isolated to this one result Pipe.
    connection.send_bytes(b'\x80\x04\x95\xff\xff\xff\xff\x00')
    connection.close()


def healthy_sender(connection):
    connection.send({'kind': 'result', 'slot': 1, 'generation': 1, 'pid': os.getpid(),
                     'token': 1, 'result': {'row': {'id': 'healthy'}}})
    connection.close()


class RecoveryTests(unittest.TestCase):
    def pool(self, workers, budget=None, events=None):
        return ResilientPool(
            workers, budget or Budget(), lambda message: None,
            event_callback=events.append if events is not None else None,
            initializer=no_gpu_initializer, worker_function=cpu_work,
            poll_seconds=.01)

    def test_isolation_order_and_persistence(self):
        events = []
        cases = [candidate('retry', fail_attempts=[1]),
                 candidate('healthy_slow', sleep=.4), candidate('healthy_fast'),
                 candidate('ordinary', ordinary_error=True), candidate('tail')]
        with self.pool(4, events=events) as pool:
            records = pool.run_batch(cases)
            self.assertEqual([r['row']['id'] for r in records], [c['id'] for c in cases])
            recovered = records[0]['row']
            self.assertEqual(recovered['attempt_count'], 2)
            self.assertEqual(len({h['worker_pid'] for h in recovered['attempt_history']}), 2)
            self.assertEqual(records[3]['row']['attempts_this_run'], 1)
            self.assertFalse(records[3]['row']['resource_failure'])
            restarted = [e for e in events if e['event'] == 'worker_restarted']
            self.assertEqual(len(restarted), 1)
            self.assertEqual(restarted[0]['slot'], 0)
            self.assertEqual(pool.stats()['worker_restarts'], 1)
            pool.run_batch([candidate('batch2')], concurrency_limit=1)
            self.assertEqual(pool.stats()['worker_restarts'], 1)
            self.assertEqual(pool.stats()['initial_processes_spawned'], 4)
            self.assertEqual(pool.stats()['terminal_candidates'], 6)

    def test_two_total_failures_and_historical_limit(self):
        with self.pool(2) as pool:
            rows = [r['row'] for r in pool.run_batch([
                candidate('new', fail_always=True),
                candidate('historical', fail_always=True, _prior_cuda_failures=1),
                candidate('healthy'),
            ])]
            self.assertEqual(rows[0]['attempts_this_run'], 2)
            self.assertEqual(rows[1]['attempts_this_run'], 1)
            self.assertEqual(rows[0]['attempt_count'], 2)
            self.assertEqual(rows[1]['attempt_count'], 2)
            for row in rows[:2]:
                self.assertEqual(row['rejection_reason'], 'resource_failure')
                self.assertIsNone(row['energy_eV_per_atom'])
                self.assertIsNone(row['formation_energy_eV_per_atom'])
            self.assertEqual(pool.stats()['resource_failed_candidates'], 2)
            self.assertTrue(rows[2]['accepted_physics'])

    def test_hard_death_recovers_only_one_worker(self):
        events = []
        with self.pool(2, events=events) as pool:
            rows = [r['row'] for r in pool.run_batch([
                candidate('hardexit', exit_attempts=[1]),
                candidate('healthy', sleep=.3),
            ])]
            self.assertEqual(rows[0]['attempt_count'], 2)
            self.assertFalse(rows[0]['error'])
            self.assertEqual(pool.stats()['worker_restarts'], 1)
            self.assertEqual(pool.stats()['worker_deaths'], 1)
            self.assertEqual(len([e for e in events if e['event'] == 'worker_restarted']), 1)

    def test_retry_is_pinned_to_fresh_original_slot(self):
        with self.pool(2) as pool:
            rows = [r['row'] for r in pool.run_batch([
                candidate('healthy_lower_slot'),
                candidate('fault_higher_slot', fail_attempts=[1], sleep=.15),
            ])]
            history = rows[1]['attempt_history']
            self.assertEqual([h['slot'] for h in history], [1, 1])
            self.assertEqual([h['generation'] for h in history], [1, 2])
            self.assertNotEqual(history[0]['worker_pid'], history[1]['worker_pid'])
            self.assertEqual(history[1]['retry_placement'], 'fresh generation of original faulted slot')

    def test_sixth_restart_drains_healthy_busy_workers(self):
        events = []
        cases = [candidate('fail0', fail_always=True, _prior_cuda_failures=1),
                 candidate('fail1', fail_always=True, _prior_cuda_failures=1),
                 candidate('healthy2', sleep=4.), candidate('healthy3', sleep=4.)]
        cases += [candidate(f'fail{i}', fail_always=True, _prior_cuda_failures=1)
                  for i in range(2, 6)]
        cases += [candidate('tail0'), candidate('tail1')]
        with self.pool(4, events=events) as pool:
            records = pool.run_batch(cases)
            self.assertEqual([r['row']['id'] for r in records], [c['id'] for c in cases])
            self.assertEqual(pool.stats()['worker_restarts'], 6)
            self.assertEqual(pool.stats()['target_workers'], 2)
            self.assertTrue(pool.stats()['degraded_to_two'])
            self.assertEqual(pool.stats()['healthy_busy_worker_terminations'], 0)
            self.assertEqual(pool.stats()['healthy_workers_retired_after_drain'], 2)
            for i in (2, 3):
                self.assertTrue(records[i]['row']['accepted_physics'])
                self.assertEqual(records[i]['row']['attempts_this_run'], 1)
            degraded = [e for e in events if e['event'] == 'concurrency_degraded']
            self.assertEqual(len(degraded), 1)
            self.assertEqual(len(degraded[0]['busy_excess_pids']), 2)
            self.assertTrue(all(e['slot'] in (0, 1) for e in events
                                if e['event'] == 'worker_restarted'))

    def test_busy_retiring_worker_death_after_degradation_is_not_lost(self):
        events = []
        cases = [candidate('fail0', fail_always=True, _prior_cuda_failures=1),
                 candidate('fail1', fail_always=True, _prior_cuda_failures=1),
                 candidate('retiring_hardexit', exit_attempts=[1],
                           sleep_by_attempt={1: 3., 2: .01}),
                 candidate('healthy_retiring', sleep=3.)]
        cases += [candidate(f'fail{i}', fail_always=True, _prior_cuda_failures=1)
                  for i in range(2, 6)]
        with self.pool(4, events=events) as pool:
            rows = [r['row'] for r in pool.run_batch(cases)]
            self.assertEqual(rows[2]['attempt_count'], 2)
            self.assertTrue(rows[2]['accepted_physics'])
            self.assertEqual(rows[2]['attempt_history'][0]['slot'], 2)
            self.assertIn(rows[2]['attempt_history'][1]['slot'], (0, 1))
            self.assertIn('unaffected surviving worker', rows[2]['attempt_history'][1]['retry_placement'])
            self.assertEqual(pool.stats()['target_workers'], 2)
            self.assertEqual(pool.stats()['worker_deaths'], 1)
            self.assertEqual(pool.stats()['worker_restarts'], 7)
            sequence = [e['event'] for e in events]
            degraded_index = sequence.index('concurrency_degraded')
            death_index = next(i for i, event in enumerate(events)
                               if event['event'] == 'candidate_attempt_failed'
                               and event['candidate_id'] == 'retiring_hardexit')
            self.assertLess(degraded_index, death_index)
            restarted_retiring = [e for e in events if e['event'] == 'worker_restarted'
                                  and e['slot'] == 2]
            self.assertEqual(len(restarted_retiring), 1)
            self.assertFalse(restarted_retiring[0]['replacement_spawned'])

    def test_budget_cancels_owned_workers_and_exposes_ordered_partial(self):
        pool = self.pool(2, budget=Budget(1.5))
        started = time.perf_counter()
        with self.assertRaises(ResilientBudgetStop) as caught:
            with pool:
                pool.run_batch([candidate('fast'), candidate('slow', sleep=8.)])
        self.assertEqual([r['row']['id'] for r in caught.exception.completed_partial], ['fast'])
        self.assertTrue(pool.closed)
        self.assertFalse(pool.stats()['live_worker_pids'])
        self.assertLess(time.perf_counter() - started, 5.)

    def test_malformed_pipe_does_not_corrupt_healthy_pipe(self):
        context = mp.get_context('spawn')
        parent_queue = queue.Queue()
        children, readers = [], []
        for slot, function in enumerate((malformed_sender, healthy_sender)):
            read_connection, write_connection = context.Pipe(duplex=False)
            process = context.Process(target=function, args=(write_connection,))
            process.start()
            write_connection.close()
            reader = threading.Thread(target=_receive_packets,
                                      args=(read_connection, slot, 1, process.pid, parent_queue),
                                      daemon=True)
            reader.start()
            children.append(process)
            readers.append(reader)
        packets = []
        until = time.perf_counter() + 3.
        while time.perf_counter() < until:
            try:
                packets.append(parent_queue.get(timeout=.1))
            except queue.Empty:
                pass
            if (any(p['kind'] == 'pipe_error' and p['slot'] == 0 for p in packets)
                    and any(p['kind'] == 'result' and p['slot'] == 1 for p in packets)):
                break
        for child in children:
            child.join(timeout=2.)
            self.assertFalse(child.is_alive())
        for reader in readers:
            reader.join(timeout=1.)
        self.assertTrue(any(p['kind'] == 'pipe_error' and p['slot'] == 0 for p in packets))
        self.assertTrue(any(p['kind'] == 'result' and p['slot'] == 1 for p in packets))


if __name__ == '__main__':
    mp.freeze_support()
    unittest.main(verbosity=2)
