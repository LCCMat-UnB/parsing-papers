import threading
import time

from parsing_papers.pipeline import _process_all


def test_process_all_sequential():
    out = _process_all([1, 2], lambda x: x + 1, max_concurrency=1)
    assert out == [2, 3]


def test_process_all_runs_concurrently():
    active = 0
    max_active = 0
    lock = threading.Lock()

    def work(x):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return x * 2

    out = _process_all(list(range(6)), work, max_concurrency=3)
    assert sorted(out) == [0, 2, 4, 6, 8, 10]
    assert max_active > 1  # prova que houve sobreposicao real de execucao
