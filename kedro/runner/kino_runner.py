"""
Kedro intelligent node order runner definition.
"""

import pandas as pd
from kedro.runner.parallel_runner import (
    ParallelRunner,
    _run_node_synchronization,
)
from kedro.kino.specific_pool import SpecificProcessPoolExecutor

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import chain
from multiprocessing.managers import BaseProxy, SyncManager  # type: ignore
from typing import Set

from pluggy import PluginManager

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node


START_PATH = "./start_times.csv"
END_PATH = "./end_times.csv"
TIME_NAMES = ["node_name", "session_id", "time"]
TIME_INDEX = ["node_name", "session_id"]


class KinoRunner(ParallelRunner):
    """Kedro intelligent node ordering runner.
    """

    def __init__(self, max_workers: int = None, is_async: bool = False):
        """Constructor."""
        super(KinoRunner, self).__init__(max_workers, is_async)

        start = pd.read_csv(START_PATH, names=TIME_NAMES, index_col=TIME_INDEX)
        end = pd.read_csv(END_PATH, names=TIME_NAMES, index_col=TIME_INDEX)

        runtimes = (end - start).groupby("node_name").mean()

        self.runtimes = {row[0]: row[1].item() for row in runtimes.iterrows()}

    def _run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager,
        session_id: str = None,
    ) -> None:
        """Override."""
        nodes = pipeline.nodes
        self._validate_catalog(catalog, pipeline)
        self._validate_nodes(nodes)

        load_counts = Counter(chain.from_iterable(n.inputs for n in nodes))
        node_dependencies = pipeline.node_dependencies
        todo_nodes = set(node_dependencies.keys())
        done_nodes = set()  # type: Set[Node]
        futures = set()
        done = None
        max_workers = self._get_required_workers_count(pipeline)

        # TODO: Get node -> worker mapping from optimization.
        # call get_machines(pipeline, self.runtime) then maybe invert the keys/items.
        node_workers = {
            "job_1": 1,
            "job_2": 0,
            "job_3": 0,
            "job_4": 0,
            "job_5": 1,
            "job_6": 0,
            "job_7": 2,
        }
        for node in nodes:
            assert node.name in node_workers

        from kedro.framework.project import LOGGING, PACKAGE_NAME

        with SpecificProcessPoolExecutor(max_workers=max_workers) as pool:
            while True:
                ready = {n for n in todo_nodes if node_dependencies[n] <= done_nodes}
                todo_nodes -= ready
                for node in ready:
                    futures.add(
                        pool.submit(
                            _run_node_synchronization,
                            node,
                            catalog,
                            self._is_async,
                            session_id,
                            package_name=PACKAGE_NAME,
                            conf_logging=LOGGING,
                            __worker_num=node_workers[node.name]
                        )
                    )
                if not futures:
                    if todo_nodes:
                        debug_data = {
                            "todo_nodes": todo_nodes,
                            "done_nodes": done_nodes,
                            "ready_nodes": ready,
                            "done_futures": done,
                        }
                        debug_data_str = "\n".join(
                            f"{k} = {v}" for k, v in debug_data.items()
                        )
                        raise RuntimeError(
                            f"Unable to schedule new tasks although some nodes "
                            f"have not been run:\n{debug_data_str}"
                        )
                    break  # pragma: no cover
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        node = future.result()
                    except Exception:
                        self._suggest_resume_scenario(pipeline, done_nodes)
                        raise
                    done_nodes.add(node)

                    # Decrement load counts, and release any datasets we
                    # have finished with. This is particularly important
                    # for the shared, default datasets we created above.
                    for data_set in node.inputs:
                        load_counts[data_set] -= 1
                        if (
                            load_counts[data_set] < 1
                            and data_set not in pipeline.inputs()
                        ):
                            catalog.release(data_set)
                    for data_set in node.outputs:
                        if (
                            load_counts[data_set] < 1
                            and data_set not in pipeline.outputs()
                        ):
                            catalog.release(data_set)
