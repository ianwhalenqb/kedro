"""
Specific pool class definition.
"""

import os
from concurrent.futures import Future
from concurrent.futures.process import (
    ProcessPoolExecutor,
    _SafeQueue,
    _global_shutdown,
    _WorkItem,
    _CallItem,
    BrokenProcessPool,
    _RemoteTraceback,
    _ExceptionWithTraceback,
    _sendback_result,
)
from queue import Full, Empty
import multiprocessing as mp
import threading
import weakref
import traceback

_threads_wakeups = weakref.WeakKeyDictionary()


def _process_worker(call_queue, result_queue, initializer, initargs):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        call_queue: A ctx.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A ctx.Queue of _ResultItems that will written
            to by the worker.
        initializer: A callable initializer, or None
        initargs: A tuple of args for the initializer
    """
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical("Exception in initializer:", exc_info=True)
            # The parent will notice that the process stopped and
            # mark the pool broken
            return
    while True:
        call_item = call_queue.get(block=True)
        if call_item is None:
            # Wake up queue management thread
            result_queue.put(os.getpid())
            return
        try:
            r = call_item.fn(*call_item.args, **call_item.kwargs)
        except BaseException as e:
            exc = _ExceptionWithTraceback(e, e.__traceback__)
            _sendback_result(result_queue, call_item.work_id, exception=exc)
        else:
            _sendback_result(result_queue, call_item.work_id, result=r)
            del r

        # Liberate the resource as soon as possible, to avoid holding onto
        # open files or shared memory that is not needed anymore
        del call_item


def _add_call_item_to_queue(pending_work_items, work_ids, call_queue):
    """Override."""
    while True:
        try:
            work_id = work_ids.get(block=False)
        except Empty:
            return

        idx = 0
        for i, items in enumerate(pending_work_items):
            idx = i
            if work_id in items:
                break

        if call_queue[idx].full():
            return

        else:
            work_item = pending_work_items[idx][work_id]

            if work_item.future.set_running_or_notify_cancel():
                call_queue[idx].put(
                    _CallItem(work_id, work_item.fn, work_item.args, work_item.kwargs),
                    block=True,
                )
            else:
                del pending_work_items[work_id]
                continue


def _queue_management_worker(
    executor_reference,
    processes,
    pending_work_items,
    work_ids_queue,
    call_queue,
    result_queue,
    thread_wakeup,
):
    """Overriden queue management worker function."""
    executor = None

    def shutting_down():
        return _global_shutdown or executor is None or executor._shutdown_thread

    def shutdown_worker():
        # This is an upper bound on the number of children alive.
        for q in call_queue:
            q.put_nowait(None)

        # Release the queue's resources as soon as possible.
        for q in call_queue:
            q.close()
        # If .join() is not called on the created processes then
        # some ctx.Queue methods may deadlock on Mac OS X.
        for p in processes.values():
            p.join()

    result_reader = result_queue._reader
    wakeup_reader = thread_wakeup._reader
    readers = [result_reader, wakeup_reader]

    while True:
        _add_call_item_to_queue(pending_work_items, work_ids_queue, call_queue)

        # Wait for a result to be ready in the result_queue while checking
        # that all worker processes are still running, or for a wake up
        # signal send. The wake up signals come either from new tasks being
        # submitted, from the executor being shutdown/gc-ed, or from the
        # shutdown of the python interpreter.
        worker_sentinels = [p.sentinel for p in processes.values()]
        ready = mp.connection.wait(readers + worker_sentinels)

        cause = None
        is_broken = True
        if result_reader in ready:
            try:
                result_item = result_reader.recv()
                is_broken = False
            except BaseException as e:
                cause = traceback.format_exception(type(e), e, e.__traceback__)

        elif wakeup_reader in ready:
            is_broken = False
            result_item = None
        thread_wakeup.clear()
        if is_broken:
            # Mark the process pool broken so that submits fail right now.
            executor = executor_reference()
            if executor is not None:
                executor._broken = (
                    "A child process terminated "
                    "abruptly, the process pool is not "
                    "usable anymore"
                )
                executor._shutdown_thread = True
                executor = None
            bpe = BrokenProcessPool(
                "A process in the process pool was "
                "terminated abruptly while the future was "
                "running or pending."
            )
            if cause is not None:
                bpe.__cause__ = _RemoteTraceback(f"\n'''\n{''.join(cause)}'''")
            # All futures in flight must be marked failed
            for items in pending_work_items:
                for work_id, work_item in items.items():
                    work_item.future.set_exception(bpe)
                    # Delete references to object. See issue16284
                    del work_item
                items.clear()
            # Terminate remaining workers forcibly: the queues or their
            # locks may be in a dirty state and block forever.
            for p in processes.values():
                p.terminate()
            shutdown_worker()
            return
        if isinstance(result_item, int):
            # Clean shutdown of a worker using its PID
            # (avoids marking the executor broken)
            assert shutting_down()
            p = processes.pop(result_item)
            p.join()
            if not processes:
                shutdown_worker()
                return
        elif result_item is not None:
            for items in pending_work_items:
                if result_item.work_id in items:
                    work_item = items.pop(result_item.work_id, None)

            # work_item can be None if another process terminated (see above)
            if work_item is not None:
                if result_item.exception:
                    work_item.future.set_exception(result_item.exception)
                else:
                    work_item.future.set_result(result_item.result)
                # Delete references to object. See issue16284
                del work_item
            # Delete reference to result_item
            del result_item

        # Check whether we should start shutting down.
        executor = executor_reference()
        # No more work items can be added if:
        #   - The interpreter is shutting down OR
        #   - The executor that owns this worker has been collected OR
        #   - The executor that owns this worker has been shutdown.
        if shutting_down():
            try:
                # Flag the executor as shutting down as early as possible if it
                # is not gc-ed yet.
                if executor is not None:
                    executor._shutdown_thread = True
                # Since no new work items can be added, it is safe to shutdown
                # this thread if there are no pending work items.
                if not pending_work_items:
                    shutdown_worker()
                    return
            except Full:
                # This is not a problem: we will eventually be woken up (in
                # result_queue.get()) and be able to send a sentinel again.
                pass
        executor = None


class SpecificProcessPoolExecutor(ProcessPoolExecutor):
    """Hacky way to assign specific jobs to specific processes."""

    def __init__(
        self, max_workers=None, mp_context=None, initializer=None, initargs=()
    ):
        """Constructor."""
        super(SpecificProcessPoolExecutor, self).__init__(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=initializer,
            initargs=initargs,
        )

        work_items = []
        for _ in range(self._max_workers):
            work_items.append({})

        self._pending_work_items = work_items

        queues = []
        for i in range(self._max_workers):
            q = _SafeQueue(
                max_size=max_workers,
                ctx=self._mp_context,
                pending_work_items=self._pending_work_items[i],
            )
            q._ignore_epipe = True
            queues.append(q)

        self._call_queue = queues

    def _start_queue_management_thread(self):
        if self._queue_management_thread is None:
            # When the executor gets garbarge collected, the weakref callback
            # will wake up the queue management thread so that it can terminate
            # if there is no pending work item.
            def weakref_cb(_, thread_wakeup=self._queue_management_thread_wakeup):
                mp.util.debug(
                    "Executor collected: triggering callback for" " QueueManager wakeup"
                )
                thread_wakeup.wakeup()

            # Start the processes so that their sentinels are known.
            self._adjust_process_count()
            self._queue_management_thread = threading.Thread(
                target=_queue_management_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._processes,
                    self._pending_work_items,
                    self._work_ids,
                    self._call_queue,
                    self._result_queue,
                    self._queue_management_thread_wakeup,
                ),
                name="QueueManagerThread",
            )
            self._queue_management_thread.daemon = True
            self._queue_management_thread.start()
            _threads_wakeups[
                self._queue_management_thread
            ] = self._queue_management_thread_wakeup

    def _adjust_process_count(self) -> None:
        """Override adjust process count to send per-process queues."""
        for i in range(self._max_workers):
            p = self._mp_context.Process(
                target=_process_worker,
                args=(
                    self._call_queue[i],
                    self._result_queue,
                    self._initializer,
                    self._initargs,
                ),
            )
            p.start()
            self._processes[p.pid] = p

    def shutdown(self, wait=True):
        """Override shutdown."""
        if self._call_queue is not None:
            for q in self._call_queue:
                q.put_nowait(None)

        with self._shutdown_lock:
            self._shutdown_thread = True

        if self._queue_management_thread:
            # Wake up queue management thread
            self._queue_management_thread_wakeup.wakeup()

            if wait:
                self._queue_management_thread.join()

        # To reduce the risk of opening too many files, remove references to
        # objects that use file descriptors.
        for i, _ in enumerate(self._call_queue):
            self._call_queue[i] = None

        self._queue_management_thread = None
        self._result_queue = None
        self._processes = None

        if self._queue_management_thread_wakeup:
            self._queue_management_thread_wakeup.close()
            self._queue_management_thread_wakeup = None

    def submit(*args, **kwargs):
        """Submit a job to the pool.

        Returns:
            Future object.
        """
        if len(args) >= 2:
            self, fn, *args = args
        elif not args:
            raise TypeError(
                "descriptor 'submit' of 'ProcessPoolExecutor' object "
                "needs an argument"
            )
        elif "fn" in kwargs:
            fn = kwargs.pop("fn")
            self, *args = args
            import warnings

            warnings.warn(
                "Passing 'fn' as keyword argument is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            raise TypeError(
                "submit expected at least 1 positional argument, "
                "got %d" % (len(args) - 1)
            )

        if "__worker_num" not in kwargs:
            raise RuntimeError(
                "Expected worker number in keyword arguments (__worker_num)."
            )

        worker_num = kwargs.pop("__worker_num")

        if worker_num >= self._max_workers or worker_num < 0:
            raise RuntimeError(
                f"Worker number must be in [0, {self._max_workers}], got {worker_num}"
            )

        with self._shutdown_lock:
            if self._broken:
                raise BrokenProcessPool(self._broken)
            if self._shutdown_thread:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if _global_shutdown:
                raise RuntimeError(
                    "cannot schedule new futures after " "interpreter shutdown"
                )

            f = Future()
            w = _WorkItem(f, fn, args, kwargs)

            self._pending_work_items[worker_num][self._queue_count] = w
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            # Wake up queue management thread
            self._queue_management_thread_wakeup.wakeup()

            self._start_queue_management_thread()
            return f
