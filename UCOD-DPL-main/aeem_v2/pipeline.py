"""Bounded staged pipeline for overlapping CPU work with inference."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Deque, Iterable, Iterator, Tuple, TypeVar


InputValue = TypeVar("InputValue")
PreparedValue = TypeVar("PreparedValue")
InferenceValue = TypeVar("InferenceValue")
OutputValue = TypeVar("OutputValue")


def ordered_staged_map(
    items: Iterable[InputValue],
    prepare: Callable[[InputValue], PreparedValue],
    infer: Callable[[PreparedValue], InferenceValue],
    finish: Callable[[PreparedValue, InferenceValue], OutputValue],
    finish_workers: int = 2,
    max_pending: int = 4,
) -> Iterator[OutputValue]:
    """Overlap one CPU prefetch stage and bounded CPU finish work around inference."""
    if finish_workers <= 0:
        raise ValueError("finish_workers must be positive")
    if max_pending < finish_workers:
        raise ValueError("max_pending must be at least finish_workers")

    item_iterator = iter(items)
    try:
        first_item = next(item_iterator)
    except StopIteration:
        return

    pending: Deque[Future] = deque()
    with ThreadPoolExecutor(max_workers=1) as prepare_executor, ThreadPoolExecutor(
        max_workers=finish_workers
    ) as finish_executor:
        prepared_future = prepare_executor.submit(prepare, first_item)
        while prepared_future is not None:
            prepared = prepared_future.result()
            try:
                next_item = next(item_iterator)
            except StopIteration:
                prepared_future = None
            else:
                prepared_future = prepare_executor.submit(prepare, next_item)

            inference = infer(prepared)
            pending.append(finish_executor.submit(finish, prepared, inference))
            while pending and (
                pending[0].done() or len(pending) >= max_pending
            ):
                yield pending.popleft().result()

        while pending:
            yield pending.popleft().result()
