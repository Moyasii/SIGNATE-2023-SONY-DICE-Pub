from typing import Optional, List, Iterable, Sequence, Union

from torch.utils.data import Dataset, DataLoader, Sampler

from .utils import worker_init_fn, NullCollate, BalanceSampler


class ExampleDataLoader(DataLoader):
    def __init__(self,
                 dataset: Dataset,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None,
                 balance_sampler: bool = False,
                 balance_sampler_ratio: int = 8,
                 sampler: Optional[Union[Sampler, Iterable]] = None,
                 batch_sampler: Optional[Union[Sampler[Sequence], Iterable[Sequence]]] = None,
                 num_workers: int = 0,
                 collate_stack_columns: List[str] = ['image', 'target'],
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 timeout: float = 0,
                 multiprocessing_context=None,
                 generator=None,
                 *,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = False,
                 pin_memory_device: str = "") -> None:
        if sampler is not None and balance_sampler:
            raise ValueError('sampler and balance_sampler cannot be set at the same time.')
        if balance_sampler:
            sampler = BalanceSampler(dataset, ratio=balance_sampler_ratio)
        collate_fn = NullCollate(collate_stack_columns)
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         sampler=sampler,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=pin_memory,
                         drop_last=drop_last,
                         timeout=timeout,
                         worker_init_fn=worker_init_fn,
                         multiprocessing_context=multiprocessing_context,
                         generator=generator,
                         prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)
