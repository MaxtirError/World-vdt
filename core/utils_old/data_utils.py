from typing import Optional, Iterator
import math
import torch
from torch.utils.data import Sampler, Dataset, DataLoader, DistributedSampler
import torch.distributed as dist


def cycle(data_loader: DataLoader) -> Iterator:
    while True:
        for data in data_loader:
            if isinstance(data_loader.sampler, ResumableSampler):
                data_loader.sampler.idx += data_loader.batch_size   # type: ignore[attr-defined]
            yield data
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.epoch += 1
        if isinstance(data_loader.sampler, ResumableSampler):
            data_loader.sampler.epoch += 1
            data_loader.sampler.idx = 0
        

class ResumableSampler(Sampler):
    r"""
    Distributed sampler that is resumable.

    Args:
        dataset: Dataset used for sampling.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.epoch = 0
        self.idx = 0
        self.drop_last = drop_last
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.world_size != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.world_size) / self.world_size  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.world_size)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.world_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.world_size]
        
        # resume from previous state
        indices = indices[self.idx:]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict[str, int]:
        return {
            'epoch': self.epoch,
            'idx': self.idx,
        }
        
    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.idx = state_dict['idx']
        