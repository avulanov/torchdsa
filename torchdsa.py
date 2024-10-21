import torch
from torch import Tensor

class TorchStack:
    def __init__(self, shape: torch.Size, max_size: int) -> None:
        array_shape = torch.Size([max_size]) + shape
        self.array = torch.zeros(array_shape)
        self.max_size = max_size
        self.expected_shape = shape
        self.index = 0
        self.batches = []
        print(array_shape)

    def push(self, data: Tensor) -> None:
        assert((data.shape[1:]) == self.expected_shape)
        batch_size = data.shape[0]
        assert(self.index + batch_size <= self.max_size)
        self.array[self.index : self.index + batch_size, :] = data
        self.batches.append(batch_size)
        self.index += batch_size
        print(self.array)

    def pop(self,) -> Tensor:
        assert(self.index > 0)
        prev_batch_size = self.batches.pop()
        res = self.array[self.index - prev_batch_size : self.index, :]
        self.index -= prev_batch_size
        return res
    
    def isEmpty(self) -> bool:
        return self.index == 0