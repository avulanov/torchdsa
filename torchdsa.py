import torch
from torch import Tensor

class TorchStack:
    def __init__(self, max_size: int, batch_size: int) -> None:
        self.array = torch.zeros([max_size, batch_size])
        self.max_size = max_size
        self.batch_size = batch_size
        self.index = 0
    def push(self, data: Tensor) -> None:
        assert(data.shape[-1] == self.batch_size)
        self.array[self.index : self.index + 1:, :self.batch_size] = data
    def pop(self,) -> Tensor:
        res = self.array[self.index : self.index + 1, :]
        self.index -= 1
        return res
    def isEmpty(self) -> bool:
        return self.index == 0