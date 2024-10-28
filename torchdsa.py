import collections
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
    
class TorchQueue:
    def __init__(self, shape: torch.Size, max_size: int) -> None:
        array_shape = torch.Size([max_size]) + shape
        self.array = torch.zeros(array_shape)
        self.max_size = max_size
        self.expected_shape = shape
        self.frontIndex = -1
        self.backIndex = 0
        self.batches = collections.deque()
        self.size = 0
        print(array_shape)
    
    def append(self, data: Tensor) -> None:
        assert((data.shape[1:]) == self.expected_shape)
        batch_size = data.shape[0]
        newBackIndex = (self.backIndex + batch_size) % self.max_size
        assert(self.size + batch_size <= self.max_size)
        if self.backIndex + batch_size <= self.max_size:
            self.array[self.backIndex : self.backIndex + batch_size, :] = data
        else:
            firstHalf = self.max_size - self.backIndex
            secondHalf = newBackIndex
            self.array[self.backIndex : self.backIndex + batch_size, :] = data[0 : firstHalf, :]
            self.array[0 : newBackIndex, :] = data[firstHalf : secondHalf, :]
        self.batches.append(batch_size)
        self.backIndex = newBackIndex
        self.size += batch_size
    
    def appendleft(self, data: Tensor) -> None:
        assert((data.shape[1:]) == self.expected_shape)
        batch_size = data.shape[0]
        newFrontIndex = (self.frontIndex - batch_size) % self.max_size
        assert(newFrontIndex >= self.backIndex)
        if self.frontIndex - batch_size + 1 >= 0:
            self.array[self.frontIndex - batch_size + 1: self.frontIndex + 1, :] = data
        else:
            secondHalf = self.frontIndex + 1
            firstHalf = batch_size - secondHalf
            self.array[self.frontIndex - batch_size + 1: self.frontIndex + 1, :] = data[firstHalf : secondHalf, :]
            self.array[self.max_size - firstHalf : self.max_size, :] = data[0 : firstHalf, :]
        self.batches.appendleft(batch_size)
        self.frontIndex = newFrontIndex
        self.size += batch_size

    def popleft(self,) -> Tensor:
        prev_batch_size = self.batches.popleft()
        self.size -= prev_batch_size
        if self.frontIndex + prev_batch_size < self.max_size:
            res = self.array[self.frontIndex + 1 : self.frontIndex + prev_batch_size + 1, :]
            return res
        firstHalf = self.max_size - self.frontIndex - 1
        secondHalf = prev_batch_size - firstHalf
        res = torch.cat([self.array[self.frontIndex + 1 : self.max_size, :], self.array[0 : secondHalf, :]], dim=0)
        return res

    def pop(self,) -> Tensor:
        prev_batch_size = self.batches.pop()
        self.size -= prev_batch_size
        if self.backIndex - prev_batch_size >= 0:
            res = self.array[self.backIndex - prev_batch_size : self.backIndex, :]
            return res
        firstHalf = self.backIndex
        secondHalf = prev_batch_size - firstHalf
        res = torch.cat([self.array[self.max_size - firstHalf : self.max_size, self.array[0 : secondHalf, :] :]], dim=0)
        return res  