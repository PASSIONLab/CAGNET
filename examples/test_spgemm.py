import torch
from sparse_coo_tensor_cpp import spgemm_gpu

x = torch.sparse_coo_tensor(indices=torch.cuda.LongTensor([[0, 1, 1], [1, 0, 0]]), values=torch.cuda.DoubleTensor([2, 2, 3]))
y = torch.sparse_coo_tensor(indices=torch.cuda.LongTensor([[0, 1], [1, 0]]), values=torch.cuda.DoubleTensor([3, 3]))
print(f"{spgemm_gpu(x._indices()[0,:].int(), x._indices()[1,:].int(), x._values().float(), y._indices()[0,:].int(), y._indices()[1,:].int(), y._values().float(), 2, 2, 2)}")
