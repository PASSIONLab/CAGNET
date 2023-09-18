import torch
from sparse_coo_tensor_cpp import nsparse_spgemm

mata_rows = torch.randint(0, 512, (1, 512)).cuda().squeeze()
mata_cols = torch.randint(0, 589380, (1, 512)).cuda().squeeze()
mata_indices = torch.stack((mata_rows, mata_cols))
mata_values = torch.rand(512).float().cuda()

matb_rows = torch.randint(0, 589380, (1, 4694891)).cuda().squeeze()
matb_cols = torch.randint(0, 9430088, (1, 4694891)).cuda().squeeze()
matb_indices = torch.stack((matb_rows, matb_cols))
matb_values = torch.rand(4694891).float().cuda()

mata = torch.sparse_coo_tensor(mata_indices, mata_values, size=torch.Size([512, 589380])).coalesce()
matb = torch.sparse_coo_tensor(matb_indices, matb_values, size=torch.Size([589380, 9430088])).coalesce()

mata_csr = mata.to_sparse_csr()
matb_csr = matb.to_sparse_csr()

matc = nsparse_spgemm(mata_csr.crow_indices().int(), mata_csr.col_indices().int(), mata_csr.values(), matb_csr.crow_indices().int(), matb_csr.col_indices().int(), matb_csr.values(), mata_csr.size(0), mata_csr.size(1), matb_csr.size(1))

c_crow_indices = matc[0]
c_col_indices = matc[1]
c_values = matc[2]

print(c_crow_indices)
print(c_col_indices)
print(c_values)
