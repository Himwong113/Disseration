import torch

# Define batched matrices A, B
batch_size = 3
matrix_dim = 4


A = torch.randn(batch_size, matrix_dim, matrix_dim).cuda()  # Batch of matrices A
B = torch.randn(batch_size, matrix_dim, matrix_dim).cuda()  # Batch of matrices B
print(f'Matrix A ={A}\n Matrix B={B}')

# Perform batched matrix multiplication using einsum
result = torch.einsum('bij,bjk->bik', A, B)

print("Batched Matrix A:")
print(A)

print("\nBatched Matrix B:")
print(B)

print("\nResult of Batched Matrix Multiplication:")
print(result)
