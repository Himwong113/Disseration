import torch

# Assuming you have tensors tensor1 and tensor2 with different sizes
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([[7, 8, 9]])

# Concatenate along the second dimension (columns)
result = torch.cat((tensor1, tensor2), dim=0)
print(result)
