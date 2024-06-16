import torch
import numpy as np

# Create sample tensors
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])

# Convert tensors to numpy arrays for easier handling
tensor1_np = tensor1.numpy()
tensor2_np = tensor2.numpy()
print('tensor1_np ', tensor1_np)
# Define the file path
file_path = 'multiple_tensors.csv'

# Save the tensors to a CSV file
np.savetxt(file_path, np.concatenate((tensor1_np, tensor2_np), axis=0), delimiter=',', fmt='%d')

# Load the tensors from the CSV file
loaded_data = np.loadtxt(file_path, delimiter=',')

# Split the loaded data into tensors
loaded_tensor1 = torch.tensor(loaded_data[:2, :])
loaded_tensor2 = torch.tensor(loaded_data[2:, :])

print("Tensor 1:")
print(loaded_tensor1)
print("Tensor 2:")
print(loaded_tensor2)
