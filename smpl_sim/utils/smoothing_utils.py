import math
import torch 
import torch.nn.functional as F

def gaussian_kernel_1d(size, sigma):
    size = int(size)
    sigma = float(sigma)
    kernel = torch.tensor([math.exp(-((x - size // 2) ** 2) / (2 * sigma ** 2)) for x in range(size)])
    kernel /= kernel.sum()
    return kernel

# Function to apply 1D Gaussian filter
def gaussian_filter_1d_batch(input_data, kernel_size, sigma):
    # Create 1D Gaussian kernel
    kernel = gaussian_kernel_1d(kernel_size, sigma)
    
    # Reshape kernel for convolution
    kernel = kernel.view(1, 1, kernel_size)
    
    # Adjust kernel size for batch and channel
    kernel = kernel.repeat(input_data.size(1), 1, 1)
    
    padding_size = kernel_size // 2
    
    # Apply replicate padding
    padded_input = F.pad(input_data, (padding_size, padding_size), mode='replicate')
 
    # Apply convolution (gaussian filtering)
    filtered_data = F.conv1d(padded_input, kernel, padding="valid", groups=input_data.size(1))
    return filtered_data