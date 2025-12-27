import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Check for image path argument
if len(sys.argv) < 2:
    print("Usage: python real_image_edges.py <image_path>")
    print("Example: python real_image_edges.py test_photo.jpg")
    sys.exit(1)

image_path = sys.argv[1]

def rgb_to_grayscale(image):
    """Convert RGB image to grayscale"""
    if len(image.shape) == 3:
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])
    return image

def convolve2d(image, kernel):
    """Apply 2D convolution"""
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image)
    
    for i in range(img_height):
        for j in range(img_width):
            region = padded[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    return output

def sobel_edge_detection(image):
    """Detect edges using Sobel operator"""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    gradient_x = convolve2d(image, sobel_x)
    gradient_y = convolve2d(image, sobel_y)
    
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude = (magnitude / magnitude.max()) * 255
    
    return magnitude.astype(np.uint8)

# Load image
print(f"Loading image: {image_path}")
try:
    img = Image.open(image_path)
except FileNotFoundError:
    print(f"Error: Could not find image '{image_path}'")
    sys.exit(1)

# Resize if too large
max_size = 800
if max(img.size) > max_size:
    ratio = max_size / max(img.size)
    new_size = tuple([int(x * ratio) for x in img.size])
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    print(f"Resized to: {img.size}")

# Convert to grayscale
img_array = np.array(img)
gray_image = rgb_to_grayscale(img_array)

print(f"Processing {gray_image.shape[0]}x{gray_image.shape[1]} image...")

# Detect edges
edges = sobel_edge_detection(gray_image)

# Try multiple thresholds
thresholds = [20, 40, 60]

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(gray_image, cmap='gray')
axes[0, 1].set_title('Grayscale')
axes[0, 1].axis('off')

axes[0, 2].imshow(edges, cmap='hot')
axes[0, 2].set_title('Edge Magnitude')
axes[0, 2].axis('off')

for i, thresh in enumerate(thresholds):
    binary = np.zeros_like(edges)
    binary[edges > thresh] = 255
    
    axes[1, i].imshow(binary, cmap='gray')
    axes[1, i].set_title(f'Threshold = {thresh}')
    axes[1, i].axis('off')
    
    edge_percentage = 100 * np.sum(binary > 0) / binary.size
    axes[1, i].text(10, 30, f'{edge_percentage:.1f}% edges', 
                    color='red', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('real_image_edges.png', dpi=150)

print("\nEdge Detection Complete!")
print(f"Saved: real_image_edges.png")
