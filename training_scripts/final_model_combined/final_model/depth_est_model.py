
# # import model start
# import torch
# from mmengine import Config
# from PIL import Image
# from torchvision import transforms

# # Load the configuration file
# config_path = "../_data_base_.py"  # Path to the downloaded config
# cfg = Config.fromfile(config_path)
# print(cfg.data_basic.canonical_space)

# # Modify the configuration
# # cfg.dataset.img_size = (384, 384)
# # cfg.dataset.intrinsics = [1480, 1480, 192, 108]
# # cfg.depth_range = (0.1, 10.0)
# # cfg.canonical_space.focal_length = 1480.0

# # Print to verify changes


# image_path = "../images/has_exif.jpeg"
# image = Image.open(image_path).convert('RGB')

# preprocess = transforms.Compose([
#     transforms.ToTensor(),  # Converts [H, W, C] -> [C, H, W]
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
#                          0.229, 0.224, 0.225])  # Standard normalization
# ])

# input_tensor = preprocess(image).unsqueeze(0)  # Shape: [1, 3, H, W]
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)
# input_dict = {'input': input_tensor.to(device)}
# model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
# pred_depth, confidence, output_dict = model.inference(input_dict)
# pred_normal = output_dict['prediction_normal'][:, :3, :, :]
# normal_confidence = output_dict['prediction_normal'][:, 3, :, :]
# # import model end

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms

print("Is GPU available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0))
img_path = "../image/has_exif.jpeg"

image_path = img_path
image = Image.open(image_path).convert('RGB')

preprocess = transforms.Compose([
    transforms.ToTensor(),  # Converts [H, W, C] -> [C, H, W]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])
])

input_tensor = preprocess(image).unsqueeze(0)  # shape: [1, 3, H, W]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

input_dict = {'input': input_tensor.to(device)}

model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
model = model.to(device)
pred_depth, confidence, output_dict = model.inference(input_dict)
pred_normal = output_dict['prediction_normal'][:, :3, :, :]
# see https://arxiv.org/abs/2109.09881 for details
normal_confidence = output_dict['prediction_normal'][:, 3, :, :]

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis('off')
plt.title(f"Actual image", fontsize=12)
plt.show()

# Convert depth to numpy for visualization
pred_depth_np = pred_depth.squeeze().detach().cpu().numpy()
plt.imshow(pred_depth_np, cmap='plasma')
plt.colorbar()
plt.title('Predicted Depth')
plt.show()

depth_values = pred_depth.squeeze().cpu().numpy()  # shape: [H, W]

# Display specific value from centre (i am assuming the object is in the centre of the image)
print("Depth at (150, 150):", depth_values[150, 150], "meters")

# # Save the depth map as a file for calculation(?)
# np.save("absolute_depth_map.npy", depth_values)
