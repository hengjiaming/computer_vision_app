
# import model start
import torch
from mmengine import Config
from PIL import Image
from torchvision import transforms

# Load the configuration file
config_path = "../_data_base_.py"  # Path to the downloaded config
cfg = Config.fromfile(config_path)
print(cfg.data_basic.canonical_space)

# Modify the configuration
# cfg.dataset.img_size = (384, 384)
# cfg.dataset.intrinsics = [1480, 1480, 192, 108]
# cfg.depth_range = (0.1, 10.0)
# cfg.canonical_space.focal_length = 1480.0

# Print to verify changes


image_path = "../images/has_exif.jpeg"
image = Image.open(image_path).convert('RGB')

preprocess = transforms.Compose([
    transforms.ToTensor(),  # Converts [H, W, C] -> [C, H, W]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # Standard normalization
])

input_tensor = preprocess(image).unsqueeze(0)  # Shape: [1, 3, H, W]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
input_dict = {'input': input_tensor.to(device)}
model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
pred_depth, confidence, output_dict = model.inference(input_dict)
pred_normal = output_dict['prediction_normal'][:, :3, :, :]
normal_confidence = output_dict['prediction_normal'][:, 3, :, :]
# import model end
