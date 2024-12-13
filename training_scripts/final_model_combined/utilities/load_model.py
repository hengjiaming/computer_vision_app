import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EfficientNetBase(nn.Module):
    def __init__(self):
        super(EfficientNetBase, self).__init__()
        self.base_model = efficientnet_v2_s(weights=None)
        self.base_model.classifier = nn.Identity()  # Remove classification head so that we can train regression head

    def forward(self, x):
        return self.base_model(x)
    
class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model
        in_features = 1280

        self.protein_branch = self._create_branch(in_features)
        self.fat_branch = self._create_branch(in_features)
        self.carbs_branch = self._create_branch(in_features)
        self.mass_branch = self._create_branch(in_features)

    def _create_branch(self, in_features):
        return nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.base_model(x)
        protein = self.protein_branch(x)
        fat = self.fat_branch(x)
        carbs = self.carbs_branch(x)
        mass = self.mass_branch(x)
        return {
            'protein': protein,
            'fat': fat,
            'carbs': carbs,
            'mass': mass
        }

def load_depth_model():
    model = torch.hub.load(
        'yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model = model.to(device).float()
    return model.to(device)

def load_custom_model(pth_file_path):
    base_model = EfficientNetBase()
    model = MultiTaskModel(base_model)

    state_dict = torch.load(pth_file_path, map_location=device)
    model.load_state_dict(state_dict)

    # Set model to eval mode
    model.eval()
    model = model.to(device)
    return model