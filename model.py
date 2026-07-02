import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

class SpatioTemporalModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SpatioTemporalModel, self).__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
        encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        x = x.view(batch_size * time_steps, C, H, W)
        spatial_features = self.backbone(x)
        spatial_features = spatial_features.view(batch_size, time_steps, -1)
        temporal_features = self.transformer(spatial_features)
        out = temporal_features.mean(dim=1)
        return self.fc(out)

def load_paddy_model(model_path, num_classes=10):
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model

def get_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
