import torch
from torch import nn
import torchvision.models as models


class ResNet18(nn.Module):
    """ResNet18 classifier adapted for grayscale MRI images."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        
        # Load backbone with pre-trained ImageNet weights
        self.backbone = models.resnet18(weights="DEFAULT")
        
        # Modify input layer to accept 1-channel grayscale images
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # Replace classification head
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    model = ResNet18(num_classes=4)
    print(f"Model: ResNet18 | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate dummy input (Batch, Channel, Height, Width)
    dummy_input = torch.randn(1, 1, 224, 224)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    