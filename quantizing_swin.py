import torch
import timm
from tqdm import tqdm
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.model_preparer import prepare_model
from aimet_torch.model_validator.model_validator import ModelValidator
import aimet_torch.quantsim as qsim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from timm.models.layers.drop import DropPath
from timm.models.layers.adaptive_avgmax_pool import FastAdaptiveAvgPool2d

# Load pre-trained model
model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
model.eval()

# Quantization wrapper classes
@QuantizationMixin.implements(DropPath)
class QuantizedDropPath(QuantizationMixin, DropPath):
    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, x):
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)
        with self._patch_quantized_parameters():
            ret = super().forward(x)
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)
        return ret

@QuantizationMixin.implements(FastAdaptiveAvgPool2d)
class QuantizedFastAdaptiveAvgPool(QuantizationMixin, FastAdaptiveAvgPool2d):
    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, x):
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)
        with self._patch_quantized_parameters():
            ret = super().forward(x)
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)
        return ret

# Validation function with optional subset testing
def validate_pytorch(model, dataloader, device, subset=False):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Validating PyTorch Model", leave=False)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred_top1 = outputs.topk(1, dim=1)
            _, pred_top5 = outputs.topk(5, dim=1)

            top1_correct += (pred_top1.squeeze() == labels).sum().item()
            top5_correct += sum([labels[i] in pred_top5[i] for i in range(len(labels))])
            total += labels.size(0)

            if subset and total >= 100:  # Validate only the first 100 samples for subset check
                break

    top1_acc = 100 * top1_correct / total
    top5_acc = 100 * top5_correct / total
    return top1_acc, top5_acc

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder("/media/bmw/datasets/imagenet-1k/val", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Set device
device = torch.device("cpu")

# Run initial subset validation check
print("\nPerforming subset validation check before full evaluation...")
subset_top1, subset_top5 = validate_pytorch(model, dataloader, device, subset=True)
print(f"Subset Check - Top-1 Accuracy: {subset_top1:.2f}%, Top-5 Accuracy: {subset_top5:.2f}%")

# Full validation on original model
top1_acc, top5_acc = validate_pytorch(model, dataloader, device)
print(f"Original Model - Top-1 Accuracy: {top1_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%")

# Model Preparation & Validation
print("\nPreparing and validating the model for quantization...")
prepared_model = prepare_model(model).to(device)
validation_result = ModelValidator.validate(prepared_model, input_shapes=(1, 3, 224, 224))
if validation_result.has_errors():
    raise RuntimeError("Model validation failed! Please fix the issues before proceeding.")

# Apply AIMET Quantization
def apply_aimet_quantization(model, device, dataloader):
    dummy_input = torch.rand(1, 3, 224, 224).to(device)

    # Fold batch normalization layers
    fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224), dummy_input=dummy_input)

    quant_sim = qsim.QuantizationSimModel(
        model,
        dummy_input=dummy_input,
        quant_scheme='tf_enhanced',
        rounding_mode='nearest',
        default_param_bw=8,
        default_output_bw=8
    )

    # Calibration using a small dataset
    def forward_pass(model, dataloader):
        model.eval()
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= 10:  # Use only first 10 batches for calibration
                    break
                model(images.to(device))

    quant_sim.compute_encodings(forward_pass, dataloader)

    return quant_sim.model

# Perform quantization
quantized_model = apply_aimet_quantization(prepared_model, device, dataloader)

# Validate quantized model
print("\nValidating quantized model...")
top1_acc, top5_acc = validate_pytorch(quantized_model, dataloader, device)
print(f"Quantized Model - Top-1 Accuracy: {top1_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%")
