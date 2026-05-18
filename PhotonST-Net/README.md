# PhotonST-Net

Spatiotemporal fusion network for photon-counted imaging under ultra-low light conditions.

## Network Architecture

PhotonST-Net adopts a hierarchical encoder-decoder structure with lateral skip connections. Each encoder stage consists of two 3×3 convolutions with Batch Normalization and ReLU activation, with 2×2 MaxPool downsampling. The decoder uses ConvTranspose2d upsampling followed by feature concatenation and refinement convolutions. A ConvLSTM unit is inserted at the bottleneck to capture temporal dependencies across frames.

## Requirements

```bash
pip install -r requirements.txt
```

Tested with PyTorch 2.0.0, Python 3.8+.

## Training

Configure dataset paths and hyperparameters in `train.py`:

```python
input_root_directory = './data/input'  # Input images
target_directory = './data/target'      # Ground-truth images
model_directory = './checkpoints'       # Model output path
```

Then run:

```bash
python train.py
```

After training, `checkpoints/` will contain:
- `best_model.pth` — best model (overwritten each time validation loss improves)
- `epoch_X.pth` — snapshot of every epoch

Key hyperparameters (matching the paper):

| Parameter | Value |
|-----------|-------|
| Batch size | 16 |
| Epochs | 100 |
| Learning rate | 0.001 |
| Loss weights (α, β) | 0.7, 0.3 |
| Frame window | 3 frames |
| LR patience | 3 epochs |
| LR factor | 0.5 |

### Dataset Structure

Place your data under `./data/` as follows:

Input images:
```
data/
└── input/
    ├── scene_001/
    │   ├── frame_000.png
    │   ├── frame_001.png
    │   └── ...
    └── scene_002/
        └── ...
```

Ground-truth images for single-target reconstruction:
```
data/
└── target/
    ├── scene_001/
    │   ├── frame_000.png
    │   └── ...
    └── scene_002/
        └── ...
```

Ground-truth images for double-target separation:
```
data/
└── target/
    ├── scene_001/
    │   ├── 1/
    │   │   └── frame_000.png  # target 1
    │   └── 2/
    │       └── frame_000.png  # target 2
    └── scene_002/
        └── ...
```

### Loss Function

The training objective follows:

```
Loss_recon = α · MSE + β · (1 - SSIM)
Loss_total = Loss_recon + λ · Loss_class,  λ = 0.15
```

where `Loss_class` is cross-entropy loss for target count classification.

## Inference

Configure paths in `inference.py`:

```python
input_dir = './data/test'                    # Test images
output_base_dir = './output'                 # Output directory
model_path = './checkpoints/best_model.pth' # Model weights
```

Then run:

```bash
python inference.py
```

Outputs are saved to `output_base_dir/1/` (target 1) and `output_base_dir/2/` (target 2, if dual-target).

## Model

`model.py` defines `UNetWithDynamicOutputs`, the core PhotonST-Net architecture. Input channels are `1 + 2 * frame_range`. For the default 3-frame window, set `n_channels=3`.

## Citation

If this code is helpful for your research, please cite our paper.
