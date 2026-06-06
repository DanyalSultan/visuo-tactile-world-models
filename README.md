# Visuo-Tactile World Models

A research prototype for multimodal predictive world modelling in robotic manipulation.

This project fuses visual, tactile, proprioceptive and action-command data into a unified Transformer-based representation for predicting future robot kinematic states. The main experiment uses the RH20T robotic manipulation dataset and combines NVIDIA Cosmos visual tokens with a custom tactile encoder, proprioception and action commands to form a 1,027-token multimodal sequence.

## Project Purpose

Modern robotic manipulation systems often rely on reactive feedback, adjusting only after contact changes or slip have already occurred. This project explores a predictive alternative: a visuo-tactile world model that learns to anticipate the robot's next kinematic state from synchronised multimodal sensor streams.

The model combines:

* RGB visual observations
* Geometric tactile readings
* Robot proprioceptive state
* Action-command data

These inputs are temporally aligned, embedded into a shared latent space, fused into a multimodal token sequence, and passed through a Transformer encoder to predict the next 7-DoF robot TCP pose.

## Core Pipeline

The main pipeline is:

```text
RH20T dataset
    ↓
RH20main.py
    ↓
vision frames + tactile data + proprioception + action commands
    ↓
Cosmos image tokenizer + custom tactile encoder
    ↓
fusion_layer.py
    ↓
[Batch, 1027, 256] multimodal sequence
    ↓
world_transformer.py
    ↓
predicted next 7-DoF robot pose
```

The final fused sequence contains:

```text
1024 vision tokens
+ 1 tactile token
+ 1 proprioception token
+ 1 action-command token
= 1027 multimodal tokens
```

## Main Project Files

### `train.py`

Main training entry point for the visuo-tactile world model.

It:

* Loads RH20T task folders from `data/RH20T_cfg7/`
* Creates a combined dataset across available task folders
* Loads the NVIDIA Cosmos image tokenizer
* Initialises the custom tactile encoder
* Initialises the multimodal fusion layer
* Initialises the Transformer-based predictive world model
* Trains the model using MSE loss against the next robot TCP pose
* Saves checkpoints after each epoch

Run with:

```bash
python train.py
```

### `RH20main.py`

Dataset-loading and temporal synchronisation logic for RH20T.

It loads:

* RGB video frames from `color.mp4`
* Video timestamps from `timestamps.npy`
* Tactile data from `transformed/tactile.npy`
* High-frequency robot state data from `transformed/high_freq_data.npy`
* Action-command data from `robot_command/tcpcommand_timestamp.npy`

It performs nearest-neighbour timestamp alignment between the different sensor streams and returns PyTorch tensors for:

```text
vision
tactile
proprio
action
target
```

The target is the next timestep robot TCP pose.

### `tactile_encoder.py`

Custom tactile encoder.

This file defines `SparshTactileEncoder`, a 3-layer MLP that converts raw geometric tactile data into the shared 256-dimensional embedding space.

Expected input shape:

```text
[Batch, 2, 16, 3]
```

Output shape:

```text
[Batch, 256]
```

### `fusion_layer.py`

Multimodal fusion layer.

This file defines `MultimodalFusion`, which projects and combines:

* Cosmos visual tokens
* Tactile embeddings
* Proprioceptive state
* Action-command vectors

The output is a fused sequence:

```text
[Batch, 1027, 256]
```

### `world_transformer.py`

Transformer-based predictive world model.

This file defines `PredictiveWorldModel`, a Transformer encoder with:

* 256-dimensional embeddings
* 8 attention heads
* 4 Transformer encoder layers
* GELU activations
* A final linear prediction head

The model predicts the next 7-DoF robot TCP pose.

### `download_cosmos.py`

Downloads the NVIDIA Cosmos image tokenizer weights:

```text
Cosmos-0.1-Tokenizer-CI8x8
```

The weights are downloaded into:

```text
pretrained_ckpts/Cosmos-0.1-Tokenizer-CI8x8/
```

Run with:

```bash
python download_cosmos.py
```

### `download_cosmos_video.py`

Downloads the NVIDIA Cosmos video tokenizer weights:

```text
Cosmos-0.1-Tokenizer-CV8x8x8
```

This was used for video-tokenisation experiments and ALOHA-related testing.

Run with:

```bash
python download_cosmos_video.py
```

## Experimental / Secondary Files

### `aloha_main.py`

Experimental ALOHA perception script for combining Cosmos video tokens and Sparsh-X tactile tokens from ALOHA-style HDF5 recordings.

This is not the main RH20T training entry point, but it documents an extension path for ALOHA-based multimodal perception.

### `aloha_test.py`

Hardware-aware ALOHA test script using dummy video and tactile tensors to validate the Cosmos video tokenizer and Sparsh-X encoder pipeline.

### `train_task.py`

Inherited/support training script from the tactile representation learning codebase. This is mainly associated with the `tactile_ssl/` framework and is not the main visuo-tactile world model training entry point.

### `inference.py`

Example Sparsh-X inference script for testing tactile encoders on dummy inputs. This is useful for checking the inherited tactile encoder pipeline, but it is not the main evaluation script for the RH20T world model.

## Sanity Test Scripts

The repository includes several small test scripts used during development.

```text
test_vision.py
test_video_sync.py
test_tactile.py
test_sync.py
test_action.py
test_joints.py
test_cosmos.py
test_cosmos_video.py
test_sparsh.py
```

Recommended sanity-test order:

```bash
python test_vision.py
python test_tactile.py
python test_sync.py
python test_video_sync.py
python test_action.py
python test_cosmos.py
```

Some tests require local RH20T data or pretrained checkpoints.

## Support Folders

### `tactile_ssl/`

Inherited/support code for tactile representation learning and Sparsh/Sparsh-X style encoders.

This folder includes:

* encoder-building utilities
* tactile datasets
* Transformer backbones
* training utilities
* downstream task components
* loss functions
* plotting and logging utilities

This folder supports the tactile foundation model side of the project, but the core world-model contribution is mainly in:

```text
RH20main.py
tactile_encoder.py
fusion_layer.py
world_transformer.py
train.py
```

### `config/`

Configuration files for tactile SSL models, encoders, datasets, tasks, W&B settings and experiment definitions.

Important subfolders include:

```text
config/encoder/
config/data/
config/task/
config/experiment/
config/paths/
```

### `scripts/`

Helper scripts inherited from the tactile representation learning codebase, including conversion, dataset and ROS-related utilities.

### `assets/`

Robot/tactile sensor assets, meshes, URDF files and README figures inherited from the tactile representation learning codebase.

## Environment Setup

The repository uses a Conda environment file:

```text
environment.yaml
```

Create the environment with:

```bash
conda env create -f environment.yaml
```

Activate the environment:

```bash
conda activate worldmodel
```

If the environment file is edited or renamed, check the `name:` field inside `environment.yaml` and activate that environment name.

## Important Encoding Note

If `environment.yaml` was edited on Windows, ensure it is saved as UTF-8.

In VS Code:

1. Open `environment.yaml`
2. Click the encoding label in the bottom-right
3. Select `Save with Encoding`
4. Choose `UTF-8`

This prevents Conda and GitHub from misreading the file.

## Required External Files

Large datasets and model checkpoints are intentionally not included in this repository.

You must manually provide:

### RH20T dataset

Expected local structure:

```text
data/
└── RH20T_cfg7/
    ├── task_0001_user_0014_scene_0001_cfg_0007/
    │   ├── cam_037522061512/
    │   │   ├── color.mp4
    │   │   └── timestamps.npy
    │   ├── transformed/
    │   │   ├── tactile.npy
    │   │   └── high_freq_data.npy
    │   └── robot_command/
    │       └── tcpcommand_timestamp.npy
    └── task_...
```

The training script scans:

```text
data/RH20T_cfg7/
```

for folders beginning with:

```text
task_
```

### Cosmos image tokenizer

Download with:

```bash
python download_cosmos.py
```

Expected location:

```text
pretrained_ckpts/
└── Cosmos-0.1-Tokenizer-CI8x8/
    └── encoder.jit
```

### Cosmos video tokenizer

Only needed for video-tokeniser or ALOHA experiments.

Download with:

```bash
python download_cosmos_video.py
```

Expected location:

```text
pretrained_ckpts/
└── Cosmos-0.1-Tokenizer-CV8x8x8/
    └── encoder.jit
```

### Sparsh-X / tactile SSL checkpoints

If using the inherited tactile SSL pipeline, place checkpoints in:

```text
checkpoints/
```

or:

```text
pretrained_ckpts/
```

depending on the script being used.

Example paths referenced by the scripts include:

```text
checkpoints/d360_sparshx_img_mic_imu_pressure_tiny.pth
checkpoints/xela_sparshskin_tiny.pth
pretrained_ckpts/sparsh_base.pth
```

These files are not uploaded to GitHub because they are large model weights.

## Files Excluded from GitHub

The following are excluded through `.gitignore`:

```text
data/
datasets/
RH20T/
pretrained_ckpts/
checkpoints/
models/
wandb/
runs/
outputs/
logs/
*.pth
*.pt
*.ckpt
*.safetensors
*.onnx
*.npy
*.npz
*.mp4
*.avi
*.mov
```

This prevents large datasets, checkpoints, videos and experiment logs from being accidentally committed.

## Reproducing the Main RH20T Experiment

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/visuo-tactile-world-models.git
cd visuo-tactile-world-models
```

Replace `YOUR-USERNAME` with the correct GitHub username after publishing the repository.

### 2. Create the environment

```bash
conda env create -f environment.yaml
conda activate worldmodel
```

### 3. Download the Cosmos image tokenizer

```bash
python download_cosmos.py
```

Confirm that this file exists:

```text
pretrained_ckpts/Cosmos-0.1-Tokenizer-CI8x8/encoder.jit
```

### 4. Add the RH20T dataset

Place the dataset under:

```text
data/RH20T_cfg7/
```

The folder should contain one or more task folders beginning with `task_`.

### 5. Run sanity checks

Check the RH20T video stream:

```bash
python test_vision.py
```

Check tactile formatting:

```bash
python test_tactile.py
```

Check timestamp synchronisation:

```bash
python test_sync.py
python test_video_sync.py
```

Check action-command loading:

```bash
python test_action.py
```

Check Cosmos image tokenisation:

```bash
python test_cosmos.py
```

### 6. Train the world model

```bash
python train.py
```

On a cluster or SLURM-based system, use:

```bash
bash run_training.sh
```

`run_training.sh` is configured for a GPU job and activates the `worldmodel` environment before running:

```bash
python train.py
```

## Expected Training Outputs

Training saves model checkpoints after each epoch as:

```text
world_model_epoch_1.pth
world_model_epoch_2.pth
...
```

These checkpoint files are ignored by Git because they can become large.

Each checkpoint contains:

```text
epoch
tactile_state
fusion_state
world_state
optimizer
loss
```

## Hardware Notes

The full experiment is computationally heavy.

Recommended hardware:

* CUDA-compatible NVIDIA GPU
* Large VRAM GPU preferred
* Sufficient CPU RAM for video frame loading
* Large local storage for RH20T data and checkpoints

The original training pipeline was designed for workstation/cluster-style hardware rather than standard laptop execution.

## Precision and Stability Notes

The main training script uses Cosmos image tokens, then converts the token output back to `float32` before fusion and prediction.

The fused sequence is long:

```text
[Batch, 1027, 256]
```

Long sequence lengths can make Transformer training memory-intensive. Avoid changing precision or attention settings unless you have tested stability carefully.

## Known Limitations

This repository is a research prototype rather than a polished software package.

Current limitations:

* Dataset files are not included.
* Pretrained model weights are not included.
* Some scripts contain local assumptions and may need path edits.
* `inference.py` is currently a tactile encoder demonstration rather than a full RH20T evaluation pipeline.
* `aloha_main.py` and `aloha_test.py` are experimental ALOHA extensions, not the main RH20T training route.
* Some inherited folders come from the tactile SSL/Sparsh support codebase.
* Full training is GPU-heavy.
* More systematic downstream evaluation is future work.

## Recommended Final Repository Structure

```text
visuo-tactile-world-models/
│
├── README.md
├── environment.yaml
├── .gitignore
├── run_training.sh
│
├── train.py
├── RH20main.py
├── tactile_encoder.py
├── fusion_layer.py
├── world_transformer.py
│
├── inference.py
├── aloha_main.py
├── aloha_test.py
│
├── download_cosmos.py
├── download_cosmos_video.py
│
├── test_vision.py
├── test_tactile.py
├── test_sync.py
├── test_video_sync.py
├── test_action.py
├── test_joints.py
├── test_cosmos.py
├── test_cosmos_video.py
├── test_sparsh.py
│
├── config/
├── scripts/
├── tactile_ssl/
├── assets/
│
├── data/                  # ignored, user provides locally
├── pretrained_ckpts/      # ignored, user provides locally
├── checkpoints/           # ignored, user provides locally
└── outputs/               # ignored, generated locally
```

## Project Background

This repository accompanies my BEng General Engineering (Software) dissertation at the University of Sheffield:

**Visuo-Tactile World Models: Multimodal Tokenisation and Predictive Representation for Robotic Manipulation**

The project was supervised by **Dr Amir Ghalamzan** and investigates multimodal predictive representation learning for robotic manipulation using visual, tactile, proprioceptive and action-command data.

## Acknowledgements

I would like to thank **Dr Amir Ghalamzan** for his guidance and supervision throughout this project. I also acknowledge the support of the **Intelligent Manipulation Laboratory (IML)** and thank **Emma Harrison** for her insights into world models, which helped shape the direction of this research.

This project uses and adapts support code from the **Sparsh/Sparsh-X tactile representation learning codebase** for tactile encoder experimentation. The main project contribution is the RH20T-based visuo-tactile world model pipeline implemented through:

```text
RH20main.py
tactile_encoder.py
fusion_layer.py
world_transformer.py
train.py

## Author

Muhammad Danyal Sultan
University of Sheffield
General Engineering
Final Year Project: Multimodal Predictive World Models for Robotic Manipulation
