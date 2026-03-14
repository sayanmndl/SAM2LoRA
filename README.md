# SAM2LoRA for Retinal Vessel and Optic Disc Segmentation

This project is a LoRA implementation of SAM2 by Meta for segmenting fundus images to identify optic discs and blood vessels.

## Datasets

### Blood Vessel:
- CHASEDB1
- DRIVE
- FIVES
- HRF

### Optic Disc:
- DRISHTIGS
- G1020
- GRAPE
- IDRID
- ORIGA
- PAPILADB
- REFUGE2

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/sayanmandal/SAM2LoRA.git
cd SAM2LoRA
```

### 2. Install SAM2

SAM2LoRA depends on Meta's Segment Anything Model 2. Clone it into the project root and install it:

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..
```

### 3. Download SAM2 checkpoints

Download the SAM2.1 model checkpoints into `segment-anything-2/checkpoints/`:

```bash
cd segment-anything-2/checkpoints
# Large (recommended)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
# Small
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
# Base+
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
cd ../..
```

### 4. Install Python dependencies

```bash
pip install torch torchvision tensorboard tqdm numpy
```

### 5. Prepare datasets

Place each dataset under a directory accessible by the dataloaders. Each dataset class in `dataloader/` expects a specific folder structure — refer to the individual loader files for path conventions.

---

## How to Use

### Training

**Optic disc / cup segmentation (recommended config):**

```bash
python3 train.py \
  --dataset_name optic_disc \
  --seg_type cup_only \
  --model_size l \
  --lora_rank 256 \
  --lora_alpha 512 \
  --learning_rate 1e-4 \
  --number_steps 120000 \
  --batch_size 2 \
  --accumulation_steps 4 \
  --num_pos_points 5 \
  --num_neg_points 0 \
  --save_frequency 10000 \
  --output_path ./checkpoints \
  --output_name cup_only
```

**Blood vessel segmentation:**

```bash
python3 train.py \
  --dataset_name vessel \
  --seg_type od \
  --model_size l \
  --lora_rank 256 \
  --lora_alpha 512 \
  --learning_rate 1e-4 \
  --number_steps 120000 \
  --batch_size 2 \
  --accumulation_steps 4 \
  --num_pos_points 5 \
  --output_path ./checkpoints \
  --output_name vessel
```

**Key training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | `vessel` | `vessel`, `optic_disc`, or a single dataset (e.g. `drive`, `refuge2`) |
| `--seg_type` | `od` | Segmentation target: `od` (optic disc), `cup_only`, `cup`, `rim` |
| `--model_size` | `l` | SAM2 model size: `l` (large), `s` (small), `b` (base) |
| `--lora_rank` | `64` | LoRA rank (paper uses 256) |
| `--lora_alpha` | `128` | LoRA alpha — typically `2 × lora_rank` |
| `--learning_rate` | `1e-4` | Learning rate |
| `--number_steps` | `1000` | Total training steps (paper uses 120 000) |
| `--batch_size` | `8` | Batch size per step |
| `--accumulation_steps` | `2` | Gradient accumulation steps |
| `--num_pos_points` | `200` | Positive prompt points per sample |
| `--num_neg_points` | `0` | Negative prompt points per sample |
| `--num_boxes` | `0` | Box prompts per sample |
| `--save_frequency` | `1000` | Save a checkpoint every N steps |
| `--optimizer_type` | `adamw` | `adamw` or `sgd` |
| `--scheduler_type` | `cosinewarm` | `cosinewarm` or `steplr` |
| `--checkpoint_path` | `./checkpoints` | Pass a `.ckpt` file to resume training |
| `--output_path` | `./checkpoints` | Directory for saved checkpoints |
| `--tensorboard_path` | `./runs` | Directory for TensorBoard logs |

Saved checkpoints follow the naming convention:
`fundus_{dataset}_{output_name}_sam2_{size}_r{rank}_a{alpha}.ckpt`

Resume training by passing the `.ckpt` path to `--checkpoint_path`.

### Evaluation

**Single config, all eval modes:**

```bash
for eval_mode in 0 1 2 3 4 5 6; do
  python3 test.py \
    --dataset_name optic_disc \
    --seg_type cup_only \
    --model_size l \
    --lora_rank 256 \
    --lora_alpha 512 \
    --checkpoint_path ./checkpoints/fundus_optic_disc_cup_only_sam2_l_r256_a512_best.ckpt \
    --output_path ./results \
    --output_name cup_only \
    --eval_mode $eval_mode
done
```

**Eval modes:**

| Mode | Prompt type | Points |
|---|---|---|
| `0` | None | — |
| `1` | Point | 1 positive |
| `2` | Point | 2 positive |
| `3` | Point | 5 positive |
| `4` | Point | 5 pos + 1 neg |
| `5` | Box | 1 bounding box |
| `6` | All | 5 pos + 1 neg + box |

Results are saved as a JSON file under `--output_path`.

### PowerShell Scripts (Windows)

Three helper scripts are included for convenience:

| Script | Purpose |
|---|---|
| `train_script.ps1` | Single training run (optic disc, cup segmentation) |
| `eval_script_2.ps1` | Evaluate one checkpoint across all 7 eval modes |
| `eval_script_1.ps1` | Sweep over multiple dataset/seg-type pairs, ranks, and eval modes |

Run them from PowerShell:

```powershell
.\train_script.ps1
.\eval_script_2.ps1
.\eval_script_1.ps1
```

Edit the variable block at the top of each script to adjust paths, ranks, and hyperparameters before running.

### TensorBoard

Monitor training loss and metrics:

```bash
tensorboard --logdir ./runs
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{mandal2025sam2lora,
  title={SAM2LoRA: Composite Loss-Guided, Parameter-Efficient Finetuning of SAM2 for Retinal Fundus Segmentation},
  author={Mandal, Sayan and Karthikeyan, Divyadarshini and Paldhe, Manas},
  journal={arXiv preprint arXiv:2510.10288},
  year={2025}
}
```
