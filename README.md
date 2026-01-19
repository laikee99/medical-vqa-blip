# Medical VQA: Enhanced BLIP + CNN Baseline

A medical Visual Question Answering (VQA) system using fine-tuned BLIP model with comprehensive evaluation and interactive web interface.

## Overview

This project implements a medical image question answering system that can:
- Answer questions about medical images (X-ray, CT, MRI)
- Compare BLIP transformer model with CNN baseline
- Provide evaluation breakdown by question types
- Generate medical advice recommendations
- Offer an interactive Gradio web interface

## Features

- **BLIP Model**: Fine-tuned Salesforce/blip-vqa-base on VQA-RAD dataset
- **CNN Baseline**: Question-aware ResNet50 model for fair comparison
- **Evaluation Metrics**: Exact Accuracy, Contains Accuracy, BLEU, ROUGE-L
- **Question Type Analysis**: Separate metrics for CLOSED_YESNO, CLOSED_SHORT, and OPEN questions
- **Visualizations**: Training curves, Grad-CAM heatmaps, attention maps
- **Web Interface**: Gradio-based UI with Cloudflare tunnel support

## Dataset

**VQA-RAD** (Visual Question Answering in Radiology)
- Total samples: 2,248
- Question types:
  - CLOSED_YESNO: 53.1%
  - CLOSED_SHORT: 39.1%
  - OPEN: 7.8%
- Train/Test split: 80%/20%

## Model Architecture

### BLIP (Fine-tuned)
- Base model: `Salesforce/blip-vqa-base`
- Parameters: 361M
- Training: 50 epochs, LR=1e-5, cosine annealing
- Mixed precision (FP16) training

### CNN Baseline
- Image encoder: ResNet50 (ImageNet pretrained)
- Question encoder: Word embedding + projection
- Fusion: Concatenation + MLP classifier
- Training: 50 epochs, LR=1e-4

## Results

### BLIP Performance (Before vs After Fine-tuning)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Exact Accuracy | 27.00% | 64.00% | +37.00% |
| Contains Accuracy | 28.00% | 66.50% | +38.50% |
| BLEU | 4.99 | 16.19 | +11.20 |
| ROUGE-L | 28.31% | 68.92% | +40.61% |

### Performance by Question Type (Fine-tuned BLIP)

| Type | N | Exact% | Contains% | BLEU | ROUGE-L |
|------|---|--------|-----------|------|---------|
| CLOSED_YESNO | 112 | 75.89 | 75.89 | 13.50 | 75.89 |
| CLOSED_SHORT | 71 | 53.52 | 59.15 | 14.54 | 61.14 |
| OPEN | 17 | 29.41 | 35.29 | 40.89 | 55.48 |

## Requirements

```
torch>=2.0
transformers
torchvision
pillow
nltk
rouge-score
gradio
kagglehub
matplotlib
numpy
tqdm
```

## Usage

### Run in Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Select GPU runtime (A100 recommended)
3. Run all cells sequentially
4. Access the web interface via the generated Cloudflare URL

### Quick Start (Load Saved Model)

If you have a previously trained model:
```python
# Run Cell 20 to load saved model and launch WebUI directly
```

### Training from Scratch

```python
# Run Cells 1-11 for full training pipeline
# Cell 1: Install dependencies
# Cell 2: Configuration
# Cell 3: Load dataset
# Cell 4-5: Setup BLIP
# Cell 6-8: Train and evaluate BLIP
# Cell 9-11: Train and evaluate CNN baseline
```

## Project Structure

```
.
├── course_blip_enhanced.ipynb    # Main notebook
├── README.md                      # This file
└── blip_enhanced_output/          # Output directory (generated)
    ├── blip_finetuned/           # Saved model
    ├── visualizations/            # Sample predictions
    ├── report_figures/            # Training & comparison plots
    └── all_results.json          # Evaluation results
```

## Output Visualizations

The notebook generates:
1. **Training curves**: Loss and learning rate over epochs
2. **Fine-tuning comparison**: Before/after performance
3. **Model comparison**: BLIP vs CNN across question types
4. **Grad-CAM visualizations**: Model attention on medical images
5. **Summary tables**: Comprehensive metrics comparison

## Web Interface

The Gradio interface allows:
- Upload medical images (X-ray, CT, MRI)
- Ask natural language questions
- Get VQA predictions from fine-tuned BLIP
- Receive medical recommendations (educational only)

## Limitations

- CNN baseline can only predict answers in the predefined vocabulary (top 100)
- OPEN questions have lower accuracy due to limited training samples
- Medical advice is for educational purposes only, not clinical use

## Citation

If you use this code, please cite:

```bibtex
@misc{medical-vqa-blip,
  title={Medical VQA: Enhanced BLIP + CNN Baseline},
  year={2024},
  note={Fine-tuned on VQA-RAD dataset}
}
```

## License

This project is for educational and research purposes only.

## Disclaimer

This system is a research demonstration and should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice.
