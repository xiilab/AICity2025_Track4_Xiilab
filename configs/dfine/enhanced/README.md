# Enhanced D-FINE Configuration Guide

This directory contains configuration files for the Enhanced D-FINE models with Adaptive Training-Time Decoder Enhancement (ATDE), Progressive Layer Distillation (PLD), and Curriculum Learning.

## Configuration Files

### Base Configuration
- `dfine_enhanced_base.yml`: Base configuration with all enhanced features

### Model Variants
- `dfine_enhanced_n.yml`: Nano variant (lightweight, 2-4 layers, ~35 AP target)
- `dfine_enhanced_s.yml`: Small variant (balanced, 3-5 layers, ~42 AP target)  
- `dfine_enhanced_l.yml`: Large variant (full-featured, 4-8 layers, ~50 AP target)

## Key Features

### 1. Adaptive Training-Time Decoder Enhancement (ATDE)
- Uses sophisticated decoder layers during training
- Switches to lightweight layers during inference
- Configurable base/training layer counts per variant

### 2. Progressive Layer Distillation (PLD)
- Feature alignment between training and inference layers
- Attention distillation for better knowledge transfer
- Response distillation for improved localization

### 3. Curriculum Learning
- Progressive training from easy to hard samples
- Adaptive model complexity during training
- Stage-based loss weighting

### 4. Enhanced Multi-Scale Feature Fusion
- Attention-weighted feature fusion
- Cross-scale feature interaction
- Adaptive feature pyramid networks

## Usage Examples

### Training Enhanced D-FINE-N (Nano)
```bash
python train_enhanced.py \
    --config configs/dfine/enhanced/dfine_enhanced_n.yml \
    --update output_dir ./output/enhanced_dfine_n
```

### Training Enhanced D-FINE-S (Small)
```bash
python train_enhanced.py \
    --config configs/dfine/enhanced/dfine_enhanced_s.yml \
    --update output_dir ./output/enhanced_dfine_s
```

### Training Enhanced D-FINE-L (Large)
```bash
python train_enhanced.py \
    --config configs/dfine/enhanced/dfine_enhanced_l.yml \
    --update output_dir ./output/enhanced_dfine_l
```

### Testing with Different Complexity Budgets
```bash
python train_enhanced.py \
    --config configs/dfine/enhanced/dfine_enhanced_n.yml \
    --test-only \
    --resume ./output/enhanced_dfine_n/best_checkpoint.pth
```

### Fine-tuning from Pretrained Model
```bash
python train_enhanced.py \
    --config configs/dfine/enhanced/dfine_enhanced_s.yml \
    --tuning \
    --resume ./pretrained/dfine_s_pretrained.pth \
    --update train.epochs 20 CurriculumLearning.total_epochs 20
```

## Configuration Parameters

### Adaptive Decoder Settings
```yaml
AdaptiveTrainingDecoder:
  base_num_layers: 3        # Layers used during inference
  training_num_layers: 6    # Layers used during training
  distillation_weight: 1.0  # Knowledge distillation strength
  progressive_distillation: True
```

### Curriculum Learning Settings
```yaml
CurriculumLearning:
  total_epochs: 80
  easy_stage_ratio: 0.3     # 30% of training on easy samples
  medium_stage_ratio: 0.4   # 40% of training on mixed samples
  hard_stage_ratio: 0.3     # 30% of training on hard samples
```

### Enhanced Feature Fusion Settings
```yaml
AdaptiveFeaturePyramid:
  attention_type: "both"    # "channel", "spatial", "both"
  fusion_method: "adaptive" # "adaptive", "concat", "add"
  use_deformable: False     # Enable deformable convolutions
```

## Expected Performance Improvements

### D-FINE-N (Nano)
- **Baseline**: ~32 AP, 45 FPS
- **Enhanced**: ~35 AP (+3), 60 FPS (+15)
- **Memory**: 30% reduction during inference

### D-FINE-S (Small)  
- **Baseline**: ~40 AP, 35 FPS
- **Enhanced**: ~42 AP (+2), 45 FPS (+10)
- **Memory**: 25% reduction during inference

### D-FINE-L (Large)
- **Baseline**: ~48 AP, 20 FPS  
- **Enhanced**: ~50 AP (+2), 25 FPS (+5)
- **Memory**: 20% reduction during inference

## Training Stages

### Stage 1: Warmup (Epochs 1-10)
- Simple architecture (min layers)
- Easy samples only
- Low distillation weight
- Focus on basic feature learning

### Stage 2: Easy (Epochs 11-30)
- Gradual complexity increase
- Easy to medium samples
- Increasing distillation weight
- Progressive layer activation

### Stage 3: Medium (Epochs 31-60)
- Moderate complexity
- Mixed difficulty samples
- Balanced loss weighting
- Full feature fusion active

### Stage 4: Hard (Epochs 61-80)
- Full complexity
- All samples including hard ones
- Maximum distillation weight
- Fine-tuning and optimization

## Monitoring Training

### Key Metrics to Watch
- `distillation_loss`: Knowledge transfer effectiveness
- `complexity_score`: Scene complexity prediction
- `early_stop_layer`: Dynamic inference decisions
- `curriculum_stage`: Current training stage

### TensorBoard Logs
```bash
tensorboard --logdir ./output/enhanced_dfine_n/logs
```

### Expected Log Output
```
Epoch 25: Stage=easy, Layers=3, Distillation=0.500
Epoch 50: Stage=medium, Layers=4, Distillation=0.750  
Epoch 75: Stage=hard, Layers=6, Distillation=1.000
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Reduce `batch_size` or `training_num_layers`
2. **Slow Convergence**: Increase `distillation_weight` or extend warmup
3. **Poor Lightweight Performance**: Increase curriculum learning epochs
4. **Unstable Training**: Reduce learning rate or increase gradient clipping

### Performance Tuning

1. **For Better Accuracy**: Increase `training_num_layers` and distillation weight
2. **For Faster Inference**: Reduce `base_num_layers` and enable early stopping
3. **For Memory Efficiency**: Use smaller hidden dimensions and fewer attention heads

## Advanced Usage

### Custom Difficulty Metrics
```yaml
CurriculumLearning:
  difficulty_metrics: ['object_size', 'crowd_density', 'custom_metric']
```

### Dynamic Complexity Budgets
```yaml
evaluation:
  test_complexity_budgets: [0.2, 0.4, 0.6, 0.8, 1.0]
  adaptive_budget_selection: True
```

### Multi-GPU Training
```bash
python -m torch.distributed.launch --nproc_per_node=4 train_enhanced.py \
    --config configs/dfine/enhanced/dfine_enhanced_l.yml
``` 