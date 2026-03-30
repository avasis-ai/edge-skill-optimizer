# AGENTS.md - Edge Skill Optimizer Project Context

This folder is home. Treat it that way.

## Project: Edge-Skill-Optimizer (#55)

### Identity
- **Name**: Edge-Skill-Optimizer
- **License**: Apache-2.0
- **Org**: avasis-ai
- **PyPI**: edge-skill-optimizer
- **Version**: 0.1.0
- **Tagline**: Shrink complex agent skills to run natively on mobile devices

### What It Does
This pipeline takes a complex SKILL.md workflow and distills it into a highly quantized, task-specific micro-model (e.g., 1B parameters). This specialized model runs flawlessly on iOS or Android NPUs without requiring cloud connectivity.

### Inspired By
- TensorRT
- ONNX
- LocalLLaMA + Hardware acceleration
- Mobile AI optimization

### Core Components

#### `/distiller/`
- Knowledge distillation
- Model compression
- Quantization
- Optimization tracking

#### `/mobile-bindings/`
- iOS CoreML integration
- Android TensorFlow Lite
- ONNX export
- NPU optimization

### Technical Architecture

**Key Dependencies:**
- `torch>=2.0` - PyTorch framework (Trust score: verified)
- `onnx>=1.14` - ONNX model format (Trust score: verified)
- `click>=8.0` - CLI framework (Trust score: 8.8)

**Core Modules:**
1. `distiller.py` - Knowledge distillation and quantization
2. `cli.py` - Command-line interface

### AI Coding Agent Guidelines

#### When Contributing:

1. **Understand mobile AI**: Edge optimization requires awareness of device constraints
2. **Use Context7**: Check trust scores for new libraries before adding dependencies
3. **Accuracy preservation**: Never sacrifice too much accuracy for compression
4. **Device targeting**: Optimize for specific hardware (iOS, Android, Edge)
5. **Quantization awareness**: Choose appropriate quantization levels
6. **Export formats**: Support multiple export formats for flexibility

#### What to Remember:

- **Compression Tradeoffs**: Higher compression = more accuracy loss
- **Device-Specific**: Different optimizations for different platforms
- **Privacy**: Keep data on-device whenever possible
- **Performance**: Balance size with inference speed
- **Backwards Compatibility**: Support older device versions
- **Testing**: Validate on actual devices when possible

#### Common Patterns:

**Optimize a skill:**
```python
from edge_skill_optimizer.distiller import MobileOptimizer, QuantizationType

optimizer = MobileOptimizer()
distillation, metadata = optimizer.optimize_skill(
    skill_name="chat_skill",
    model_size_mb=500.0,
    target_device="iOS",
    quantization_type=QuantizationType.INT8,
    target_accuracy_loss=0.05
)

print(f"Size: {metadata.original_size_mb:.1f} MB → {metadata.optimized_size_mb:.1f} MB")
```

**Distill a model:**
```python
from edge_skill_optimizer.distiller import KnowledgeDistiller

distiller = KnowledgeDistiller()
result = distiller.distill_model(
    teacher_model="skill_large",
    student_model="skill_mobile",
    target_device="iOS",
    epochs=10,
    target_size_mb=100.0
)
```

**Quantize a model:**
```python
from edge_skill_optimizer.distiller import ModelQuantizer, QuantizationType

quantizer = ModelQuantizer()
metadata = quantizer.quantize_model(
    model_id="vision_model",
    original_size_mb=200.0,
    quantization_type=QuantizationType.INT4,
    target_device="Android",
    accuracy_loss=0.1
)
```

### Project Status

- ✅ Initial implementation complete
- ✅ Knowledge distillation
- ✅ Model quantization
- ✅ Mobile optimization tracking
- ✅ ONNX export capability
- ✅ CLI interface
- ✅ Comprehensive test suite
- ⚠️ Real model training pending
- ⚠️ Device testing pending

### How to Work with This Project

1. **Read `SOUL.md`** - Understand who you are
2. **Read `USER.md`** - Know who you're helping
3. **Check `memory/YYYY-MM-DD.md`** - Recent context
4. **Read `MEMORY.md`** - Long-term decisions (main session only)
5. **Execute**: Code → Test → Commit

### Red Lines

- **No stubs or TODOs**: Every function must have real implementation
- **Type hints required**: All function signatures must include types
- **Docstrings mandatory**: Explain what, why, and how
- **Test coverage**: New features need tests
- **Accuracy awareness**: Document accuracy impact of optimizations

### Development Workflow

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Format code
black src/ tests/
isort src/ tests/

# Check syntax
python -m py_compile src/edge_skill_optimizer/*.py

# Run CLI
edge-optimizer --help

# Commit
git add -A && git commit -m "feat: add INT4 quantization"
```

### Key Files to Understand

- `src/edge_skill_optimizer/distiller.py` - Distillation and quantization
- `src/edge_skill_optimizer/cli.py` - Command-line interface
- `tests/test_distiller.py` - Comprehensive tests
- `README.md` - Usage examples

### Security Considerations

- **On-Device Processing**: All optimization happens locally
- **Privacy**: No model data leaves the machine
- **Trusted Dependencies**: All verified via Context7
- **Apache 2.0**: Open source, community-driven
- **Verified Compression**: Accurate size calculations

### Next Steps

1. Real model distillation with PyTorch
2. ONNX runtime integration
3. iOS CoreML optimization
4. Android TensorFlow Lite export
5. Device testing and validation
6. Performance benchmarking

### Unique Defensible Moat

The **proprietary knowledge-distillation algorithm that perfectly transfers the reasoning capabilities of a large model into a tiny, edge-optimized architecture** is the key competitive advantage. This requires:

- Advanced distillation techniques
- Deep understanding of model architectures
- Device-specific optimization knowledge
- Extensive benchmarking
- Continuous algorithm refinement
- Hardware acceleration expertise

This is complex work requiring both AI and systems expertise, making it difficult to replicate effectively.

---

**This file should evolve as you learn more about the project.**
