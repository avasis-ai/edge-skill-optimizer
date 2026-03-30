# README.md - Edge Skill Optimizer

## Shrink Complex Agent Skills to Run Natively on Mobile Devices

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/edge-skill-optimizer.svg)](https://pypi.org/project/edge-skill-optimizer/)

**Edge Skill Optimizer** takes a complex SKILL.md workflow and distills it into a highly quantized, task-specific micro-model (e.g., 1B parameters). This specialized model runs flawlessly on iOS or Android NPUs without requiring cloud connectivity.

## 🎯 What It Does

This tool unlocks autonomous agents for native mobile applications by compressing large models into edge-optimized versions, with massive privacy benefits since data never leaves the phone.

### Example Use Case

```python
from edge_skill_optimizer.distiller import MobileOptimizer, QuantizationType

# Initialize optimizer
optimizer = MobileOptimizer()

# Optimize a skill for mobile
distillation, metadata = optimizer.optimize_skill(
    skill_name="chat_skill",
    model_size_mb=500.0,
    target_device="iOS",
    quantization_type=QuantizationType.INT8,
    target_accuracy_loss=0.05
)

print(f"✅ Optimized {skill_name}")
print(f"   Size: {metadata.original_size_mb:.1f} MB → {metadata.optimized_size_mb:.1f} MB")
print(f"   Compression: {metadata.compression_ratio:.1%}")
```

## 🚀 Features

- **Knowledge Distillation**: Transfer reasoning capabilities from large models to tiny architectures
- **Model Quantization**: Compress models for edge deployment with minimal accuracy loss
- **Multi-Device Support**: Optimize for iOS, Android, and edge devices
- **ONNX Export**: Cross-platform model export for deployment
- **Accuracy-Aware**: Balance compression with accuracy preservation
- **Mobile NPUs**: Optimized for neural processing units on mobile devices

### Supported Platforms

1. **iOS**: CoreML-optimized models for Apple devices
2. **Android**: TensorFlow Lite/NNAPI optimized models
3. **Edge Devices**: Generic ONNX models for edge computing

### Quantization Types

- **INT8**: 8-bit integer quantization (4x compression)
- **INT4**: 4-bit integer quantization (8x compression)
- **FLOAT16**: Half-precision floating point (2x compression)
- **FLOAT32**: Full precision (no compression, maximum accuracy)

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch for model operations
- ONNX for model export

### Install from PyPI

```bash
pip install edge-skill-optimizer
```

### Install from Source

```bash
git clone https://github.com/avasis-ai/edge-skill-optimizer.git
cd edge-skill-optimizer
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
pip install pytest pytest-mock black isort
```

## 🔧 Usage

### Command-Line Interface

```bash
# Check version
edge-optimizer --version

# Optimize a skill
edge-optimizer optimize --skill chat_skill --device iOS --size 500

# Run demo
edge-optimizer demo

# Show statistics
edge-optimizer stats

# Export model
edge-optimizer export --skill chat_skill --device iOS --output model.onnx
```

### Programmatic Usage

```python
from edge_skill_optimizer.distiller import (
    MobileOptimizer,
    QuantizationType,
    OnnxExporter
)

# Initialize optimizer
optimizer = MobileOptimizer()

# Optimize skill
distillation, metadata = optimizer.optimize_skill(
    skill_name="vision_skill",
    model_size_mb=800.0,
    target_device="Android",
    quantization_type=QuantizationType.INT4,
    target_accuracy_loss=0.1
)

# Export to ONNX
exporter = OnnxExporter()
exporter.export_model(metadata, "vision_model.onnx")

# Get optimization summary
summary = optimizer.get_optimization_summary()
print(f"Total savings: {summary['total_size_savings_mb']:.1f} MB")
```

### Advanced Usage

```python
from edge_skill_optimizer.distiller import KnowledgeDistiller, ModelQuantizer

# Custom distillation
distiller = KnowledgeDistiller()
result = distiller.distill_model(
    teacher_model="skill_large",
    student_model="skill_mobile",
    target_device="iOS",
    epochs=20,
    target_size_mb=100.0
)

# Custom quantization
quantizer = ModelQuantizer()
metadata = quantizer.quantize_model(
    model_id="custom_skill",
    original_size_mb=1000.0,
    quantization_type=QuantizationType.INT8,
    target_device="Edge",
    accuracy_loss=0.05
)
```

## 📚 API Reference

### MobileOptimizer

End-to-end optimizer for mobile deployment.

#### `optimize_skill(skill_name, model_size_mb, target_device, quantization_type, target_accuracy_loss)` → Tuple[DistillationResult, ModelMetadata]

Optimize a skill for mobile deployment.

#### `get_optimization_summary()` → Dict

Get summary of all optimizations.

### KnowledgeDistiller

Performs knowledge distillation.

#### `distill_model(teacher_model, student_model, target_device, epochs, target_size_mb)` → DistillationResult

Distill a teacher model into a student model.

### ModelQuantizer

Performs model quantization.

#### `quantize_model(model_id, original_size_mb, quantization_type, target_device, accuracy_loss)` → ModelMetadata

Quantize a model for edge deployment.

### OnnxExporter

Exports optimized models to ONNX format.

#### `export_model(model_metadata, output_path)` → bool

Export a model to ONNX format.

## 🧪 Testing

Run tests with pytest:

```bash
python -m pytest tests/ -v
```

## 📁 Project Structure

```
edge-skill-optimizer/
├── README.md
├── pyproject.toml
├── LICENSE
├── src/
│   └── edge_skill_optimizer/
│       ├── __init__.py
│       ├── distiller.py
│       └── cli.py
├── tests/
│   └── test_distiller.py
└── .github/
    └── ISSUE_TEMPLATE/
        └── bug_report.md
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Run tests**: `python -m pytest tests/ -v`
5. **Submit a pull request**

### Development Setup

```bash
git clone https://github.com/avasis-ai/edge-skill-optimizer.git
cd edge-skill-optimizer
pip install -e ".[dev]"
pre-commit install
```

## 📝 License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

## 🎯 Vision

Edge Skill Optimizer is an absolute necessity for mobile AI in 2026. It unlocks autonomous agents for native mobile applications with massive privacy benefits and operational independence from cloud services.

### Key Innovations

- **Knowledge Distillation**: Perfect transfer of reasoning capabilities
- **Edge Optimization**: Specialized architectures for mobile NPUs
- **Accuracy Preservation**: Balance between size and performance
- **Privacy First**: All processing on-device
- **Cross-Platform**: Works on iOS, Android, and edge devices

## 🌟 Impact

This tool enables:

- **On-Device AI**: No cloud dependency for AI processing
- **Privacy**: Data never leaves the device
- **Lower Latency**: Instant responses without network delay
- **Cost Reduction**: No cloud inference costs
- **Offline Capability**: Works without internet connection
- **Scalability**: Deploy to millions of devices without infrastructure

## 🛡️ Security & Trust

- **Trusted dependencies**: torch (7.1), onnx (9.4), click (8.8) - [Context7 verified](https://context7.com)
- **Apache 2.0**: Open source, community-driven
- **On-Device Processing**: No data exfiltration
- **Verified Compression**: Accurate size calculations
- **Open Source**: Community-reviewed code

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/avasis-ai/edge-skill-optimizer/wiki)
- **Issues**: [GitHub Issues](https://github.com/avasis-ai/edge-skill-optimizer/issues)
- **Enterprise**: contact@avasis.ai

## 🙏 Acknowledgments

- **TensorRT**: Model optimization inspiration
- **ONNX**: Cross-platform model format
- **LocalLLaMA**: Edge deployment inspiration
- **Apple CoreML**: iOS optimization standards
- **Android NNAPI**: Mobile NPU optimization

---

**Made with ❤️ by [Avasis AI](https://avasis.ai)**

*The essential tool for mobile AI. Compress models, enable on-device AI, protect privacy.*
