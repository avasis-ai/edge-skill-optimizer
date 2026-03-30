"""Command-line interface for Edge Skill Optimizer."""

import click
import json
from typing import Optional

from .distiller import (
    ModelQuantizer,
    KnowledgeDistiller,
    MobileOptimizer,
    OnnxExporter,
    QuantizationType
)


@click.group()
@click.version_option(version="0.1.0", prog_name="edge-optimizer")
def main() -> None:
    """Edge Skill Optimizer - Optimize AI models for mobile devices."""
    pass


@main.command()
@click.option("--skill", "-s", default="chat_skill", help="Skill name to optimize")
@click.option("--size", "-z", default=500.0, type=float, help="Original model size in MB")
@click.option("--device", "-d", default="iOS", type=click.Choice(["iOS", "Android", "Edge"]), help="Target device")
@click.option("--quantization", "-q", default="int8", type=click.Choice(["int8", "int4", "float16", "float32"]), help="Quantization type")
def optimize(skill: str, size: float, device: str, quantization: str) -> None:
    """Optimize a skill for mobile deployment."""
    optimizer = MobileOptimizer()
    
    # Convert quantization string to enum
    quant_type = QuantizationType(quantization)
    
    click.echo(f"\n🔧 Optimizing {skill} for {device}")
    click.echo("=" * 50)
    click.echo(f"Original Size: {size:.1f} MB")
    click.echo(f"Quantization: {quant_type.value}")
    click.echo(f"Target Device: {device}\n")
    
    # Optimize
    distillation, metadata = optimizer.optimize_skill(
        skill_name=skill,
        model_size_mb=size,
        target_device=device,
        quantization_type=quant_type,
        target_accuracy_loss=0.05
    )
    
    click.echo(f"✅ Optimization Complete")
    click.echo(f"   Distillation Result: {distillation.result_id}")
    click.echo(f"   Model ID: {metadata.model_id}")
    click.echo(f"   Original Size: {metadata.original_size_mb:.1f} MB")
    click.echo(f"   Optimized Size: {metadata.optimized_size_mb:.1f} MB")
    click.echo(f"   Compression: {metadata.compression_ratio:.1%}")
    click.echo(f"   Accuracy Loss: {metadata.accuracy_loss:.2%}")
    click.echo(f"   Parameter Count: {metadata.parameter_count:,}")


@main.command()
def demo() -> None:
    """Run optimization demo."""
    optimizer = MobileOptimizer()
    distiller = KnowledgeDistiller()
    quantizer = ModelQuantizer()
    
    click.echo("\n🧪 Edge Skill Optimizer Demo")
    click.echo("=" * 50)
    
    # Optimize multiple skills
    click.echo("\n📦 Optimizing Skills")
    click.echo("-" * 50)
    
    skills = [
        ("chat_skill", 500.0, "iOS", QuantizationType.INT8),
        ("vision_skill", 800.0, "Android", QuantizationType.INT4),
        ("nlp_skill", 300.0, "Edge", QuantizationType.FLOAT16)
    ]
    
    for skill_name, size, device, quant_type in skills:
        click.echo(f"\n⚡ Optimizing {skill_name}")
        distillation, metadata = optimizer.optimize_skill(
            skill_name=skill_name,
            model_size_mb=size,
            target_device=device,
            quantization_type=quant_type
        )
        
        click.echo(f"   ✅ Size: {size:.0f} MB → {metadata.optimized_size_mb:.1f} MB")
        click.echo(f"   📉 Compression: {metadata.compression_ratio:.1%}")
        click.echo(f"   🎯 Target: {device}")
    
    # Export to ONNX
    click.echo("\n📦 Exporting to ONNX")
    click.echo("-" * 50)
    
    exporter = OnnxExporter()
    quantized = optimizer._quantizer.get_quantized_models()
    
    for model in quantized:
        output_path = f"{model.model_id}.onnx"
        if exporter.export_model(model, output_path):
            click.echo(f"   ✅ {model.model_id} → {output_path}")
    
    # Show summary
    click.echo("\n📊 Optimization Summary")
    click.echo("-" * 50)
    
    summary = optimizer.get_optimization_summary()
    click.echo(f"Total Optimizations: {summary['total_optimizations']}")
    click.echo(f"Total Size Savings: {summary['total_size_savings_mb']:.1f} MB")
    click.echo(f"Average Compression: {summary['average_compression']:.1%}")


@main.command()
def stats() -> None:
    """Show optimization statistics."""
    optimizer = MobileOptimizer()
    distiller = KnowledgeDistiller()
    
    click.echo("\n📊 Optimization Statistics")
    click.echo("=" * 50)
    
    # Distillation stats
    results = distiller.get_results()
    if results:
        click.echo("\n🤖 Distillation Results")
        click.echo("-" * 50)
        for result in results:
            click.echo(f"\n  {result.result_id}")
            click.echo(f"     From: {result.source_model}")
            click.echo(f"     To: {result.student_model}")
            click.echo(f"     Epochs: {result.training_epochs}")
            click.echo(f"     Compression: {result.compression_factor:.1f}x")
    else:
        click.echo("\n🤖 No distillation results yet")
    
    # Quantization stats
    quantized = optimizer._quantizer.get_quantized_models()
    if quantized:
        click.echo("\n💾 Quantized Models")
        click.echo("-" * 50)
        for model in quantized:
            click.echo(f"\n  {model.model_id}")
            click.echo(f"     Size: {model.original_size_mb:.1f} MB → {model.optimized_size_mb:.1f} MB")
            click.echo(f"     Type: {model.quantization_type.value}")
            click.echo(f"     Device: {model.target_device}")
    else:
        click.echo("\n💾 No quantized models yet")


@main.command()
@click.option("--skill", "-s", required=True, help="Skill name to export")
@click.option("--device", "-d", required=True, type=click.Choice(["iOS", "Android", "Edge"]), help="Target device")
@click.option("--output", "-o", default="output.onnx", help="Output file path")
def export(skill: str, device: str, output: str) -> None:
    """Export optimized skill to ONNX format."""
    exporter = OnnxExporter()
    
    # Simulate model export
    click.echo(f"\n📦 Exporting {skill} for {device}")
    click.echo("=" * 50)
    click.echo(f"Output: {output}")
    click.echo(f"Format: ONNX\n")
    
    # Create simulated model metadata
    model_id = f"{skill}_optimized"
    model_data = {
        "model_id": model_id,
        "output_path": output,
        "format": "onnx",
        "size_mb": 50.0,  # Simulated
        "quantization": "int8",
        "timestamp": datetime.now().isoformat()
    }
    
    exporter._exported_models.append(model_data)
    
    click.echo(f"✅ Export successful")
    click.echo(f"   Model: {model_id}")
    click.echo(f"   Size: {model_data['size_mb']:.1f} MB")
    click.echo(f"   Path: {output}")


@main.command()
def help_text() -> None:
    """Show extended help information."""
    click.echo("""
Edge Skill Optimizer - Optimize AI Models for Mobile Devices

FEATURES:
  • Knowledge distillation for model compression
  • Quantization for edge deployment
  • ONNX export for cross-platform compatibility
  • Multi-device support (iOS, Android, Edge)
  • Accuracy-aware optimization

USAGE:
  edge-optimizer COMMAND [OPTIONS]
  
Commands:
  optimize      Optimize a skill for mobile
  demo          Run optimization demo
  stats         Show optimization statistics
  export        Export optimized model to ONNX
  config        Show configuration

OPTIONS:
  --skill, -s      Skill name
  --size, -z       Original model size in MB
  --device, -d     Target device (iOS, Android, Edge)
  --quantization, -q  Quantization type (int8, int4, float16)
  --output, -o     Output file path

EXAMPLES:
  edge-optimizer optimize --skill chat_skill --device iOS
  edge-optimizer demo
  edge-optimizer stats
  edge-optimizer export --skill chat_skill --device iOS --output chat_model.onnx

For more information, visit: https://github.com/avasis-ai/edge-skill-optimizer
    """)


def main_entry() -> None:
    """Main entry point."""
    main(prog_name="edge-optimizer")


if __name__ == "__main__":
    main_entry()
