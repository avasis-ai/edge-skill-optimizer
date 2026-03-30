"""Tests for Edge Skill Optimizer."""

import pytest
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from edge_skill_optimizer.distiller import (
    KnowledgeDistiller,
    ModelQuantizer,
    MobileOptimizer,
    OnnxExporter,
    ModelMetadata,
    DistillationResult,
    QuantizationType
)


class TestKnowledgeDistiller:
    """Tests for KnowledgeDistiller."""
    
    def test_initialization(self):
        """Test distiller initialization."""
        distiller = KnowledgeDistiller()
        
        assert len(distiller.get_results()) == 0
    
    def test_distill_model(self):
        """Test distilling a model."""
        distiller = KnowledgeDistiller()
        
        result = distiller.distill_model(
            teacher_model="chat_model_large",
            student_model="chat_model_mobile",
            target_device="iOS",
            epochs=10,
            target_size_mb=50.0
        )
        
        assert result.result_id is not None
        assert result.source_model == "chat_model_large"
        assert result.student_model == "chat_model_mobile"
        assert result.training_epochs == 10
        assert result.optimized_for == "iOS"
    
    def test_distill_multiple_models(self):
        """Test distilling multiple models."""
        distiller = KnowledgeDistiller()
        
        result1 = distiller.distill_model("model1_large", "model1_mobile", "iOS", 10)
        result2 = distiller.distill_model("model2_large", "model2_mobile", "Android", 15)
        
        results = distiller.get_results()
        
        assert len(results) == 2
        assert results[0].source_model == "model1_large"
        assert results[1].source_model == "model2_large"
    
    def test_distillation_metrics(self):
        """Test distillation metrics."""
        distiller = KnowledgeDistiller()
        
        result = distiller.distill_model(
            teacher_model="test_large",
            student_model="test_mobile",
            target_device="iOS",
            epochs=20
        )
        
        # Metrics should be positive
        assert result.final_loss > 0
        assert result.accuracy_improvement > 0
        assert result.compression_factor > 1


class TestModelQuantizer:
    """Tests for ModelQuantizer."""
    
    def test_quantize_model_float32(self):
        """Test quantization with float32."""
        quantizer = ModelQuantizer()
        
        metadata = quantizer.quantize_model(
            model_id="test_float32",
            original_size_mb=100.0,
            quantization_type=QuantizationType.FLOAT32,
            target_device="iOS",
            accuracy_loss=0.0
        )
        
        assert metadata.optimized_size_mb == 100.0
        assert metadata.compression_ratio == 0.0
    
    def test_quantize_model_float16(self):
        """Test quantization with float16."""
        quantizer = ModelQuantizer()
        
        metadata = quantizer.quantize_model(
            model_id="test_float16",
            original_size_mb=100.0,
            quantization_type=QuantizationType.FLOAT16,
            target_device="Android",
            accuracy_loss=0.05
        )
        
        assert metadata.optimized_size_mb == 50.0
        assert metadata.compression_ratio == 0.5
    
    def test_quantize_model_int8(self):
        """Test quantization with int8."""
        quantizer = ModelQuantizer()
        
        metadata = quantizer.quantize_model(
            model_id="test_int8",
            original_size_mb=200.0,
            quantization_type=QuantizationType.INT8,
            target_device="Edge",
            accuracy_loss=0.1
        )
        
        assert metadata.optimized_size_mb == 50.0
        assert metadata.compression_ratio == 0.75
    
    def test_quantize_model_int4(self):
        """Test quantization with int4."""
        quantizer = ModelQuantizer()
        
        metadata = quantizer.quantize_model(
            model_id="test_int4",
            original_size_mb=160.0,
            quantization_type=QuantizationType.INT4,
            target_device="iOS",
            accuracy_loss=0.15
        )
        
        assert metadata.optimized_size_mb == 20.0
        assert metadata.compression_ratio == 0.875
    
    def test_get_quantized_models(self):
        """Test getting quantized models."""
        quantizer = ModelQuantizer()
        
        quantizer.quantize_model("model1", 100.0, QuantizationType.INT8, "iOS", 0.05)
        quantizer.quantize_model("model2", 200.0, QuantizationType.INT4, "Android", 0.1)
        
        models = quantizer.get_quantized_models()
        
        assert len(models) == 2
        assert models[0].model_id == "model1"
        assert models[1].model_id == "model2"


class TestMobileOptimizer:
    """Tests for MobileOptimizer."""
    
    def test_optimize_skill(self):
        """Test optimizing a skill."""
        optimizer = MobileOptimizer()
        
        distillation, metadata = optimizer.optimize_skill(
            skill_name="chat_skill",
            model_size_mb=500.0,
            target_device="iOS",
            quantization_type=QuantizationType.INT8
        )
        
        assert distillation.result_id is not None
        assert metadata.model_id is not None
        assert metadata.original_size_mb == 500.0
    
    def test_optimize_multiple_skills(self):
        """Test optimizing multiple skills."""
        optimizer = MobileOptimizer()
        
        distillation1, metadata1 = optimizer.optimize_skill("skill1", 300.0, "Android")
        distillation2, metadata2 = optimizer.optimize_skill("skill2", 400.0, "Edge")
        
        assert metadata1.model_id != metadata2.model_id
        assert metadata1.original_size_mb != metadata2.original_size_mb
    
    def test_optimization_summary(self):
        """Test getting optimization summary."""
        optimizer = MobileOptimizer()
        
        optimizer.optimize_skill("skill1", 100.0, "iOS")
        optimizer.optimize_skill("skill2", 200.0, "Android")
        
        summary = optimizer.get_optimization_summary()
        
        assert summary["total_optimizations"] == 2
        assert summary["total_size_savings_mb"] > 0
        assert summary["average_compression"] > 0
    
    def test_quantization_types(self):
        """Test different quantization types."""
        optimizer = MobileOptimizer()
        
        # Int8 quantization
        _, metadata_int8 = optimizer.optimize_skill(
            "skill_int8", 100.0, "iOS", QuantizationType.INT8
        )
        
        # Int4 quantization
        _, metadata_int4 = optimizer.optimize_skill(
            "skill_int4", 100.0, "Android", QuantizationType.INT4
        )
        
        # Int4 should have higher compression than Int8
        assert metadata_int4.compression_ratio > metadata_int8.compression_ratio


class TestOnnxExporter:
    """Tests for OnnxExporter."""
    
    def test_export_model(self):
        """Test exporting model to ONNX."""
        exporter = OnnxExporter()
        
        metadata = ModelMetadata(
            model_id="test_model",
            original_size_mb=100.0,
            optimized_size_mb=25.0,
            compression_ratio=0.75,
            quantization_type=QuantizationType.INT8,
            target_device="iOS",
            accuracy_loss=0.05,
            parameter_count=25000000,
            timestamp=datetime.now()
        )
        
        result = exporter.export_model(metadata, "output.onnx")
        
        assert result is True
        assert len(exporter.get_export_history()) == 1
    
    def test_export_history(self):
        """Test export history tracking."""
        exporter = OnnxExporter()
        
        metadata = ModelMetadata(
            model_id="model1",
            original_size_mb=100.0,
            optimized_size_mb=25.0,
            compression_ratio=0.75,
            quantization_type=QuantizationType.INT8,
            target_device="iOS",
            accuracy_loss=0.05,
            parameter_count=25000000,
            timestamp=datetime.now()
        )
        
        exporter.export_model(metadata, "model1.onnx")
        
        history = exporter.get_export_history()
        
        assert len(history) == 1
        assert history[0]["model_id"] == "model1"


class TestModelMetadata:
    """Tests for ModelMetadata."""
    
    def test_metadata_creation(self):
        """Test creating model metadata."""
        metadata = ModelMetadata(
            model_id="test_model",
            original_size_mb=100.0,
            optimized_size_mb=25.0,
            compression_ratio=0.75,
            quantization_type=QuantizationType.INT8,
            target_device="iOS",
            accuracy_loss=0.05,
            parameter_count=25000000,
            timestamp=datetime.now()
        )
        
        assert metadata.model_id == "test_model"
        assert metadata.original_size_mb == 100.0
        assert metadata.compression_ratio == 0.75
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ModelMetadata(
            model_id="test_model",
            original_size_mb=100.0,
            optimized_size_mb=25.0,
            compression_ratio=0.75,
            quantization_type=QuantizationType.INT8,
            target_device="iOS",
            accuracy_loss=0.05,
            parameter_count=25000000,
            timestamp=datetime.now()
        )
        
        data = metadata.to_dict()
        
        assert data["model_id"] == "test_model"
        assert data["quantization_type"] == "int8"
        assert data["target_device"] == "iOS"


class TestDistillationResult:
    """Tests for DistillationResult."""
    
    def test_result_creation(self):
        """Test creating distillation result."""
        result = DistillationResult(
            result_id="distill_001",
            source_model="teacher_model",
            student_model="student_model",
            training_epochs=10,
            final_loss=0.15,
            accuracy_improvement=0.05,
            compression_factor=10.0,
            optimized_for="iOS",
            timestamp=datetime.now()
        )
        
        assert result.result_id == "distill_001"
        assert result.training_epochs == 10
        assert result.compression_factor == 10.0
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = DistillationResult(
            result_id="distill_002",
            source_model="teacher",
            student_model="student",
            training_epochs=20,
            final_loss=0.12,
            accuracy_improvement=0.08,
            compression_factor=15.0,
            optimized_for="Android",
            timestamp=datetime.now()
        )
        
        data = result.to_dict()
        
        assert data["result_id"] == "distill_002"
        assert data["training_epochs"] == 20
        assert data["optimized_for"] == "Android"


class TestQuantizationType:
    """Tests for QuantizationType enum."""
    
    def test_quantization_values(self):
        """Test quantization type values."""
        assert QuantizationType.INT8.value == "int8"
        assert QuantizationType.INT4.value == "int4"
        assert QuantizationType.FLOAT16.value == "float16"
        assert QuantizationType.FLOAT32.value == "float32"


class TestCompressionCalculations:
    """Tests for compression calculations."""
    
    def test_100mb_int8(self):
        """Test compression of 100MB model with int8."""
        quantizer = ModelQuantizer()
        
        metadata = quantizer.quantize_model(
            model_id="test",
            original_size_mb=100.0,
            quantization_type=QuantizationType.INT8,
            target_device="iOS",
            accuracy_loss=0.05
        )
        
        assert metadata.optimized_size_mb == 25.0
        assert metadata.compression_ratio == 0.75
    
    def test_200mb_int4(self):
        """Test compression of 200MB model with int4."""
        quantizer = ModelQuantizer()
        
        metadata = quantizer.quantize_model(
            model_id="test",
            original_size_mb=200.0,
            quantization_type=QuantizationType.INT4,
            target_device="Android",
            accuracy_loss=0.1
        )
        
        assert metadata.optimized_size_mb == 25.0
        assert metadata.compression_ratio == 0.875
    
    def test_zero_size(self):
        """Test compression of zero-size model."""
        quantizer = ModelQuantizer()
        
        metadata = quantizer.quantize_model(
            model_id="test",
            original_size_mb=0.0,
            quantization_type=QuantizationType.INT8,
            target_device="iOS",
            accuracy_loss=0.0
        )
        
        assert metadata.optimized_size_mb == 0.0
        assert metadata.compression_ratio == 0.0
