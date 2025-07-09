# Atria Core

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Atria Core** is a PyTorch-based toolkit designed for training and testing machine learning and deep learning models at scale. It provides a comprehensive framework for handling document analysis, OCR processing, computer vision tasks, and scalable data processing pipelines.

## 🚀 Features

### Core Data Models
- **Flexible Data Models**: Pydantic-based data models with automatic validation and serialization
- **Batching Support**: Intelligent batching of heterogeneous data types including tensors, images, and text
- **Device Management**: Seamless GPU/CPU tensor management with automatic device conversion
- **Table Serialization**: PyArrow integration for efficient data storage and retrieval

### Document & OCR Processing
- **OCR Integration**: Built-in support for Tesseract OCR with HOCR format parsing
- **Document Analysis**: Comprehensive document instance handling with images and OCR data
- **Bounding Box Operations**: Advanced bounding box manipulation and coordinate system conversion
- **Graph Parsing**: NetworkX-based OCR graph parsing for hierarchical text analysis

### Machine Learning Infrastructure
- **Tensor Operations**: Advanced tensor manipulation with device-aware operations
- **Repeatable Experiments**: Built-in support for reproducible model training and evaluation
- **Dataset Management**: Flexible dataset configuration and metadata handling
- **Ground Truth Processing**: Structured ground truth data for various ML tasks

### Utilities & Tools
- **Rich Representations**: Beautiful console output with rich formatting
- **Comprehensive Logging**: Structured logging with configurable levels and colors
- **Type Safety**: Full type annotations with runtime validation
- **Testing Framework**: Extensive test coverage with factory-based data generation

## 📦 Installation

### Requirements
- Python 3.11
- PyTorch 2.1.2

### Install from source
```bash
git clone https://github.com/saifullah3396/atria_core.git
cd atria_core
```

### CPU-only installation
```bash
pip install -e ".[torch-cpu]"
```

### GPU installation (CUDA 12.1)
```bash
pip install -e ".[torch-gpu]"
```

### Development installation
```bash
uv sync --group dev
```

## 🏗️ Quick Start

### Basic Document Processing

```python
from atria_core.types import DocumentInstance, OCR, Image
from atria_core.types.common import OCRType

# Create a document instance with image and OCR
doc = DocumentInstance(
    page_id=0,
    image=Image(file_path="document.jpg"),
    ocr=OCR(
        file_path="document.hocr", 
        type=OCRType.tesseract
    )
)

# Process the document (loads OCR automatically)
processed_doc = doc.model_validate(doc.model_dump())
print(f"Extracted {len(processed_doc.gt.ocr.words)} words")
```

### Batch Processing

```python
from atria_core.types import DocumentInstance

# Create multiple document instances
documents = [
    DocumentInstance(page_id=i, image=Image(file_path=f"doc_{i}.jpg"))
    for i in range(10)
]

# Batch them together
batched_docs = DocumentInstance.batched(documents)
print(f"Batched {batched_docs.batch_size} documents")

# Convert to tensors and move to GPU
tensor_docs = batched_docs.to_tensor().to_device("cuda")
```

### OCR Processing

```python
from atria_core.types.ocr_parsers.hocr_parser import OCRProcessor
from atria_core.types.common import OCRType

# Parse HOCR content
hocr_content = """<html>...</html>"""  # Your HOCR content
ocr_result = OCRProcessor.parse(hocr_content, OCRType.tesseract)

print(f"Words: {ocr_result.words}")
print(f"Bounding boxes: {ocr_result.word_bboxes}")
print(f"Confidence scores: {ocr_result.word_confs}")
```

### Custom Data Models

```python
from typing import Annotated
import torch
import pyarrow as pa
from atria_core.types.base.data_model import RawDataModel
from atria_core.utilities.encoding import TableSchemaMetadata

class CustomModel(RawDataModel):
    name: Annotated[str, TableSchemaMetadata(pyarrow=pa.string())]
    features: torch.Tensor
    confidence: Annotated[float, TableSchemaMetadata(pyarrow=pa.float32())]

# Use all the built-in functionality
model = CustomModel(name="example", features=torch.randn(10), confidence=0.95)

# Batch multiple instances
models = [CustomModel(...) for _ in range(100)]
batched = CustomModel.batched(models)

# Convert to table format
row_data = model.to_row()
schema = CustomModel.pa_schema()
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests with coverage
./scripts/test.sh

# Run specific test modules
pytest tests/types/
pytest tests/utilities/
```

## 🛠️ Development

### Code Quality

The project uses several tools to maintain code quality:

```bash
# Format code
./scripts/format.sh

# Lint code
./scripts/lint.sh

# Run all checks
./scripts/test.sh
```

### Project Structure

```
atria_core/
├── src/atria_core/           # Main package
│   ├── types/                # Data models and type definitions
│   │   ├── base/            # Base classes and mixins
│   │   ├── data_instance/   # Document and image instances
│   │   ├── generic/         # Reusable data types
│   │   └── ocr_parsers/     # OCR processing utilities
│   ├── utilities/           # Common utilities
│   └── logger/              # Logging infrastructure
├── tests/                   # Test suite
├── scripts/                 # Development scripts
└── htmlcov/                # Coverage reports
```

## 📚 Core Concepts

### Data Models

Atria Core uses Pydantic-based data models that provide:

- **Type Safety**: Runtime validation of data types
- **Automatic Serialization**: JSON and binary serialization support
- **Batching**: Automatic batching of multiple instances
- **Device Management**: GPU/CPU tensor handling
- **Table Support**: PyArrow integration for efficient storage

### Mixins

The framework provides several powerful mixins:

- `Batchable`: Batch multiple instances together
- `Repeatable`: Repeat and gather operations for beam search
- `ToDeviceConvertible`: Move tensors between devices
- `TableSerializable`: Convert to/from tabular formats
- `Loadable`: Lazy loading of file-based data

### OCR Pipeline

1. **Input**: HOCR files from Tesseract or similar OCR engines
2. **Parsing**: Extract words, bounding boxes, and confidence scores
3. **Normalization**: Convert coordinates to relative values
4. **Integration**: Combine with images in document instances
5. **Processing**: Batch and tensor conversion for ML pipelines

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:

- Code style and formatting
- Testing requirements
- Documentation standards
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 👥 Authors

- **Saifullah** - *Initial work* - [saifullah.saifullah@dfki.de](mailto:saifullah.saifullah@dfki.de)

## 🔗 Links

- [Homepage](https://github.com/saifullah3396/atria_core/)
- [Bug Reports](https://github.com/saifullah3396/atria_core/issues)
- [Source Code](https://github.com/saifullah3396/atria_core/)

## 📈 Roadmap

- [ ] Additional OCR engine support (PaddleOCR, EasyOCR)
- [ ] Enhanced document layout analysis
- [ ] More pre-trained model integrations
- [ ] Distributed training utilities
- [ ] Web-based annotation tools

---

Built with ❤️ for the machine learning and document analysis community.