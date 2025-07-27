# PDF Heading Detection System (LALIT)

An intelligent PDF heading detection system that uses machine learning to identify and extract document structure from PDF files. The system combines neural network models with rule-based approaches to provide accurate heading detection with hierarchical structure (H1, H2, H3).

## 🚀 Features

- **PDF Text Extraction**: Advanced PDF processing with font size and style analysis
- **Neural Network Model**: Custom PyTorch model for heading classification
- **Rule-based Fallback**: Intelligent fallback system when neural model is unavailable
- **Multiple PDF Support**: Batch processing capabilities
- **JSON Output**: Structured output format with heading levels and page numbers
- **Hugging Face Integration**: Pre-trained model downloading from Hugging Face Hub
- **Comprehensive Error Handling**: Robust error handling throughout the system

## 📋 Requirements

See `requirements.txt` for complete dependencies:

```
pdfplumber>=0.7.6
torch>=2.0.0
numpy>=1.21.0
huggingface-hub>=0.16.0
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/orange-19/lALIT.git
cd lALIT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Download pre-trained models:
```bash
python Lalitdownload.py
```

## 💻 Usage

### Basic Usage
```bash
python Lalit.py <pdf_file>
```

### Train Custom Model
```bash
python Lalit.py <pdf_file> --train
```

### Save Output to JSON
```bash
python Lalit.py <pdf_file> --json_out output.json
```

### Advanced Usage
```bash
python Lalit.py document.pdf --train --model custom_model.pth --json_out results.json
```

## 📊 Output Format

The system outputs heading detection results in the following JSON structure:

```json
[
  {
    "level": "H1",
    "text": "Introduction to Foundation Level Extensions",
    "page": 1
  },
  {
    "level": "H2", 
    "text": "Intended Audience",
    "page": 2
  }
]
```

## 🧪 Testing

Run comprehensive analysis and testing:

```bash
python comprehensive_analysis.py
```

This will:
- Test all dependencies
- Validate PDF processing functionality
- Test model training and prediction
- Generate detailed analysis report

## 📁 Project Structure

```
LalitProject/
├── Lalit.py                           # Main PDF heading detection script
├── Lalitdownload.py                   # Hugging Face model downloader
├── requirements.txt                   # Python dependencies
├── comprehensive_analysis.py          # Testing and analysis script
├── lalit_heading_model.pth           # Trained model file
├── my-model/                         # Downloaded Hugging Face models
├── *.pdf                             # Sample PDF files
└── README.md                         # This file
```

## 🔧 Technical Details

### Neural Network Architecture
- **Input Features**: Font size, vertical position, text length, font hash
- **Hidden Layer**: 32 neurons with ReLU activation
- **Output**: Multi-class classification (H1, H2, H3, Normal)
- **Framework**: PyTorch

### PDF Processing
- **Library**: pdfplumber for robust PDF text extraction
- **Features**: Font size, font name, text position analysis
- **Error Handling**: Graceful handling of malformed PDFs

### Heading Detection Algorithm
1. **Text Extraction**: Extract text with font metadata
2. **Feature Engineering**: Create numerical features from text properties
3. **Classification**: Use neural network or rule-based approach
4. **Post-processing**: Filter and structure results

## 🎯 Model Performance

- **Accuracy**: High accuracy on document structure detection
- **Fallback System**: Rule-based approach ensures reliability
- **Speed**: Fast processing for typical document sizes
- **Memory**: Efficient memory usage with PyTorch

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Open an issue on GitHub
- Check the comprehensive analysis report for troubleshooting
- Review the requirements.txt for dependency issues

## 🔄 Recent Updates

- ✅ Fixed KeyError in PDF extraction
- ✅ Added comprehensive error handling
- ✅ Improved heading detection accuracy
- ✅ Added model architecture consistency
- ✅ Created comprehensive testing suite
- ✅ Added Hugging Face model integration

---

**Built with ❤️ for intelligent document processing**
