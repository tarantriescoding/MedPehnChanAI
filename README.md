# 🏥 MedPehchaan AI+ - Intelligent Clinical Text Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![AI/ML](https://img.shields.io/badge/AI-Transformers-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Class_Project-blueviolet.svg)

**🎓 Class Project: Advanced AI-Powered Clinical Text Analysis**

*Transform clinical text into actionable medical insights using cutting-edge AI*

[🚀 Live Demo](#-quick-start) • [📖 Documentation](#-features) • [🛠️ Installation](#-installation)

</div>

---

## ⚠️ Medical Disclaimer

**This is an educational/research prototype for demonstration purposes only.**  
**NOT for clinical diagnosis, treatment decisions, or medical practice.**  
Always consult qualified healthcare professionals for medical advice.

---

## 🌟 Overview

MedPehchaan AI+ is a sophisticated web application that leverages state-of-the-art AI models to analyze clinical text and extract critical medical information. Built with modern web technologies and advanced natural language processing, it provides healthcare professionals and researchers with powerful tools for clinical text intelligence.

### 🎯 Key Capabilities

- **📝 Multi-format Input**: Process typed text, PDFs, CSVs, Excel files, JSONL datasets, and images with OCR
- **🔍 Advanced NER**: Biomedical entity extraction (Disease, Symptom, Medication, Procedure) using transformer models
- **📊 Risk Assessment**: Rule-based patient risk classification (High/Medium/Low)
- **💡 Clinical Insights**: Pattern-based medical insights and clinical observations
- **📋 Automated Summaries**: Entity-grounded clinical summaries with evidence linking
- **🖼️ Image Processing**: OCR support for clinical documents and handwritten notes
- **🎨 Modern UI**: Beautiful, responsive interface with real-time processing
- **📈 Analytics Dashboard**: Comprehensive patient-wise and aggregate analytics

---

## ✨ Features

### 🤖 AI-Powered Analysis
- **Biomedical NER**: Transformer-based extraction of diseases, symptoms, medications, and procedures
- **Confidence Scoring**: Quality assessment with confidence thresholds for all extracted entities
- **Noise Filtering**: Removes low-confidence and irrelevant text spans (threshold: 50%)
- **Context Preservation**: Maintains clinical meaning through intelligent preprocessing
- **Rule-Based Risk Classification**: Evidence-based triage with configurable risk thresholds
- **Pattern-Based Insights**: Generates clinical observations from extracted entity patterns

### 📊 Data Processing
- **Large Dataset Support**: Handles 100k+ patient records with optimized chunking
- **Memory-Optimized Batching**: Adaptive batch sizes based on dataset size
- **Multiple Input Formats**: CSV, TSV, Excel, PDF, TXT, JSONL support
- **Image Processing**: OCR-based text extraction from images and scanned documents
- **Streaming Mode**: For extremely large datasets with configurable flush intervals
- **Multiprocessing Support**: Parallel processing with worker pool optimization

### 🎨 User Experience
- **Modern UI**: Gradient-based design with glassmorphism effects
- **Real-time Processing**: Live progress indicators and status updates
- **Interactive Dashboard**: Patient-wise and aggregate analysis views
- **Download Reports**: PDF and CSV export capabilities

### 🔒 Safety & Quality
- **Medical Compliance**: Designed with healthcare standards in mind
- **Quality Assurance**: Built-in validation and error handling
- **Transparent Processing**: Clear visibility into AI decision-making
- **Educational Focus**: Optimized for learning and demonstration

---

## 🏗️ Architecture

```
MedPehchaan AI+
├── 🎨 UI Layer (ui.py - Streamlit)
│   ├── Multi-format Input Handling
│   ├── Real-time Processing with Progress Tracking
│   ├── Interactive Patient & Aggregate Dashboards
│   └── Report Export (PDF, CSV)
│
├── 🧠 Intelligence Engine (intelligence.py)
│   ├── Unified Processing Pipeline
│   ├── Batch Processing for Scalability
│   ├── Memory-Optimized Streaming
│   └── Result Aggregation & Analytics
│
├── 📝 Text Processing Pipeline
│   ├── preprocessing.py: Text normalization & patient record splitting
│   ├── ner_engine.py: Transformer-based biomedical NER
│   ├── postprocessing.py: Confidence filtering & entity validation
│   └── text_utils.py: Text utilities
│
├── 🎯 Clinical Analysis Engines
│   ├── risk_engine.py: Rule-based risk classification
│   ├── insight_engine.py: Pattern-based insight generation
│   ├── summary_engine.py: Entity-grounded summary generation
│   └── evaluation.py: Metrics computation & validation
│
├── 📊 Input/Output Processing
│   ├── pdf_utils.py: PDF text extraction & processing
│   ├── image_utils.py: Image preprocessing & OCR
│   ├── report_utils.py: PDF & CSV report generation
│   └── utils.py: General utilities
│
└── ⚙️ Configuration & Setup
    ├── config.py: Model configs, entity labels, risk rules, risk thresholds
    ├── requirements.txt: Dependencies
    └── app.py: Streamlit entry point
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tarantriescoding/MedPehnChanAI.git
   cd MedPehnChanAI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start analyzing clinical text!

---

## 📖 Usage Guide

### Input Methods

1. **📝 Typed Text**: Direct text input for quick analysis
2. **📎 File Upload**: Support for multiple formats:
   - `.txt` - Plain text files
   - `.pdf` - PDF documents
   - `.csv` - Comma-separated values
   - `.tsv` - Tab-separated values
   - `.xlsx` - Excel spreadsheets
   - `.jsonl` - JSON Lines format
3. **🖼️ Image Upload**: Clinical document images and scanned notes:
   - `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp` - OCR extraction
4. **🔄 Combined Input**: Mix any of the above for comprehensive analysis

### Analysis Workflow

1. **Input Preparation**: Accept text, files, or images
2. **Text Extraction**: Convert PDFs, images (OCR), and other formats to clean text
3. **Preprocessing**: Normalize text, detect and split patient records
4. **Entity Extraction**: Biomedical NER with confidence scoring
5. **Risk Assessment**: Rule-based triage using disease/symptom thresholds
6. **Insight Generation**: Pattern-based clinical observations
7. **Summary Generation**: Entity-grounded clinical summary creation
8. **Report Generation**: Downloadable PDF/CSV patient and aggregate reports

### Output Formats

- **Patient Reports**: Individual analysis including:
  - Extracted entities with confidence scores
  - Risk classification with evidence
  - Pattern-based clinical insights
  - Entity-grounded summary
  - Highlighted clinical text with entity annotations
  
- **Aggregate Analytics**: Population-level analysis
  - Risk distribution across patient cohort
  - Most common diseases, symptoms, medications, procedures
  - Entity frequency analysis
  - Visual charts and statistics
  
- **Export Options**: 
  - PDF reports with formatted clinical findings
  - CSV data export for downstream analysis
  - Interactive visualizations with Plotly

---

## 🛠️ Technical Details

### Dependencies
- **streamlit** (v1.35+): Modern web app framework
- **transformers** (v4.41+): Hugging Face transformers for biomedical NER
- **torch** (v2.2+): Deep learning framework
- **pandas** (v2.2+): Data manipulation and analysis
- **plotly** (v5.0+): Interactive visualizations
- **pypdf** (v4.2+): PDF text extraction
- **openpyxl** (v3.1+): Excel file processing
- **easyocr** (v1.6+): OCR for image-based clinical documents
- **opencv-python** (v4.10+): Image processing
- **beautifulsoup4** (v4.12+): HTML parsing
- **reportlab** (v4.2+): PDF report generation

### AI Models
- **Primary Biomedical NER**: `d4data/biomedical-ner-all` (main entity extraction model)
- **Clinical NER Fallback**: `samrawal/bert-base-uncased_clinical-ner` (alternative if primary fails)
- **Risk Classification**: Rule-based system with configurable disease/symptom thresholds
- **Summarization**: Entity-based pattern matching (non-generative)
- **Insight Generation**: Template-based clinical observations from extracted patterns

### Performance Features
- **Large Dataset Support**: Optimized for 100k+ records with adaptive chunking
- **Memory Efficient**: Streaming mode + garbage collection for large-scale analysis
- **GPU Support**: Automatic CUDA detection and acceleration when available
- **Batch Processing**: Configurable batch sizes (8-128) based on dataset size
- **Real-time Updates**: Live progress indicators with streaming output
- **Multiprocessing**: Worker pool for parallel extraction

---

## 📊 Sample Output

### Patient Analysis
```
Patient ID: PAT_001
Risk Level: Medium
Entities Found: 12
- Diseases: Diabetes, Hypertension
- Symptoms: Chest pain, Fatigue
- Medications: Aspirin, Metformin
- Procedures: ECG, Blood test

Clinical Insights:
• Monitor blood glucose levels closely
• Consider cardiovascular risk assessment
• Regular follow-up recommended
```

### Aggregate Analytics
- Total Patients: 1,247
- High Risk: 23% | Medium Risk: 45% | Low Risk: 32%
- Most Common Diseases: Diabetes (28%), Hypertension (22%)
- Average Entities per Patient: 8.3

---

## 🎓 Educational Value

This project demonstrates:

- **🤖 AI/ML Integration**: Real-world application of transformers
- **🏥 Healthcare AI**: Medical NLP and clinical decision support
- **🎨 UI/UX Design**: Modern web application development
- **📊 Data Engineering**: Large-scale data processing pipelines
- **🔒 Ethical AI**: Responsible AI development practices

### Learning Objectives
- Advanced Python programming
- Machine learning model deployment
- Web application development
- Healthcare data processing
- UI/UX design principles
- Software engineering best practices

---

## 🚀 Deployment

### Render Deployment (Recommended)

1. **Push to GitHub**: Upload your project to a GitHub repository

2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Create a new Web Service
   - Connect your GitHub repository

3. **Configure Build Settings**:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.headless true --server.runOnSave false`
   - **Service Name**: `medpehchaan-ai-clinical-text` (to match `https://medpehchaan-ai-clinical-text.onrender.com`)

4. **Environment Variables** (Optional):
   - Add `HF_TOKEN` if using Hugging Face authentication for faster downloads

5. **Deploy**: Click "Create Web Service" and wait for deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

### Alternative Platforms

- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Requires `Procfile` with `web: streamlit run app.py --server.port $PORT`
- **Vercel/Netlify**: Not recommended (not optimized for ML workloads)

---

## 🤝 Contributing

This is a class project, but contributions and feedback are welcome!

### Ways to Contribute
- 🐛 Bug reports and fixes
- ✨ Feature suggestions
- 📖 Documentation improvements
- 🎨 UI/UX enhancements
- 🔧 Performance optimizations

### Development Setup
```bash
# Fork the repository
# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
# Commit changes
git commit -m "Add amazing feature"

# Push to branch
git push origin feature/amazing-feature

# Create Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License - Academic/Research Use Only
Not for commercial medical applications without proper validation
```

---

## 🙏 Acknowledgments

- **Hugging Face** for transformer models and datasets
- **Streamlit** for the amazing web app framework
- **Open-source AI Community** for research and tools
- **Healthcare AI Research** for inspiration and guidance

---

## 📞 Contact

**Project Author**: Tarandeep Singh
**Institution**: Manav Rachna University , Faridabad 
**Course**: Btech CSE AIML
**Project**: Class Assignment - AI Clinical Text Intelligence

For questions or feedback:
- 📧 trackster45@gmail.com
- 🔗 https://in.linkedin.com/in/tarandeep-singh-1851a5284

---

<div align="center">

**Made with ❤️ for learning and healthcare innovation**

⭐ **Star this repository** if you found it helpful!

[⬆️ Back to Top](#-medpehchaan-ai--intelligent-clinical-text-intelligence-system)

</div>
```

## Demo Input

Use `data/sample_demo_input.txt` as a quick test sample.

## Notes

- This app is built to prefer **precision over recall**.
- If confidence is weak, entities are filtered or flagged instead of being forced.
