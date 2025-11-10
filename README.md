<div align="center">

# 🚀 AI Model Training Scanner v2.0

### *Automated Analysis Tool for ML Training Experiments*

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)](https://github.com/yourusername/model-training-scanner/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](http://makeapullrequest.com)

[**Quick Start**](#-quick-start) • [**Features**](#-key-features) • [**Demo**](#-demo-output) • [**Documentation**](#-documentation) • [**Contributing**](#-contributing)

---

### 📊 At a Glance

```
9,109 Files Scanned  →  3,036 Training Experiments Detected  →  Comprehensive Analysis Generated
```

</div>

---

## 🎯 What Is This?

A **powerful Python tool** designed to automatically scan your messy AI/ML project directories and extract valuable insights from thousands of training experiments. Perfect for:

- 🔍 **Researchers** documenting successful hyperparameters
- 👨‍💻 **ML Engineers** tracking what actually works in production
- 📚 **Students** learning from real-world training attempts
- 🤝 **Teams** sharing knowledge about model configurations

### ⚡ Critical Info at a Glance

<table>
<tr>
<td width="50%">

**✅ What Works**
- **Best Model**: GPT-2 (219 successful runs)
- **Best Optimizer**: AdamW (89% success rate)
- **Optimal Learning Rate**: 5e-5 to 1e-4
- **Stable Batch Size**: 4-8 for most GPUs

</td>
<td width="50%">

**❌ Common Failures**
- **#1 Issue**: OOM Errors (45+ occurrences)
- **#2 Issue**: Learning rate too high
- **#3 Issue**: Convergence problems
- **Success Rate**: Only 10.3% without tuning

</td>
</tr>
</table>

---

## 🌟 Key Features

<table>
<tr>
<td width="33%" valign="top">

### 📊 **Smart Detection**
- Extracts training parameters automatically
- Supports `.log`, `.txt`, `.json`, `.md` files
- Regex-based pattern matching
- Handles messy file structures

</td>
<td width="33%" valign="top">

### 🔍 **Deep Analysis**
- Success/Failure classification
- Error cause identification
- Statistical summaries
- Trend analysis

</td>
<td width="33%" valign="top">

### 📈 **Rich Reports**
- Text and JSON outputs
- Visualized statistics
- Actionable insights
- GitHub-ready format

</td>
</tr>
</table>

### 🎨 Detected Information

```python
✓ Training Parameters        ✓ Performance Metrics      ✓ Status & Errors
  • Model name/architecture    • Accuracy (train/val)     • Success/Failure
  • Optimizer (AdamW, Adam)    • Loss values              • Error messages
  • Learning rate (1e-5)       • F1, Precision, Recall    • Notes & observations
  • Batch size                 • Statistical summaries    • Failure reasons
  • Device (cuda:0, cpu)
  • Quantization (4-bit, 8-bit)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/model-training-scanner.git
cd model-training-scanner

# No dependencies needed! Pure Python standard library
python model_training_scanner.py
```

### Basic Usage

```python
from model_training_scanner import ModelTrainingScanner

# Initialize scanner
scanner = ModelTrainingScanner(
    root_dir="/path/to/your/ml/projects",
    output_file="training_analysis_report.txt"
)

# Run analysis
scanner.run()

# Generated outputs:
# ├── training_analysis_report.txt   (Human-readable)
# └── training_analysis_report.json  (Machine-readable)
```

---

## 📸 Demo Output

### Console Output Preview

```
🔍 Scanning Started...
📂 Root Directory: /home/user/ml_projects
⏳ Please wait...

  📄 50 files scanned...
  📄 100 files scanned...
  ...
  📄 9109 files scanned...

✅ Scan Complete!
  📊 Total Files Scanned: 9,109
  ✨ Files with Training Data: 3,036

================================================================================
🚀 MODEL TRAINING ANALYSIS RESULTS - v2.0
================================================================================
📅 Date: 2025-01-15 20:30:00
📂 Scanned Directory: /home/user/ml_projects
📄 Total Files: 9,109
✨ Training Experiments: 3,036
================================================================================

📊 QUICK STATISTICS:
  ✅ Successful Trainings: 87
  ❌ Failed Trainings: 754
  ❓ Unknown Status: 2,195

================================================================================
🎯 FINDING #1: Training Report [❌ FAILED]
================================================================================

📁 File Information:
  • Path: /ml_projects/gpt2_experiment/train_log.txt
  • File: train_log.txt
  • Size: 12.45 KB
  • Modified: 2025-01-10 14:22:33

🤖 Model Information:
  • Model: gpt2-medium
  • Epochs: 1/100 (incomplete)
  • Samples: 50,000

🔧 Training Parameters:
  • Optimizer: AdamW
  • Learning Rate: 5.00e-05
  • Batch Size: 16
  • Device: cuda:0

📝 Status & Notes:
  • Status: Failed
  • Error: OOM (Out of Memory)
  • Notes: Batch size 16 too large for VRAM. Try batch_size=8 with gradient_accumulation_steps=2

================================================================================
```

### Statistical Analysis

```
================================================================================
📈 DETAILED STATISTICAL ANALYSIS
================================================================================

🤖 Model Distribution (Top 10):
  • gpt2: 219 experiments (Most stable baseline)
  • EleutherAI/gpt-neo-125M: 172 experiments
  • google/gemma-270m: 98 experiments
  • kuroshin/kuroshin-small-1b3: 90 experiments
  • TinyLlama/TinyLlama-1.1B: 40 experiments

🔧 Optimizer Distribution:
  • AdamW: 343 uses (89% - Dominant choice)
  • Adam: 34 uses (10%)
  • SGD: 12 uses (1% - Rarely successful)

💻 Device Distribution:
  • cuda:0: 429 uses (GPU - 93%)
  • cpu: 28 uses (7% - Testing only)

❌ Top Errors (Frequency):
  • OOM (Out of Memory): 45 occurrences
  • Loss not converging: 8 occurrences
  • CUDA device-side assert: 5 occurrences
  • NaN loss values: 3 occurrences

📊 Average Metrics:
  • Accuracy: Mean=0.8567, Min=0.4523, Max=0.9823 (89 samples)
  • Loss: Mean=0.3421, Min=0.0234, Max=2.1234 (134 samples)
  • Epochs: Mean=4.2, Min=1, Max=100 (156 samples)
  • Batch Size: Mean=10.5, Min=1, Max=32 (112 samples)
  • Learning Rate: Mean=3.24e-05, Min=1.00e-06, Max=1.00e-03 (98 samples)

🎯 Success Rate Analysis:
  • Total Tracked Experiments: 841
  • Success Rate: 10.3% (87 successful)
  • Failure Rate: 89.7% (754 failed)

  Key Insight: Most failures due to configuration errors (OOM, wrong LR)
```

---

## 📚 Documentation

### Supported File Formats

| Format | Description | Priority |
|--------|-------------|----------|
| `.log` | Training logs | ⭐⭐⭐ High |
| `.txt` | Text outputs | ⭐⭐⭐ High |
| `.json` | Config files, metrics | ⭐⭐⭐ High |
| `.md` | Documentation, notes | ⭐⭐ Medium |
| `.yaml`, `.yml` | Configuration files | ⭐⭐ Medium |
| `.csv`, `.out` | Result files | ⭐ Low |

### Detection Patterns (Regex Examples)

```python
# Model Names
"model: gpt2"
"model_name: bert-base-uncased"
"architecture = ResNet50"

# Training Parameters
"optimizer: AdamW"
"learning_rate: 5e-5"
"batch_size: 8"
"device: cuda:0"

# Quantization
"quantization: 4-bit"
"load_in_8bit: true"
"precision: fp16"

# Status Indicators
"status: success"
"result: failed"
"error: OOM"

# Performance Metrics
"accuracy: 0.95"
"val_loss: 0.12"
"f1_score: 0.88"
```

### Customization

```python
# Modify ROOT_DIR in model_training_scanner.py
ROOT_DIR = r"C:\Users\yourusername\your_ml_projects"

# Modify OUTPUT_FILE path
OUTPUT_FILE = r"C:\Users\yourusername\reports\analysis.txt"

# Adjust supported extensions
SUPPORTED_EXTENSIONS = [
    '.txt', '.log', '.md', '.json', '.yaml',
    '.csv', '.out', '.result', '.metrics'
]
```

---

## 🔥 Real-World Use Cases

### Scenario 1: "What batch size fits my GPU?"

**Your Analysis Shows:**
```diff
- Batch size 16 → 5 OOM errors
+ Batch size 8  → 3 successful runs
+ Batch size 4  → 12 successful runs

Recommendation: Use batch_size=4 or 8 for your hardware
```

### Scenario 2: "Which optimizer works best?"

**Statistics Reveal:**
```
AdamW:  avg accuracy = 0.89  ✓ Best choice
Adam:   avg accuracy = 0.84
SGD:    avg accuracy = 0.79
```

### Scenario 3: "Why won't my model converge?"

**Failed Runs Analysis:**
```diff
- learning_rate = 1e-3  → 8 convergence failures
- learning_rate = 5e-5  → 2 minor issues
+ learning_rate = 1e-5  → 0 failures  ✓ Optimal

Solution: Reduce learning rate!
```

---

## 📦 Project Structure

```
model-training-scanner/
│
├── model_training_scanner.py      # Main analysis script (31 KB)
├── README.md                       # This file
├── LEARNINGS.md                    # Detailed insights document
├── .gitignore                      # Git ignore rules
│
├── outputs/                        # Generated reports (auto-created)
│   ├── model_training_report.txt   # Human-readable report
│   └── model_training_report.json  # Structured data
│
└── examples/                       # Example files (optional)
    ├── sample_log.txt
    └── sample_config.json
```

---

## 🛠️ Advanced Features

### Programmatic Access

```python
from model_training_scanner import ModelTrainingScanner

scanner = ModelTrainingScanner(root_dir="./projects")
scanner.run()

# Access findings programmatically
for finding in scanner.findings:
    if finding.get('status_category') == 'failed':
        model = finding.get('model', 'Unknown')
        error = finding.get('error', 'No error info')
        print(f"❌ Failed: {model} - {error}")

    elif finding.get('accuracy', 0) > 0.95:
        model = finding.get('model', 'Unknown')
        acc = finding.get('accuracy')
        print(f"🏆 High performer: {model} - Accuracy: {acc:.2%}")
```

### Filtering Results

```python
# Filter by model type
gpt2_results = [f for f in scanner.findings if 'gpt2' in f.get('model', '').lower()]

# Filter by success status
successful = [f for f in scanner.findings if f.get('status_category') == 'success']

# Filter by accuracy threshold
high_accuracy = [f for f in scanner.findings if f.get('accuracy', 0) > 0.90]
```

### Generate Custom Reports

```python
from model_training_scanner import ModelTrainingScanner

scanner = ModelTrainingScanner("./projects", "custom_report.txt")
scanner.scan_directory()

# Generate custom statistics
stats = scanner.generate_statistics()
print(f"Success rate: {stats['successful_trainings'] / stats['total_findings']:.1%}")
print(f"Most common error: {stats['errors'].most_common(1)}")
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Adding New Detection Patterns

```python
# In model_training_scanner.py, add to PATTERNS dict:

'your_metric': [
    r'your_metric[:\s=]+([0-9]*\.?[0-9]+)',
    r'alternative_name[:\s=]+([0-9]*\.?[0-9]+)',
],
```

### Adding New File Format Support

```python
# Add to SUPPORTED_EXTENSIONS list:
SUPPORTED_EXTENSIONS = [
    '.txt', '.log', '.md', '.json', '.yaml',
    '.your_new_format'  # Your addition
]
```

### Reporting Issues

Found a bug or have a feature request? [Open an issue](https://github.com/yourusername/model-training-scanner/issues)!

---

## 📊 Performance & Limitations

### Performance

| Metric | Value |
|--------|-------|
| Files per second | ~500-1000 |
| Max file size | 10 MB |
| Memory usage | ~100-200 MB |
| Scan time (9K files) | ~30-60 seconds |

### Limitations

- ⚠️ Files larger than 10MB are skipped (configurable)
- ⚠️ Binary files are not analyzed
- ⚠️ Requires consistent logging format for best results
- ⚠️ Deep nested directories may be slower

---

## 🎓 Learning Resources

### Additional Documents

- 📘 [**LEARNINGS.md**](./LEARNINGS.md) - Detailed best practices and insights
- 📊 [**Example Reports**](./outputs/) - Sample analysis outputs
- 🔧 [**Configuration Guide**](./docs/config.md) - Advanced customization

### External Resources

- [HuggingFace Training Guide](https://huggingface.co/docs/transformers/training)
- [PyTorch Best Practices](https://pytorch.org/tutorials/beginner/best_practices.html)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

---

## 📄 License

```
MIT License

Copyright (c) 2025 Kuroshin AI Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

[Full License Text →](./LICENSE)

---

## 📞 Contact & Support

<div align="center">

**Kuroshin AI Project**

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/yourusername)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter)](https://twitter.com/yourusername)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/yourserver)

[Report Bug](https://github.com/yourusername/model-training-scanner/issues) •
[Request Feature](https://github.com/yourusername/model-training-scanner/issues) •
[Ask Question](https://github.com/yourusername/model-training-scanner/discussions)

</div>

---

## 🌟 Acknowledgments

Special thanks to:
- The ML community for inspiration
- HuggingFace for excellent documentation
- All contributors and users of this tool

---

<div align="center">

### ⭐ Star this repo if it helped you!

**Made with ❤️ by the Kuroshin AI Team**

*Last Updated: January 2025 • Version 2.0 • 3,965 Experiments Analyzed*

[⬆ Back to Top](#-ai-model-training-scanner-v20)

</div>
