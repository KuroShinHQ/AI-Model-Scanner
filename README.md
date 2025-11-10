# Model Eğitim Raporu Tarayıcı v2.0

## Genel Bakış

AI model eğitim deneyimlerinizi, parametrelerinizi, başarı/başarısızlık durumlarınızı ve öğrendiğiniz dersleri GitHub'da paylaşmak için tasarlanmış kapsamlı bir Python aracıdır. Bu script, tüm projelerinizi otomatik olarak tarayarak:

- **Eğitim Parametrelerini** (optimizer, learning rate, batch size, device, quantization)
- **Performans Metriklerini** (accuracy, loss, F1 score, precision, recall)
- **Başarı/Başarısızlık Durumlarını** (hangi denemeler çalıştı, hangilerinde sorun yaşandı)
- **Hata Sebeplerini** (OOM, overfitting, vb.)
- **Notlar ve Gözlemlerinizi**

analiz eder ve paylaşılabilir bir rapor oluşturur.

## Neden Bu Tool?

AI model eğitimi deneysel bir süreçtir. Hangi parametrelerin hangi koşullarda başarılı olduğunu, hangi hataların neden çıktığını ve neleri öğrendiğimizi dokümante etmek önemlidir. Bu tool:

- Geçmiş eğitim deneyimlerinizi organize eder
- Başarılı/başarısız denemeleri karşılaştırmanızı sağlar
- Toplulukla bilgi paylaşımını kolaylaştırır
- Hangi parametrelerin hangi modellerde çalıştığını görmenizi sağlar

## Özellikler

### Kapsamlı Parametre Tespiti

- **Model Bilgileri**: Model adı, mimari, base model
- **Eğitim Ayarları**: Epoch, sample sayısı, batch size
- **Optimizer Bilgisi**: AdamW, Adam, SGD, vb.
- **Learning Rate**: 1e-5 gibi scientific notation desteği
- **Device**: cuda:0, cpu, TPU vb.
- **Quantization**: 4-bit, 8-bit, QLoRA, vb.

### Metrik Analizi

- Accuracy (train, val, test)
- Loss değerleri
- F1 Score
- Precision ve Recall
- İstatistiksel özetler (ortalama, min, max)

### Durum Takibi

- ✅ **Başarılı Eğitimler**: Tamamlanan ve başarılı olan denemeler
- ❌ **Başarısız Eğitimler**: Hata veren veya yarıda kalan denemeler
- ❓ **Bilinmeyen Durum**: Status bilgisi olmayan denemeler

### Hata Analizi

- OOM (Out of Memory) hataları
- Overfitting/underfitting durumları
- Configuration hataları
- Convergence problemleri
- En sık karşılaşılan hatalar istatistiği

### İstatistiksel Analiz

- Model dağılımı (hangi modeller ne kadar kullanılmış)
- Optimizer tercihleri
- Device kullanımı (GPU/CPU)
- Başarı oranı analizi
- Ortalama parametre değerleri

## Kurulum

### Gereksinimler

- Python 3.7 veya üzeri
- Standart Python kütüphaneleri (ek kurulum gerekmez!)

```python
# Kullanılan kütüphaneler - hepsi Python'la birlikte gelir
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import Counter
```

### Dosyayı İndirme

```bash
# Git clone
git clone https://github.com/yourusername/model-training-scanner.git
cd model-training-scanner

# Veya doğrudan indirin
wget https://raw.githubusercontent.com/yourusername/model-training-scanner/main/model_training_scanner.py
```

## Kullanım

### Hızlı Başlangıç

1. Script'i projenizin içinde çalıştırın:

```bash
python model_training_scanner.py
```

2. Raporlar otomatik olarak oluşturulur:
   - `model_training_report.txt` - Okunabilir metin raporu
   - `model_training_report.json` - JSON formatında yapılandırılmış veri

### Özelleştirilmiş Kullanım

Script içindeki ayarları değiştirerek kendi projenize uyarlayın:

```python
# model_training_scanner.py içinde

# Taranacak dizin
ROOT_DIR = r"C:\Users\yourusername\your_project"

# Rapor çıktı yeri
OUTPUT_FILE = r"C:\Users\yourusername\your_project\report.txt"
```

### Programatik Kullanım

Script'i kendi Python kodunuzda kullanabilirsiniz:

```python
from model_training_scanner import ModelTrainingScanner

# Scanner oluştur
scanner = ModelTrainingScanner(
    root_dir="/path/to/your/projects",
    output_file="training_analysis.txt"
)

# Tarama yap
scanner.run()

# Bulgulara programatik erişim
for finding in scanner.findings:
    if finding.get('status_category') == 'failed':
        print(f"Başarısız: {finding.get('model')} - {finding.get('error')}")
```

## Desteklenen Dosya Formatları

Script aşağıdaki dosya türlerinden bilgi ayıklayabilir:

```
.txt, .log, .md      → Metin tabanlı loglar ve dokümantasyon
.json, .yaml, .yml   → Yapılandırma ve metrik dosyaları
.csv, .result        → Eğitim sonuç dosyaları
.out, .metrics       → Training output dosyaları
.report              → Rapor dosyaları
```

## Örnek Çıktı Formatı

### Konsol Çıktısı

```
================================================================================
🚀 MODEL EĞİTİM RAPORU TARAMA SONUÇLARI - GELİŞMİŞ VERSİYON
================================================================================
📅 Tarih: 2025-01-15 20:30:00
📂 Taranan Dizin: /home/user/ai_projects
📄 Toplam Taranan Dosya: 1,234
✨ Veri İçeren Dosya: 156
================================================================================

📊 HIZLI İSTATİSTİKLER:
  ✅ Başarılı Eğitimler: 89
  ❌ Başarısız Eğitimler: 23
  ❓ Bilinmeyen Durum: 44

================================================================================
🎯 BULGU #1: Model Eğitim Raporu [❌ BAŞARISIZ]
================================================================================

📁 Dosya Bilgileri:
  • Yol: /home/user/ai_projects/gpt2_oom_attempt/training_log.txt
  • Dosya Adı: training_log.txt
  • Boyut: 12.45 KB
  • Değiştirilme: 2025-01-10 14:22:33

🤖 Model Bilgileri:
  • Model: gpt2-medium
  • Epoch: 1
  • Sample Sayısı: 50,000

🔧 Eğitim Parametreleri:
  • Optimizer: AdamW
  • Learning Rate: 5.00e-05
  • Batch Size: 16
  • Device: cuda:0

📝 Durum ve Notlar:
  • Status: Başarısız
  • Hata: OOM (Out of Memory)
  • Notlar: Batch size 16 ile VRAM yetersiz. Gradient accumulation veya batch_size=8 denenmeli.

================================================================================
```

### İstatistiksel Analiz Bölümü

```
================================================================================
📈 DETAYLI İSTATİSTİKSEL ANALİZ
================================================================================

🤖 Model Dağılımı (Top 10):
  • gpt2: 45 eğitim
  • TinyLlama-1.1B: 23 eğitim
  • phi-2: 18 eğitim
  • bert-base-turkish: 12 eğitim

🔧 Optimizer Dağılımı:
  • AdamW: 89 kullanım
  • Adam: 34 kullanım
  • SGD: 12 kullanım

💻 Device Dağılımı:
  • cuda:0: 134 kullanım
  • cpu: 22 kullanım

❌ En Sık Karşılaşılan Hatalar (Top 5):
  • OOM (Out of Memory)... : 15 kez
  • Loss not converging... : 8 kez
  • CUDA error: device-side assert triggered... : 5 kez

📊 Ortalama Metrikler:
  • Accuracy: Ort=0.8567, Min=0.4523, Max=0.9823 (89 örnek)
  • Loss: Ort=0.3421, Min=0.0234, Max=2.1234 (134 örnek)
  • Epoch: Ort=4.2, Min=1, Max=100 (156 örnek)
  • Batch Size: Ort=10.5, Min=1, Max=32 (112 örnek)
  • Learning Rate: Ort=3.24e-05, Min=1.00e-06, Max=1.00e-03 (98 örnek)

🎯 Başarı Oranı Analizi:
  • Toplam Bilinen Durum: 112
  • Başarı Oranı: 79.5%
  • Başarısızlık Oranı: 20.5%
```

## Gerçek Kullanım Senaryoları

### Senaryo 1: "Hangi batch size GPU'ma sığar?"

Script'iniz şunu gösterir:
- Batch size 16 → 5 OOM hatası
- Batch size 8 → 3 başarılı eğitim
- Batch size 4 → 12 başarılı eğitim

**Sonuç**: GPU'nuz için ideal batch size = 4 veya 8

### Senaryo 2: "Hangi optimizer daha iyi?"

İstatistikler:
- AdamW ile ortalama accuracy: 0.89
- Adam ile ortalama accuracy: 0.84
- SGD ile ortalama accuracy: 0.79

**Sonuç**: AdamW bu modelde daha iyi performans gösteriyor

### Senaryo 3: "Neden model converge olmuyor?"

Başarısız eğitimlerde:
- Learning rate 1e-3 → 8 convergence hatası
- Learning rate 5e-5 → 2 convergence hatası
- Learning rate 1e-5 → 0 hata

**Sonuç**: Learning rate'i düşürmek gerekiyor

## JSON Çıktısı

Programatik kullanım için JSON raporu:

```json
{
  "scan_date": "2025-01-15T20:30:00",
  "root_directory": "/home/user/ai_projects",
  "total_scanned_files": 1234,
  "files_with_data": 156,
  "statistics": {
    "total_findings": 156,
    "successful_trainings": 89,
    "failed_trainings": 23,
    "unknown_status": 44,
    "models": {
      "gpt2": 45,
      "TinyLlama-1.1B": 23
    },
    "optimizers": {
      "AdamW": 89,
      "Adam": 34
    },
    "devices": {
      "cuda:0": 134,
      "cpu": 22
    },
    "errors": {
      "OOM (Out of Memory)": 15
    },
    "avg_metrics": {
      "accuracy": {
        "mean": 0.8567,
        "min": 0.4523,
        "max": 0.9823,
        "count": 89
      }
    }
  },
  "findings": [
    {
      "file_path": "/path/to/log.txt",
      "model": "gpt2-medium",
      "epoch": 5,
      "optimizer": "AdamW",
      "learning_rate": 5e-05,
      "batch_size": 8,
      "device": "cuda:0",
      "accuracy": 0.92,
      "loss": 0.15,
      "status": "başarılı",
      "status_category": "success"
    }
  ]
}
```

## Regex Pattern Örnekleri

Script aşağıdaki gibi çeşitli formatları algılar:

```python
# Model adı
"model: gpt2"
"model_name: bert-base-turkish"
"architecture = ResNet50"

# Parametreler
"optimizer: AdamW"
"learning_rate: 5e-5"
"lr = 0.0001"
"batch_size: 8"
"device: cuda:0"

# Quantization
"quantization: 4-bit"
"load_in_8bit: true"
"precision: fp16"

# Durum
"status: başarılı"
"result: failed"
"durum: completed"

# Hata
"error: OOM"
"hata: CUDA out of memory"
"exception: RuntimeError"

# Notlar
"note: Model converge olmadı, lr düşürülmeli"
"notlar: Batch size 4 ile çalıştı"
```

## Dosya Yapınız

Script tarandığında şöyle bir yapı bekler:

```
your_project/
│
├── experiments/
│   ├── gpt2_trial_1/
│   │   ├── training_log.txt          ← Taranır
│   │   ├── config.json                ← Taranır
│   │   └── results.csv                ← Taranır
│   │
│   ├── bert_finetuning/
│   │   ├── README.md                  ← Taranır
│   │   └── metrics.log                ← Taranır
│   │
│   └── failed_attempts/
│       └── oom_errors.txt             ← Taranır (başarısız olarak işaretlenir)
│
├── models/                            ← Checkpoint'ler (atlanır)
├── __pycache__/                       ← Atlanır
└── .git/                              ← Atlanır
```

## GitHub'da Paylaşım İçin İpuçları

### 1. Raporu README'nize Ekleyin

```markdown
## Model Eğitim Geçmişi

Bu projede 156 farklı eğitim denemesi yapılmıştır:
- ✅ 89 başarılı eğitim
- ❌ 23 başarısız deneme
- 🎯 %79.5 başarı oranı

En iyi sonuç: gpt2 + AdamW + lr=5e-5 + batch_size=8 → Accuracy: 0.95

Detaylı rapor için bkz: [model_training_report.txt](./model_training_report.txt)
```

### 2. Learnings Bölümü Oluşturun

```markdown
## Öğrendiklerim

### GPU Memory
- Batch size 16 → OOM (15 deneme)
- Batch size 8 → Çalışıyor ✓
- Gradient accumulation kullan!

### Learning Rate
- 1e-3 → Converge olmuyor
- 5e-5 → En iyi sonuç ✓
- 1e-6 → Çok yavaş öğreniyor

### Quantization
- 4-bit ile %2 accuracy kaybı
- Ancak 4x daha az VRAM kullanımı
- Small modeller için uygun
```

### 3. Issues Oluşturun

En sık hatalarınız için GitHub Issue'ları açın:

```markdown
Title: [SOLVED] OOM Error with batch_size=16
Labels: bug, solved, documentation

## Problem
gpt2-medium modeli batch_size=16 ile OOM veriyor

## Solution
- batch_size=8 kullan
- VEYA gradient_accumulation_steps=2 ekle

## Stats
15 deneme başarısız → 12 deneme başarılı ✓
```

## Katkıda Bulunma

### Yeni Pattern Ekleme

Kendi metriklerinizi eklemek için:

```python
# model_training_scanner.py içinde PATTERNS sözlüğüne ekleyin

'your_metric': [
    r'your_metric[:\s=]+([0-9]*\.?[0-9]+)',
    r'alternative_name[:\s=]+([0-9]*\.?[0-9]+)',
],
```

### Yeni Dosya Formatı Desteği

```python
# SUPPORTED_EXTENSIONS listesine ekleyin
SUPPORTED_EXTENSIONS = [
    '.txt', '.log', '.md', '.json', '.yaml',
    '.your_new_format'  # Yeni format
]
```

## Lisans

Bu proje açık kaynaklıdır (MIT License). İstediğiniz gibi kullanabilir, değiştirebilir ve paylaşabilirsiniz.

## Yazar & İletişim

**Kuroshin AI Project**

- GitHub: [@yourusername](https://github.com/yourusername)
- Proje: KuroshinPro AI Platform

## Changelog

### v2.0 (2025-01-15) - GitHub Paylaşım Versiyonu

**Yeni Özellikler:**
- ✨ Eğitim parametreleri tespiti (optimizer, lr, batch_size, device, quantization)
- ✨ Başarı/başarısızlık durumu analizi
- ✨ Hata sebepleri ve notlar ayıklama
- ✨ Detaylı istatistiksel analiz
- ✨ Başarı oranı hesaplama
- ✨ En sık hatalar listesi
- ✨ Model/optimizer/device dağılımı

**Geliştirmeler:**
- 🔧 Gelişmiş regex pattern'leri
- 🔧 JSON ve text dosyaları için özel parsing
- 🔧 Türkçe keyword desteği
- 🔧 Scientific notation support (1e-5)
- 🔧 Status kategorileme (success/failed/unknown)

**Düzeltmeler:**
- 🐛 JSON array hataları
- 🐛 Encoding sorunları
- 🐛 Tuple değer ayıklama

### v1.0 (2025-01-10) - İlk Sürüm
- Temel tarama özellikleri
- Model adı, epoch, sample tespiti
- Accuracy, loss, F1 metrikleri
- JSON ve text rapor çıktısı

---

**💡 Pro Tip**: Bu tool'u düzenli aralıklarla çalıştırarak eğitim geçmişinizi takip edin. Her deneme bir öğrenme fırsatıdır!

**🎯 Hedef**: AI modellerinizi eğitirken öğrendiklerinizi dokümante edin ve toplulukla paylaşın!
