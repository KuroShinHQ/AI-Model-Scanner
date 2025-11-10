# Model Eğitim Deneyimleri ve Öğrenilenler

Bu doküman, KuroshinPro AI Platform geliştirme sürecinde yapılan model eğitim denemelerinden edinilen tecrübeleri, başarılı stratejileri ve karşılaşılan sorunları içerir.

## Genel İstatistikler

📊 **Toplam Analiz:**
- **Taranan Dosya**: 9,109
- **Eğitim Verisi İçeren Dosya**: 3,036
- **Benzersiz Model Denemesi**: 3,965

🎯 **Başarı Durumu:**
- ✅ Başarılı Eğitimler: 87 (10.3%)
- ❌ Başarısız/Sorunlu: 754 (89.7%)
- ❓ Durum Belirsiz: 3,124

> **Not**: Yüksek başarısızlık oranı, deneysel geliştirme sürecinin doğal bir parçasıdır. Her başarısızlık yeni bir öğrenme fırsatıdır!

## En Çok Kullanılan Modeller

### Top 5 Model

1. **GPT-2** → 219 deneme
   - En stabil ve test edilmiş model
   - Türkçe fine-tuning için uygun
   - Resource-efficient

2. **EleutherAI/gpt-neo-125M** → 172 deneme
   - Küçük ama güçlü alternatif
   - GPT-2'den daha iyi performans
   - CPU'da bile çalışabilir

3. **Kuroshin Small 1.3B** → 72 deneme
   - Özel geliştirilmiş model
   - Türkçe optimize
   - İyi accuracy/size oranı

4. **TinyLlama 1.1B** → 40 deneme
   - Son zamanlarda popüler
   - Hızlı eğitim
   - Düşük memory footprint

5. **Microsoft Phi-2** → 18 deneme
   - Çok yeni denemeler
   - Promising sonuçlar
   - Daha fazla test gerekiyor

## Optimizer Deneyimleri

### En Başarılı Optimizers

**AdamW** ✅
- Neredeyse tüm başarılı eğitimlerde kullanıldı
- Learning rate'e toleranslı
- Default choice olarak öneriliyor

**Adam** ⚠️
- AdamW'den biraz daha az stabil
- Bazı modellerde overfitting
- Weight decay ile birlikte kullanılmalı

**SGD** ❌
- Genelde yavaş convergence
- Daha aggressive learning rate scheduler gerekiyor
- Momentum ile kullanılmazsa zor

## Device & Hardware

### GPU Kullanımı

**CUDA (GPU)** → 429 kullanım
- Açık ara en çok kullanılan
- 10-50x hızlanma
- OOM hataları en büyük sorun

**CPU** → 28 kullanım
- Küçük modeller için OK
- Test ve debug için kullanışlı
- Production için çok yavaş

### OOM (Out of Memory) Hataları

En sık karşılaşılan sorun! **45+ OOM hatası**

#### Çözümler:

```python
# ❌ BAŞARISIZ
batch_size = 16  # OOM!
gradient_accumulation_steps = 1

# ✅ BAŞARILI
batch_size = 4  # veya 8
gradient_accumulation_steps = 4  # Effective batch = 16
```

#### Memory Optimizasyon Taktikleri:

1. **Quantization Kullan**
   ```python
   load_in_8bit=True  # %50 memory tasarrufu
   load_in_4bit=True  # %75 memory tasarrufu
   ```

2. **Gradient Checkpointing**
   ```python
   model.gradient_checkpointing_enable()
   # Memory ↓30%, Speed ↓20%
   ```

3. **Mixed Precision Training**
   ```python
   from torch.cuda.amp import autocast
   # Memory ↓40%, Speed ↑30%
   ```

## Learning Rate Stratejileri

### Öğrenilenler

📉 **Çok Yüksek LR (1e-3)**
- Loss explode ediyor
- Model converge olmuyor
- NaN değerleri oluşuyor

✅ **Optimal Range (5e-5 to 1e-4)**
- Stabil eğitim
- İyi convergence
- Çoğu model için ideal

📈 **Çok Düşük LR (1e-6)**
- Çok yavaş öğrenme
- Sonsuz epoch gerekiyor
- Sabır testi!

### Learning Rate Scheduler

**Warmup + Cosine Annealing** → En başarılı stratejİ

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,  # Total steps'in %10'u
    num_training_steps=1000
)
```

## Batch Size Deneyimleri

### GPU Memory'e Göre Öneriler

| GPU VRAM | Model Size | Önerilen Batch Size |
|----------|-----------|---------------------|
| 4GB      | Small (<500M) | 1-2 |
| 8GB      | Small-Medium | 4-8 |
| 12GB     | Medium (1B) | 8-16 |
| 16GB+    | Large (3B+) | 16-32 |

### Gradient Accumulation

Küçük batch size kullanıyorsanız mutlaka gradient accumulation ekleyin:

```python
effective_batch_size = batch_size * gradient_accumulation_steps

# Örnek:
batch_size = 4
gradient_accumulation_steps = 8
# → Effective batch = 32
```

## Quantization Deneyimleri

### 4-bit Quantization (QLoRA)

**Avantajlar:**
- ✅ %75 memory tasarrufu
- ✅ Büyük modelleri küçük GPU'larda çalıştırma
- ✅ Hala fine-tune edilebilir

**Dezavantajlar:**
- ❌ %2-5 accuracy kaybı
- ❌ Biraz daha yavaş
- ❌ Inference için dequantization gerekebilir

**Örnek:**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=bnb_config
)
```

### 8-bit Quantization

- Daha dengeli: %50 memory, %1-2 accuracy kaybı
- Production için daha uygun
- Hızlı inference

## En Sık Hatalar ve Çözümleri

### 1. CUDA Out of Memory (OOM)

**Belirtiler:**
```
RuntimeError: CUDA out of memory.
Tried to allocate 2.50 GiB
```

**Çözümler:**
1. Batch size'ı düşür
2. Gradient accumulation kullan
3. Gradient checkpointing aç
4. Quantization kullan
5. Model'i değiştir (daha küçük)

### 2. Loss Not Converging

**Belirtiler:**
- Loss düşmüyor
- Veya çok yavaş düşüyor
- Plateau yapıyor

**Çözümler:**
1. Learning rate'i ayarla
2. Warmup steps ekle
3. Data quality'yi kontrol et
4. Overfitting var mı kontrol et

### 3. NaN Loss

**Belirtiler:**
```
Step 234: loss = nan
```

**Çözümler:**
1. Learning rate'i düşür (genelde bu!)
2. Gradient clipping kullan
3. Mixed precision'ı kapat
4. Data'da NaN/Inf var mı kontrol et

### 4. Model Overfitting

**Belirtiler:**
- Train accuracy yüksek
- Val accuracy düşük
- Loss gap artıyor

**Çözümler:**
1. Dropout ekle/arttır
2. Weight decay kullan
3. Data augmentation
4. Daha fazla data
5. Regularization teknikleri

## Başarılı Konfigürasyonlar

### Configuration #1: Small Model Fast Training

```python
# Model: GPT-2 (124M)
# Use case: Prototyping, testing

config = {
    "model": "gpt2",
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "epochs": 3,
    "optimizer": "AdamW",
    "scheduler": "cosine_with_warmup",
    "warmup_ratio": 0.1,
}

# Sonuç: ✅ 2 saat, good accuracy
```

### Configuration #2: Medium Model Production

```python
# Model: GPT-Neo 1.3B
# Use case: Production deployment

config = {
    "model": "EleutherAI/gpt-neo-1.3B",
    "load_in_8bit": True,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 3e-5,
    "epochs": 5,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "scheduler": "cosine_with_warmup",
    "warmup_steps": 500,
    "gradient_checkpointing": True,
}

# Sonuç: ✅ 12 saat, excellent accuracy
```

### Configuration #3: LoRA Fine-tuning

```python
# Model: Any large model
# Use case: Parameter-efficient fine-tuning

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

config = {
    "model": "large-model",
    "lora": lora_config,
    "batch_size": 8,
    "learning_rate": 1e-4,  # LoRA için biraz daha yüksek OK
    "epochs": 5,
}

# Sonuç: ✅ Sadece %1 parameter train, %95 orijinal accuracy
```

## Dataset İpuçları

### Optimal Dataset Size

| Model Size | Min Samples | Optimal Samples |
|-----------|-------------|-----------------|
| Small (125M) | 1,000 | 10,000+ |
| Medium (1B) | 10,000 | 100,000+ |
| Large (3B+) | 50,000 | 500,000+ |

### Data Quality > Quantity

**Öğrenilen:**
- 10K high-quality > 100K noisy data
- Data cleaning çok önemli
- Balanced dataset şart
- Validation split unutma! (10-20%)

## Gelecek Denemeler

### Planlanıyor:

1. **Mixtral 8x7B** with extreme quantization
2. **GPT-4 distillation** küçük modellere
3. **Multi-task learning** approach
4. **Curriculum learning** strategies
5. **Better Türkçe tokenization**

### Yeni Teknikler:

- [ ] QLoRA + Flash Attention
- [ ] Parameter-Efficient Tuning methods
- [ ] Retrieval-Augmented Generation (RAG)
- [ ] Constitutional AI principles
- [ ] Multi-modal models (text + image)

## Kaynaklar ve Referanslar

### Yararlı Linkler:

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [PEFT Library](https://github.com/huggingface/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [EleutherAI](https://www.eleuther.ai/)

### Kullanılan Tools:

- PyTorch / TensorFlow
- HuggingFace Transformers
- Weights & Biases (tracking)
- TensorBoard
- DeepSpeed (distributed training)

## Katkıda Bulunanlar

Bu learnings dokümanı, 3,965 farklı eğitim denemesinin analiziyle oluşturulmuştur.

**Proje:** KuroshinPro AI Platform
**Tool:** Model Training Scanner v2.0
**Son Güncelleme:** 2025-01-15

---

💡 **Pro Tip**: Bu dokümanı düzenli güncelleyin! Her yeni deneme yeni bir öğrenme.

🎯 **Hedef**: 100+ başarılı eğitim, < %20 başarısızlık oranı

📊 **Metric**: Her ay gelişimi takip et!
