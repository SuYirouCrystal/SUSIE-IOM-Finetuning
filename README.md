# SUSIE IOM Fine-Tuning Pipeline

This repository provides a minimal end-to-end pipeline to fine-tune a seq2seq Transformer (e.g. T5 or BART) on the FDA **Investigations Operations Manual (IOM)**. It now includes **ontology mapping** via Owlready2 against your COPE ontology (`COPE_pharma.owl`).

---

## 🚀 Features

- **Text Preprocessing**  
  Clean & segment raw IOM into coherent text units.

- **Triple & QA Generation w/ Ontology**  
  Extract (subject, verb, object) triples via spaCy, map them to COPE classes (OWL), and auto-generate Q&A pairs.

- **Training Data Preparation**  
  Format Q&A into `source`/`target` JSON-lines for Hugging Face fine-tuning.

- **Model Fine-Tuning**  
  Use 'HuggingFace Transformers’ `Seq2SeqTrainer` to specialize T5/BART on regulatory QA.

- **Inference Script**  
  `infer.py` supports single-shot or interactive QA using your fine-tuned model.

---

## 🛠️ Setup & Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/SuYirouCrystal/susie_iom_finetune.git
   cd susie_iom_finetune
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # spaCy model
   python3 -m spacy download en_core_web_sm
   # (Optional) SciSpaCy  
   pip install scispacy  
   pip install \
   https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz
   ```

3. **Add your COPE ontology**
   Place your OWL file in:
   ```bash
   data/ontology/COPE_pharma.owl
   ```

---

## 📈 Pipeline Steps

1. **Preprocess the IOM**  
   ```bash
   python scripts/preprocess_iom.py \
    --input data/raw/IOM.docx \
    --output data/processed/iom_segments.txt
   ```
   Cleans headers/footers, removes page numbers, and splits into paragraphs.

2. **Generate Triples & Q&A**
   ```bash
   python scripts/generate_qa_data.py \
     --segments      data/processed/iom_segments.txt \
     --output_qas    data/processed/qa_pairs.json \
     --output_triples data/processed/triples.json \
     --ontology      data/ontology/COPE_pharma.owl
   ```
   Extracts SVO triples, maps subject/object to COPE classes, then builds question/answer pairs.

3. **Prepare Training Data**
   ```bash
   python scripts/prepare_train_data.py \
     --input  data/processed/qa_pairs.json \
     --output data/processed/train.jsonl
   ```
   Converts Q&A JSON into `train.jsonl` of `<source,target>` for seq2seq training.

4. **Fine-Tune the Model**
   ```bash
   python scripts/finetune.py \
     --config config/finetune_config.json
   ```
   Trains your chosen model (e.g. t5-base) on `data/processed/train.jsonl`, saving to `models/iom_model/`.

5. **Inference / Interactive QA**
   * Single-Shot:
     ```bash
     python scripts/infer.py \
      --model_dir models/iom_model \
      --question "question: What is Form FDA 482?"
     ```
   * Interactive Loop:
     ```bash
     python scripts/infer.py --model_dir models/iom_model
     ```
     Type your questions at the `question:` prompt and `exit` to quit.

---

## 📦 Model Directory

The `models/` folder holds your fine‑tuned weights tracked via **Git LFS**. When you clone the repo:
```bash
# Make sure Git LFS is installed
git lfs install
# Pull LFS files
git lfs pull
```
Load the model in code:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/iom_model")
model     = AutoModelForSeq2SeqLM.from_pretrained("models/iom_model")
```

---

## 🎛️ Configuration
Edit `config/finetune_config.json` to adjust:
```json
{
  "model_name": "t5-base",
  "num_train_epochs": 3,
  "batch_size": 4,
  "learning_rate": 5e-5
}
```
Swap in `"t5-large"` or `"facebook/bart-base"` and tweak hyperparameters to match your GPU/CPU.

---

## 📜 License
MIT License © 2025 SuYirouCrystal
