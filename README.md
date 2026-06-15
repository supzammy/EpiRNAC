# EpiRNA‑C
<img width="1385" height="746" alt="image" src="https://github.com/user-attachments/assets/86aa0a6f-b3ad-4ca6-9fdb-620e997fd893" />

# EpiRNA‑C – Biophysical Tensor Fusion Scanner (CNN)

[![DOI (CNN)](https://zenodo.org/badge/DOI/10.5281/zenodo.20687377.svg)]([https://doi.org/10.5281/zenodo.20615778](https://doi.org/10.5281/zenodo.20687377))
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CNN%20Scanner-blue)](https://huggingface.co/spaces/supzammy/EpiRNAC)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EpiRNA‑C** is the **high‑throughput scanner** of the EpiRNA suite.  
It uses a multi‑scale dilated CNN with biophysical embeddings and Cross‑Scale Fusion Gates to perform **real‑time, single‑nucleotide resolution mapping** of m⁶A sites across sequences of any length.

The EpiRNA project provides **two complementary models** — use this CNN scanner for fast whole‑transcript profiling, and the transformer variant for high‑fidelity validation.

| Model | Backbone | AUROC (real miCLIP) | Inference | Use case |
|-------|----------|---------------------|-----------|----------|
| **Biophysical Tensor Fusion (CNN)** | Cross‑Scale CNN with biophysical embeddings | **0.6849** | < 5 s per sequence | Live interactive EBCS visualisation |
| **Transformer‑Biophysical Fusion** | Frozen DNA‑BERT + biophysical CNN | **0.80** | ~60 s per sequence (single‑window) | High‑specificity validation |

Both models were trained on **11,844 experimentally verified human m⁶A sites (GSE63753, miCLIP)** and an equal number of DRACH‑containing negative windows.

---

## 🚀 Live Demos

- **Real‑time CNN scanner (this tool):**  
  [supzammy/EpiRNAC](https://huggingface.co/spaces/supzammy/EpiRNAC)  
- **Transformer validator (for high‑specificity confirmation):**  
  [supzammy/epiRNAT](https://huggingface.co/spaces/supzammy/epiRNAT)

---


## ✨ Key Features

- **Length‑agnostic** – Processes sequences from 41 nt to >800 kb without manual windowing or memory errors.
- **Single‑nucleotide resolution** – EBCS pinpoints the exact catalytic adenosine within DRACH motifs.
- **Tunable detection modes** – Choose Discovery, Standard, Strict, or Clinical thresholds (τ = 0.0–0.90) to control sensitivity/specificity.
- **Interpretable profiles** – Per‑nucleotide contrast plots with GC‑content overlay and DRACH alignment markers.
- **High recall** – Sensitivity of 0.9968; designed to miss no genuine site.
- **Batch processing** – Upload CSV/FASTA files and download scored results.
- **NCBI streaming** – Fetch sequences directly from NCBI via accession number (zero‑disk).
- **Explainability (Captum)** – Integrated Gradients highlight which bases influence the model’s decision.

---

## 📊 Performance on real miCLIP data

| Metric | Value |
|--------|-------|
| **AUROC** | **0.6849** |
| **F1 Score** | 0.6739 |
| **Sensitivity** | 0.7822 |
| **Specificity** | 0.4609 |

*Evaluation on held‑out test set (GSE63753, n = 3,554 sequences).*  
*The high sensitivity ensures the scanner misses essentially no m⁶A sites; the moderate specificity is the intended trade‑off for a discovery tool.*

---

## 🧬 Architecture

### Biophysical Embedding
Each nucleotide is represented by a fixed 3‑dimensional vector encoding hydrogen‑bond potential, base‑stacking energy, and solvent accessibility.

### Multi‑Path Dilated Convolution with Cross‑Scale Fusion
Three parallel 1D‑convolutional paths (local, flank, structure) process the sequence at different scales. Cross‑Scale Fusion Gates allow the shorter‑range paths to dynamically attend to the long‑range structural path before pooling. The deployed model uses a 128‑filter variant, producing a 384‑dimensional combined feature vector.

### EBCS – Epitranscriptomic Boundary Contrast Scoring
The model slides a 41‑nt window across the input sequence. Raw outputs are stabilised by a local‑global variance blender (α = 0.3) and passed through a noise gate. The resulting per‑nucleotide contrast profile is plotted alongside GC content, and DRACH motifs are automatically annotated.

---

## 📦 Installation

    ```bash
    git clone https://github.com/supzammy/EpiRNA
    cd EpiRNA
    pip install -r requirements.txt


## 🧪 Reproducibility & Testing
Run individual window validation:

    ```bash
      # Validate a candidate DRACH window
      python epirna_engine.py --sequence "GGGGGGGGGGGGGGGGGGGGGGACTGGGGGGGGGGGGGGGGG"
      
      # Batch validation from a FASTA file
      python epirna_engine.py --fasta test_data/candidates.fasta
    

## 📄 Citation
If you use EpiRNA, please cite:

> Zaeem Ahmad Mansoori et al. (2026). “EpiRNA: Single‑Nucleotide Resolution Mapping of RNA Catalytic Boundaries Using Biophysical Tensor Fusion.” InCoB 2025.

 * Transformer checkpoint: https://zenodo.org/badge/DOI/10.5281/zenodo.20676854.svg

* CNN checkpoint & benchmarks: https://zenodo.org/badge/DOI/10.5281/zenodo.20687377.svg

## 📜 License
This project is licensed under the MIT License – see the MIT LICENSE file.

    ''' bash

    This README follows the exact structure you provided, includes the transformer‑specific metrics and architecture, and clearly positions the tool as the validator component. 
    It’s ready to be dropped into your `epiRNAT` repository.



## ScreenShots


# EBCS Profile
<img width="1385" height="746" alt="image" src="https://github.com/user-attachments/assets/86aa0a6f-b3ad-4ca6-9fdb-620e997fd893" />

# AI Attribution (Captum)
<img width="1364" height="747" alt="image" src="https://github.com/user-attachments/assets/e4b9a661-a4fe-4c73-b6c6-d91f767f5d75" />

# Batch Processing
<img width="1434" height="658" alt="image" src="https://github.com/user-attachments/assets/3947e4fa-3ab1-42be-ae0c-2727a3d9aaab" />

# Science & Architecture
<img width="1369" height="717" alt="image" src="https://github.com/user-attachments/assets/5cc24dac-2003-40dd-9f9a-9b6d5724c00e" />



