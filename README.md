# Ciclopes

**Ciclopes** is a *biology-informed* deep learning tool for inferring **cell cycle phase** from **single-cell resolution data**, including both **scRNA-seq** and **spatial transcriptomics** datasets.

---

## Overview

Ciclopes combines prior biological knowledge of cell cycle marker genes with a deep generative model that captures the **oscillatory dynamics** of gene expression during the cell cycle. It provides smooth, interpretable, and continuous representations of cell cycle progression, beyond discrete phase assignments.

---

## Key Features

- **Biology-informed initialization**  
  Uses curated S-phase and G2/M-phase marker genes to compute initial scores, ensuring biologically grounded starting points.

- **Hybrid deep learning–biophysical model**  
  The model integrates a neural network encoder with a mechanistic decoder:
  - **Encoder:** Maps gene expression profiles to a *1D circular latent variable* representing the cell cycle phase.  
  - **Decoder:** Reconstructs expression using a **Fourier series**, modeling oscillatory transcriptional behavior.

- **Supports multiple data types**  
  Works seamlessly with both **single-cell RNA-seq** and **spatial transcriptomics** data.

- **Interpretable circular latent space**  
  The inferred latent variable corresponds to the cell cycle angle, allowing visualization and interpretation of progression through G1, S, G2, and M phases.

---

## Model Architecture
    Input: gene expression
            │
    Encoder: neural network
            |
    Circular latent variable (θ)
            │
    Decoder: Fourier series model
            │
    Reconstructed gene expression


### Dependencies
Python ≥ 3.9  
PyTorch ≥ 2.0  
Scanpy   
Numpy, Pandas, Matplotlib  

