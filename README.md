# 🌌 Discrete Text Diffusion Language Model Visualizer

An interactive Streamlit application designed to visually demonstrate the inner workings of **Discrete Masked Diffusion Models for Large Language Models (LLMs)**. This tool provides a deep conceptual and mathematical walkthrough contrasting the parallel iterative denoising process against standard Auto-Regressive next-token prediction models.

Inspired by Umar Jamil's *"Build a Diffusion Language Model"* series, this project aims to break down complex architectural theory into accessible, step-by-step visual calculations without requiring local GPU compute environments.

---

## 🚀 Live Demo
You can view the application deployed live via the Streamlit Community Cloud here:
👉 **https://diffusion-llm-visualizer-hnakffscs9qtt93us555tn.streamlit.app/**

---

## 🎨 Application Layout & Features

The visualizer is organized into five core tabs built for conceptual clarity:

### 1. ⏳ Forward Noising Process
* **Discrete Masking Mechanics:** Visualizes how continuous Gaussian noise is replaced with discrete `[MASK]` tokens for language data processing.
* **Noise Schedules:** Features an interactive timeline slider showing the scaling linear mask ratio:
    $$r(t) = \frac{t}{T}$$
* **Bernoulli Sampling:** Displays text token lists changing dynamically under a simulated Bernoulli trial condition per step.

### 2. 🏗️ Transformer Architecture
* **Bidirectional Attention Flow:** Emphasizes how the absence of a causal attention mask allows the model to draw context from both past and future positions simultaneously.
* **Embedding Conflux Pipeline:** Traces hidden state arrays as they merge Token, Positional, and absolute Timestep ($t$) embeddings before entering the core layers.

### 3. 📈 Vector Flow: Step-by-Step Calculation Trace
* **Microscopic Trace Example:** Tracks a sequence through a microscopic model architecture (Vocabulary size = 6, Sequence length = 5, Dimension size = 2).
* **Exact Values Matrix:** Offers expandable calculation logs containing exact numerical steps for:
    * Combined input vector states.
    * Logits dot products ($h_i \cdot E[v]$).
    * Softmax probability mapping outputs.
    * Mask-isolated Cross-Entropy Loss computation:
        $$L_i = -\log p(\text{Target}_i)$$

### 4. 🔄 Reverse Denoising Inference Sandbox
* **Parallel Generation Emulator:** Features an interactive execution player that starts with a sequence entirely filled with `[MASK]` tokens and sequentially unfolds multiple words in parallel based on confidence metrics.

### 5. 📊 Core Architectural Summary
* **Silicon Optimization Profile:** Details how moving from sequential logic to parallel matrix computation eliminates the heavy memory overhead of **KV Caching**, shifting hardware traps from memory bandwidth constraints to raw tensor execution (FLOPs bound).

---

## 📁 Repository Structure

```text
📁 Diffusion-llm-visualizer (Root)
 ├── 📁 .streamlit
 │    └── 📄 config.toml    # Enforces custom dark corporate color theme configurations
 ├── 📄 app.py             # Main interactive application source file containing layout definitions
 └── 📄 README.md          # Project documentation mapping
