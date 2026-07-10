import streamlit as st
import numpy as np
import random
import time

# ==============================================================================
# 1. PAGE SETUP & THEME CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="LLM Diffusion Model Deep-Dive",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom injection for a clean corporate dark theme with visual structural accents
st.markdown("""
<style>
    /* Styling variables and foundational blocks */
    .main-title { font-size: 42px; font-weight: 800; color: #f8fafc; margin-bottom: 5px; }
    .subtitle { font-size: 18px; color: #94a3b8; margin-bottom: 30px; }
    
    /* Elegant visual cards */
    .feature-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .card-header { font-size: 20px; font-weight: 700; color: #38bdf8; margin-bottom: 12px; }
    .card-desc { font-size: 15px; color: #cbd5e1; line-height: 1.6; }
    
    /* Token tracking components */
    .token-grid { display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; }
    .token-unit {
        padding: 10px 18px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-weight: 700;
        font-size: 16px;
        text-align: center;
        min-width: 70px;
    }
    .token-clean { background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; }
    .token-mask { background-color: #000000; color: #64748b; border: 2px dashed #ef4444; box-shadow: 0 0 8px rgba(239, 68, 68, 0.2); }
    .token-predicted { background-color: #047857; color: #ecfdf5; border: 1px solid #10b981; }
    
    /* Pipeline item lists */
    .pipeline-step {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
        color: #e2e8f0;
    }
    
    /* Mathematical calculation display styling */
    .math-block {
        background-color: #020617;
        border: 1px solid #334155;
        font-family: 'Consolas', monospace;
        padding: 15px;
        border-radius: 8px;
        color: #38bdf8;
        font-size: 14px;
        line-height: 1.5;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# App branding headers
st.markdown("<div class='main-title'>🧬 Discrete Diffusion Language Models</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Theoretical Walkthrough, Vector Architectures, and Exact Computation Mechanics</div>", unsafe_allow_html=True)
st.markdown("---")

# ==============================================================================
# 2. TABBED APPLICATION NAVIGATION
# ==============================================================================
tabs = st.tabs([
    "⏳ 1. Forward Noising Process", 
    "🏗️ 2. Transformer Architecture", 
    "📈 3. Vector Flow: Step-by-Step Calculation", 
    "🔄 4. Reverse Denoising Inference",
    "📊 5. Core Architectural Summary"
])

# ==============================================================================
# TAB 1: THE FORWARD NOISING PROCESS
# ==============================================================================
with tabs[0]:
    st.header("The Forward Noising Process")
    st.markdown("""
    Unlike continuous diffusion models that add Gaussian noise to images, **Discrete Text Diffusion** applies noise 
    by substituting clean text tokens with special `[MASK]` values based on a predefined schedule.
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Noise Schedule Mechanics")
        st.write("The forward process isolates sequences over steps $t$ up to horizon $T$:")
        st.latex(r"r(t) = \frac{t}{T}")
        st.write("At each token index, a Bernoulli trial independently determines if the word becomes masked:")
        st.latex(r"mask_i \sim \text{Bernoulli}(r(t))")
        
        # Simulation Controls
        st.markdown("---")
        t_step = st.slider("Select Timestep (t)", min_value=1, max_value=6, value=1)
        st.metric(label="Current Masking Ratio r(t)", value=f"{int((t_step/6)*100)}%")
        
    with col2:
        st.markdown("### Interactive Masking Emulator")
        sample_sentence = ["The", "cat", "sat", "on", "the", "mat"]
        
        # Deterministic seeding mapping to timestep slider
        random.seed(t_step + 42)
        mask_prob = t_step / 6.0
        
        st.write("**Visualized Text Sequence State:**")
        html_grid = "<div class='token-grid'>"
        for word in sample_sentence:
            if random.random() < mask_prob:
                html_grid += f"<div class='token-unit token-mask'>[MASK]</div>"
            else:
                html_grid += f"<div class='token-unit token-clean'>{word}</div>"
        html_grid += "</div>"
        st.markdown(html_grid, unsafe_allow_html=True)
        
        st.info(f"At timestep $t={t_step}$, roughly {int(mask_prob*100)}% of the sentence tokens are hidden.")

# ==============================================================================
# TAB 2: MODEL ARCHITECTURE
# ==============================================================================
with tabs[1]:
    st.header("DiffusionTransformerLM Architecture")
    st.markdown("Data routing tracking noisy masked arrays entering bidirectional transformer modules:")
    
    arch_col1, arch_col2 = st.columns([2, 3])
    with arch_col1:
        st.markdown("""
        ### Core Processing Stages
        
        1. **Embedding Conflux:**
           * **Token Embeddings:** Maps vocabulary to latent state matrices.
           * **Positional Embeddings:** injects index sequence values.
           * **Time Embeddings:** Encodes the absolute diffusion step $t$.
        
        2. **Bidirectional Transformer Block:**
           * Employs full context windows (past and future tokens simultaneously).
           * Avoids casual restrictions found in standard sequence auto-regressive networks.
           * Evaluates `LayerNorm -> Multi-Head Attention -> Residual -> LayerNorm -> FeedForward -> Residual`.
        
        3. **LM Head Prediction & Optimization:**
           * Projects states back to vocabulary distribution logits.
           * **Cross-Entropy Loss Calculation** runs *exclusively* across target positions originally hidden behind `[MASK]` tokens.
        """)
        
    with arch_col2:
        st.markdown("### Active Flow Layer Blocks")
        
        # Display block visualization mimicking the visual flows
        layers = [
            ("Input Processing", "Noisy Token String + Position Vector + Timestep Embedding (t)"),
            ("Layer Normalization #1", "Standardizes embedding values before context routing"),
            ("Multi-Head Attention", "Bidirectional context calculations over entire string length"),
            ("Residual Addition #1", "Adds raw input states to attention output vectors"),
            ("Layer Normalization #2", "Standardizes hidden activations before non-linear projection"),
            ("Feed-Forward Network", "Dense multilayer projections via GELU activations"),
            ("Residual Addition #2", "Combines pre-FFN vector arrays with finalized output hidden states"),
            ("Language Model Head", "Projects hidden states back to vocabulary dimension logits matrices"),
            ("Cross-Entropy Evaluation", "Isolates and computes loss matrices exclusively for masked positions")
        ]
        
        for idx, (title, desc) in enumerate(layers, 1):
            st.markdown(f"""
            <div class='pipeline-step'>
                <strong>Step {idx}: {title}</strong><br>
                <small style='color: #a7f3d0;'>{desc}</small>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# TAB 3: VECTOR FLOW: MICROSCOPIC COMPUTATION TRACE
# ==============================================================================
with tabs[2]:
    st.header("Vector Flow: Structural Calculation Trace")
    st.markdown("A complete mathematical execution trace through a microscopic model architecture:")
    
    # Model Configuration Values
    v_col1, v_col2, v_col3, v_col4 = st.columns(4)
    v_col1.metric("Vocab Size (V)", "6")
    v_col2.metric("Sequence Length (L)", "5")
    v_col3.metric("Embedding Dim (D)", "2")
    v_col4.metric("Total Diffusion Horizon (T)", "4")
    
    st.markdown("### 🛠️ Execution Pipeline Status")
    
    with st.expander("Step 1: Embedding Initialization Matrices & Sequence Input", expanded=True):
        st.write("Let our sequence input string be: `X = [BOS, A, B, C, EOS]` with token IDs: `[0, 3, 4, 5, 1]`.")
        st.write("Assume position $idx=1$ (`A`) and position $idx=3$ (`C`) are hidden behind `[MASK]` tokens:")
        st.markdown("<div class='token-grid'><div class='token-unit token-clean'>[BOS]</div><div class='token-unit token-mask'>[MASK]</div><div class='token-unit token-clean'>B</div><div class='token-unit token-mask'>[MASK]</div><div class='token-unit token-clean'>[EOS]</div></div>", unsafe_allow_html=True)
        
        st.write("**Embedding Matrix Lookup values:**")
        st.markdown("""<div class='math-block'>
Token Embeddings E (6x2):
  ID 0 [BOS]  -> [ 0.05,  0.05]
  ID 1 [EOS]  -> [-0.05, -0.05]
  ID 2 [MASK] -> [ 0.00,  0.00]
  ID 3 A      -> [ 0.80,  0.20] (Hidden behind mask)
  ID 4 B      -> [ 0.20,  0.80]
  ID 5 C      -> [-0.80,  0.20] (Hidden behind mask)

Positional Vectors P[0..4]:
  P0=[0.0, 0.0], P1=[0.1, -0.1], P2=[0.0, 0.2], P3=[-0.1, 0.0], P4=[0.0, -0.2]

Time Embedding Vector (For t = 1):
  T_1 = [0.1, 0.1]
</div>""", unsafe_allow_html=True)

    with st.expander("Step 2: Input Conflux Ingestion", expanded=False):
        st.write("For each sequence position $i$, the combined input hidden vector $h_i^{(0)}$ equals:")
        st.latex(r"h_i^{(0)} = \text{TokenEmbed}(X_i) + P_i + T_t")
        st.markdown("""<div class='math-block'>
Calculated Combined Input State Matrix h^(0):
  Position 0 [BOS] : [0.05, 0.05] + [0.0,  0.0]  + [0.1, 0.1] = [ 0.15,  0.15]
  Position 1 [MASK]: [0.00, 0.00] + [0.1, -0.1]  + [0.1, 0.1] = [ 0.20,  0.00]
  Position 2 B     : [0.20, 0.80] + [0.0,  0.2]  + [0.1, 0.1] = [ 0.30,  1.10]
  Position 3 [MASK]: [0.00, 0.00] + [-0.1, 0.0]  + [0.1, 0.1] = [ 0.00,  0.10]
  Position 4 [EOS] : [-0.05,-0.05] + [0.0, -0.2]  + [0.1, 0.1] = [ 0.05, -0.15]
</div>""", unsafe_allow_html=True)

    with st.expander("Step 3: Self-Attention Routing & FFN Layer Updates", expanded=False):
        st.write("Vectors travel through global attention matrices. Because there is no causal mask, information flows bidirectionally across all positions:")
        st.markdown("""<div class='math-block'>
After LayerNorm #1 -> Bidirectional Attention Processing -> Residual Connections:
  Middle Activation Hidden Matrix States = [...]

After LayerNorm #2 -> Feed-Forward Network Blocks -> Final Layer Normalization:
  Finalized Hidden Vector Matrix Outputs h^(final):
    h_0 (BOS)  = [ 0.12,  0.05]
    h_1 (MASK) = [ 1.00, -1.00]  <-- Isolated vector to predict token 'A'
    h_2 (B)    = [ 0.18,  0.92]
    h_3 (MASK) = [-0.60,  0.60]  <-- Isolated vector to predict token 'C'
    h_4 (EOS)  = [ 0.02, -0.11]
</div>""", unsafe_allow_html=True)

    with st.expander("Step 4: Language Model Head Outputs & Predictions", expanded=False):
        st.write("The system calculates the dot product between the hidden layer results $h_i^{(\text{final})}$ and original candidate token vectors $E[v]$ to yield raw logits values:")
        st.latex(r"\text{logits}_{i,v} = h_i^{(\text{final})} \cdot E[v]")
        
        st.markdown("""<div class='math-block'>
Evaluating Masked Position 1 (Target Token: 'A'):
  Logits matching candidates:
    v=0 [BOS]  -> [1, -1] . [ 0.05,  0.05] = 0.0
    v=1 [EOS]  -> [1, -1] . [-0.05, -0.05] = 0.0
    v=2 [MASK] -> [1, -1] . [ 0.00,  0.00] = 0.0
    v=3 [A]    -> [1, -1] . [ 0.80,  0.20] = 0.6
    v=4 [B]    -> [1, -1] . [ 0.20,  0.80] = -0.6
    v=5 [C]    -> [1, -1] . [-0.80,  0.20] = -1.0

  Resulting Logits Array Vector = [0.0, 0.0, 0.0, 0.6, -0.6, -1.0]
  Softmax Probability Map Calculation:
    p = softmax([0.0, 0.0, 0.0, 0.6, -0.6, -1.0]) = [0.174, 0.174, 0.174, 0.318, 0.096, 0.064]
  
  Argmax Prediction Selection: Index 3 ('A') with confidence probability 31.75% ✔️
</div>""", unsafe_allow_html=True)

    with st.expander("Step 5: Mask-Isolated Cross-Entropy Loss Computation", expanded=True):
        st.write("Loss updates bypass known tokens and calculate costs exclusively across elements that were masked during setup:")
        st.latex(r"L_i = -\log p(\text{Target}_i)")
        
        st.markdown("""<div class='math-block'>
Loss evaluation scores over masked elements:
  Position 1 (Target: A): p[A] = 0.3175 -> L_1 = -log(0.3175) = 1.1473
  Position 3 (Target: C): p[C] = 0.3360 -> L_3 = -log(0.3360) = 1.0905

Total Batch Sequence Loss:
  L = (L_1 + L_3) / 2 = (1.1473 + 1.0905) / 2 = 1.1189
</div>""", unsafe_allow_html=True)

# ==============================================================================
# TAB 4: REVERSE DENOISING INFERENCE
# ==============================================================================
with tabs[3]:
    st.header("The Iterative Reverse Denoising Pipeline")
    st.markdown("""
    During inference, generation operates in reverse. The system starts with an input array entirely filled with 
    `[MASK]` values, iteratively running model evaluations to reveal tokens based on probability confidence.
    """)
    
    inf_col1, inf_col2 = st.columns([1, 2])
    with inf_col1:
        st.markdown("### Generation Loop Sequence")
        st.markdown("""
        1. **Fully Masked State:** Initialize string arrays to $100\%$ masking.
        2. **Logit Evaluation Evaluation:** Run the full bidirectional transformer pass to gather predictive vocabulary distributions.
        3. **Confidence Scoring:** Evaluate the token probabilities across all remaining masked positions.
        4. **Selective Reveal:** Select and unlock the highest confidence tokens, then feed the updated string back into the loop.
        """)
    
    with inf_col2:
        st.markdown("### Inference Pipeline Execution Sandbox")
        target_string = "Diffusion models handle text generation tasks in parallel steps"
        tokens_list = target_string.split()
        total_len = len(tokens_list)
        
        if st.button("🚀 Execute Reverse Denoising Loop"):
            status_box = st.empty()
            display_box = st.empty()
            
            # Step 0 setup
            current_state = ["[MASK]"] * total_len
            unlocked_indices = set()
            
            # Simulated 4-step parallel denoising loop
            for loop_idx in range(1, 5):
                status_box.markdown(f"**Loop Step {loop_idx}/4: Model computing predictions...**")
                
                # Determine how many tokens to unmask in this step
                chunk_count = max(1, total_len // 3) if loop_idx < 4 else total_len - len(unlocked_indices)
                
                # Filter indices that are still masked
                available_indices = [i for i in range(total_len) if i not in unlocked_indices]
                
                # Select random sample to mimic confidence selection choices
                selected_sample = random.sample(available_indices, min(chunk_count, len(available_indices)))
                
                for idx in selected_sample:
                    current_state[idx] = tokens_list[idx]
                    unlocked_indices.add(idx)
                    
                # Format current array string tokens using custom HTML styles
                grid_html = "<div class='token-grid'>"
                for tok in current_state:
                    style_class = "token-mask" if tok == "[MASK]" else "token-predicted"
                    grid_html += f"<div class='token-unit {style_class}'>{tok}</div>"
                grid_html += "</div>"
                
                display_box.markdown(grid_html, unsafe_allow_html=True)
                time.sleep(0.8)
                
            status_box.success("✨ Target sequence fully reconstructed in parallel!")

# ==============================================================================
# TAB 5: CORE ARCHITECTURAL SUMMARY
# ==============================================================================
with tabs[4]:
    st.header("Key Structural Comparison Matrix")
    st.markdown("Reviewing core operational properties between auto-regressive networks and discrete diffusion models:")
    
    # Render three high-impact summary cards side-by-side
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.markdown("""
        <div class='feature-card'>
            <div class='card-header'>⚡ Parallel Generation</div>
            <div class='card-desc'>
                Bypasses standard left-to-right loops. The model treats all sequence positions as 
                active, processing global tokens simultaneously over fixed inference step iterations.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with summary_col2:
        st.markdown("""
        <div class='feature-card'>
            <div class='card-header'>↔️ Bidirectional Context</div>
            <div class='card-desc'>
                Removes strict causal mask matrices. During evaluations, hidden layer representations can 
                draw structural cues from both preceding and succeeding items across the sequence context.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with summary_col3:
        st.markdown("""
        <div class='feature-card'>
            <div class='card-header'>⚙️ Silicon Optimization</div>
            <div class='card-desc'>
                Eliminates the heavy memory overhead of KV caching. This shifts infrastructure 
                constraints away from memory bandwidth limitations toward raw tensor matrix core calculations.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Core Performance Feature Grid Table
    st.markdown("### Comparative Performance Profile")
    st.table([
        {"Feature Operational Metric": "Context Evaluation Layer", "Auto-Regressive Models (ARM)": "Causal Masking (Past context views only)", "Discrete Text Diffusion Models": "Bidirectional Context (Full sequence visibility)"},
        {"Feature Operational Metric": "Generation Pattern Type", "Auto-Regressive Models (ARM)": "Sequential token-by-token loop strings", "Discrete Text Diffusion Models": "Parallel iterative denoising adjustments"},
        {"Feature Operational Metric": "Inference Memory Scaling Tax", "Auto-Regressive Models (ARM)": "High overhead demands (KV Cache storage)", "Discrete Text Diffusion Models": "Minimal (Transient hidden states calculations)"},
        {"Feature Operational Metric": "Hardware Core Bottleneck", "Auto-Regressive Models (ARM)": "Memory Bandwidth Bounds", "Discrete Text Diffusion Models": "Processor Compute Bounds (FLOPs Optimization)"}
    ])
