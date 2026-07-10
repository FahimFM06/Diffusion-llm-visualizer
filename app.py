import streamlit as st
import random
import time

# ==============================================================================
# PAGE CONFIG & CUSTOM THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Discrete Text Diffusion Visualizer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful, modern layout with responsive card formatting
# FIXED: Changed unsafe_allowing_html to unsafe_allow_html on line 61
st.markdown("""
<style>
    /* Main body changes */
    .reportview-container {
        background-color: #020617;
    }
    /* Token style classes */
    .token-box {
        display: inline-block;
        padding: 8px 14px;
        margin: 5px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .mask-token {
        background-color: #1e293b;
        color: #64748b;
        border: 1px dashed #475569;
    }
    .revealed-token {
        background-color: #0369a1;
        color: #f0f9ff;
        border: 1px solid #38bdf8;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
    }
    /* Step card container */
    .step-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    .step-header {
        color: #38bdf8;
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("🛠️ Generation Controls")
st.sidebar.markdown("Configure how the text diffusion pipeline unmasks the sequence.")

default_prompt = "Diffusion models shatter the sequential bottleneck of auto regressive language prediction"
text_input = st.sidebar.text_area("Target Sentence Prompt:", value=default_prompt, height=100)

total_steps = st.sidebar.slider("Total Diffusion Steps", min_value=3, max_value=12, value=6)
simulation_speed = st.sidebar.slider("Step Delay (seconds)", min_value=0.1, max_value=1.5, value=0.4, step=0.1)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Theoretical Concept:** Notice that tokens do not generate from left to right. "
    "The model globally updates the context and reveals multiple words in parallel based on prediction difficulty."
)

# ==============================================================================
# MAIN PAGE LAYOUT
# ==============================================================================
st.title("🌌 Text Diffusion Bloom Visualizer")
st.markdown("### Shattering the Sequential Bottleneck via Parallel Iterative Denoising")
st.write("This application visualizes how Language Diffusion works on discrete text using variable token masking.")

# Layout: Split into top metrics overview and bottom visual playground
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Generation Paradigm", value="Parallel / Bidirectional")
with col2:
    st.metric(label="Inference Bottleneck", value="Compute Bound (FLOPs)")
with col3:
    st.metric(label="Memory Structure Tax", value="0 KB (No KV Cache)")

st.markdown("---")

# Parse target text into tokens
target_tokens = text_input.split()
num_tokens = len(target_tokens)

if num_tokens == 0:
    st.warning("Please type or paste a valid text sentence into the sidebar control box.")
else:
    # Set up session state tracking to handle interactive click animations
    if 'current_step' not in st.session_state or st.session_state.get('prev_prompt') != text_input:
        st.session_state.current_step = 0
        st.session_state.prev_prompt = text_input
        st.session_state.generated_steps = []

    # UI Action Buttons
    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    with btn_col1:
        run_sim = st.button("🚀 Run Complete Animation", use_container_width=True)
    with btn_col2:
        reset_sim = st.button("🔄 Reset Environment", use_container_width=True)

    if reset_sim:
        st.session_state.current_step = 0
        st.session_state.generated_steps = []
        st.rerun()

    # Core Math Simulation Generator Function
    def generate_diffusion_timeline():
        timeline = []
        # Step 0: Starting fully masked
        current_seq = ["[MASK]"] * num_tokens
        timeline.append(list(current_seq))
        
        masked_indices = list(range(num_tokens))
        tokens_per_step = max(1, num_tokens // (total_steps - 1))
        
        # Seed for consistent local layout matching
        random.seed(42)
        shuffled_indices = random.sample(masked_indices, len(masked_indices))
        
        for step in range(1, total_steps):
            if step == total_steps - 1:
                # Final step unmasks everything
                reveal_count = len(shuffled_indices)
            else:
                reveal_count = min(tokens_per_step, len(shuffled_indices))
                
            for _ in range(reveal_count):
                if shuffled_indices:
                    idx = shuffled_indices.pop(0)
                    current_seq[idx] = target_tokens[idx]
            timeline.append(list(current_seq))
        return timeline

    # Calculate steps logic
    diffusion_timeline = generate_diffusion_timeline()

    # ==============================================================================
    # RENDER VISUAL TIMELINE
    # ==============================================================================
    if run_sim:
        placeholder = st.empty()
        for step_idx, step_seq in enumerate(diffusion_timeline):
            with placeholder.container():
                st.markdown(f"#### 🔄 Processing Step {step_idx} of {total_steps - 1}...")
                
                # Turn sequence arrays into customized styled HTML boxes
                html_str = "<div style='margin: 20px 0;'>"
                for token in step_seq:
                    if token == "[MASK]":
                        html_str += f"<span class='token-box mask-token'>{token}</span>"
                    else:
                        html_str += f"<span class='token-box revealed-token'>{token}</span>"
                html_str += "</div>"
                
                st.markdown(html_str, unsafe_allow_html=True)
                time.sleep(simulation_speed)
        st.success("✨ Sequence fully denoised in parallel!")

    # Static historical step cards print layout below the player
    st.markdown("### 📋 Step-by-Step Evolution Trace")
    
    for idx, step_seq in enumerate(diffusion_timeline):
        with st.container():
            st.markdown(f"""
            <div class="step-card">
                <div class="step-header">⏱️ Step {idx} {"(Pure Latent Noise Block)" if idx == 0 else "(Refining Sub-Sequences)" if idx < len(diffusion_timeline)-1 else "(Final Output Generation)"}</div>
            </div>
            """, unsafe_allow_html=True)
            
            html_str = "<div>"
            for token in step_seq:
                if token == "[MASK]":
                    html_str += f"<span class='token-box mask-token'>{token}</span>"
                else:
                    html_str += f"<span class='token-box revealed-token'>{token}</span>"
            html_str += "</div><br>"
            st.markdown(html_str, unsafe_allow_html=True)
