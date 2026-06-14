#بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ
#Bismillāhi ar‑Raḥmāni ar‑Raḥīm.
#"In the name of Allah, the Most Merciful, the Most Compassionate."
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import pandas as pd
import tempfile
from captum.attr import LayerIntegratedGradients


# ---------- NCBI streaming (zero‑disk) ----------
from Bio import Entrez
Entrez.email = "zammy@gmail.com.com"      # NCBI requires an email

def fetch_ncbi(accession):
    """Return the FASTA sequence for a given NCBI accession, or an error message."""
    if not accession or not accession.strip():
        return "", gr.update(visible=True, value="⚠️ Please enter an NCBI accession number.")
    try:
        handle = Entrez.efetch(db="nucleotide", id=accession.strip(),
                               rettype="fasta", retmode="text")
        record = handle.read()
        handle.close()
        lines = record.splitlines()
        seq = "".join(line.strip() for line in lines if not line.startswith(">"))
        if not seq:
            return "", gr.update(visible=True, value="❌ No sequence found for that accession.")
        return seq.upper(), gr.update(visible=True, value=f"✅ Loaded {len(seq)} bp from NCBI.")
    except Exception as e:
        return "", gr.update(visible=True, value=f"❌ NCBI fetch error: {e}")

        
# ==========================================
# 1. BIOPHYSICAL TENSOR FUSION MODEL
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CrossScaleFusionGate(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.query_conv = nn.Conv1d(channels, channels//4, 1)
        self.key_conv   = nn.Conv1d(channels, channels//4, 1)
        self.value_conv = nn.Conv1d(channels, channels, 1)
        self.gamma      = nn.Parameter(torch.zeros(1))

    def forward(self, source, target):
        B, C, L = source.shape
        Q = self.query_conv(source).view(B, -1, L).permute(0,2,1)
        K = self.key_conv(target).view(B, -1, L)
        attn = F.softmax(torch.bmm(Q, K), dim=-1)
        V = self.value_conv(target).view(B, -1, L)
        out = torch.bmm(V, attn.permute(0,2,1)).view(B, C, L)
        return source + self.gamma * out

class BiophysicalTensorFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        biophysical_matrix = torch.tensor([
            [0.0,  0.0,  0.0], [1.0, -1.0,  0.5], [-1.0, -1.0, -0.5],
            [-1.0,  1.0,  2.5], [1.0,  1.0, -1.0]
        ])
        self.embedding = nn.Embedding.from_pretrained(biophysical_matrix, freeze=False)
        self.local_path  = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.flank_path  = nn.Conv1d(3, 32, kernel_size=5, padding=4, dilation=2)
        self.struct_path = nn.Conv1d(3, 32, kernel_size=5, padding=8, dilation=4)
        self.fuse_local = CrossScaleFusionGate(32)
        self.fuse_flank = CrossScaleFusionGate(32)
        self.layer_norm  = nn.LayerNorm(96)
        self.fc_contrast = nn.Linear(96, 1)

    def forward(self, x):
        x_emb = self.embedding(x).transpose(1, 2)
        c1 = self.local_path(x_emb)
        c2 = self.flank_path(x_emb)
        c3 = self.struct_path(x_emb)
        c1 = self.fuse_local(c1, c3)
        c2 = self.fuse_flank(c2, c3)
        p1 = F.max_pool1d(F.pad(c1, (0, 1)), kernel_size=2, stride=1)
        p2 = F.max_pool1d(F.pad(c2, (0, 1)), kernel_size=2, stride=1)
        p3 = F.max_pool1d(F.pad(c3, (0, 1)), kernel_size=2, stride=1)
        combined = torch.cat([p1, p2, p3], dim=1).transpose(1, 2)
        return self.fc_contrast(F.relu(self.layer_norm(combined))).squeeze(-1)

# Instantiate CNN model
model = BiophysicalTensorFusionModel().to(device).eval()
if os.path.exists("EpiRNA_Biophysical_Master.pt"):
    try:
        state_dict = torch.load("EpiRNA_Biophysical_Master.pt", map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print("✅ CNN model loaded.")
    except Exception as e:
        print(f"⚠️ Could not load CNN checkpoint: {e}")

# ==========================================
# 2. ADAPTIVE PROCESSING & STABILIZATION
# ==========================================
def compute_advanced_calibrated_profile(raw_deltas):
    global_std = torch.std(raw_deltas) + 1e-4
    raw_deltas = torch.clamp(raw_deltas, min=-2.0, max=2.0)
    calibrated = torch.zeros_like(raw_deltas)
    for i in range(len(raw_deltas)):
        start = max(0, i - 6)
        end = min(len(raw_deltas), i + 7)
        local_ctx = raw_deltas[start:end]
        blended_std = (torch.std(local_ctx) * 0.3) + (global_std * 0.7) + 1e-4
        z_score = (raw_deltas[i] - torch.mean(local_ctx)) / blended_std
        calibrated[i] = torch.clamp((torch.sigmoid(z_score) - 0.5) * 2.0, min=0.0)
    return calibrated.cpu().numpy()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def calc_gc_content(sequence, window=15):
    gc_vals = []
    half = window // 2
    for i in range(len(sequence)):
        sub = sequence[max(0, i - half) : min(len(sequence), i + half + 1)]
        gc_vals.append((sub.count('G') + sub.count('C')) / len(sub))
    return gc_vals

def find_drach_motifs(sequence):
    pattern = r'[AGU][AG]AC[ACU]'
    matches = list(re.finditer(pattern, sequence))
    highlighted_seq = sequence
    for m in reversed(matches):
        start, motif = m.start(), m.group()
        highlighted_seq = (
            highlighted_seq[:start] +
            f"**<span style='color:#000000; background:#f3f4f6; padding:2px 4px; border-radius:4px; border:1px solid #d1d5db;'>{motif}</span>**" +
            highlighted_seq[start+5:]
        )
    motifs_text = ", ".join(
        [f"<span style='color:#111827;'>{m.group()} (Pos {m.start()})</span>" for m in matches]
    ) if matches else "<span style='color:#111827;'>None detected.</span>"
    return motifs_text, highlighted_seq

# ==========================================
# 4. ENHANCED PREDICT
# ==========================================
def predict(raw_seq, threshold=0.45):
    raw_seq = raw_seq.upper().strip().replace('T', 'U')
    illegal = set(raw_seq) - {'A', 'U', 'C', 'G'}
    if illegal:
        return None, f"<h3>❌ Invalid character(s) found: {', '.join(sorted(illegal))}</h3>", ""
    seq = raw_seq
    if len(seq) < 41:
        return None, "<h3>❌ Sequence too short (min 41bp).</h3>", ""

    seq_len = len(seq)
    global_raw_deltas = np.zeros(seq_len)
    counts = np.zeros(seq_len)

    mapping = {'A': 1, 'U': 2, 'C': 3, 'G': 4}
    for start in range(0, seq_len - 41 + 1):
        chunk = seq[start:start+41]
        tokens = torch.tensor([[mapping[b] for b in chunk]], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(tokens).squeeze(0).cpu().numpy()
        global_raw_deltas[start:start+41] += output
        counts[start:start+41] += 1.0

    averaged_deltas = torch.tensor(global_raw_deltas / np.maximum(counts, 1.0), dtype=torch.float32)
    scores = compute_advanced_calibrated_profile(averaged_deltas)

    # --- Multi‑target peak detection (DRACH filtered by threshold) ---
    raw_peak = int(np.argmax(scores))
    matches = list(re.finditer(r'[AGU][AG]AC[ACU]', seq))
    if matches:
        all_candidates = []
        for m in matches:
            pos = m.start() + 2
            if pos >= seq_len:
                continue
            local_conf = float(np.max(scores[max(0, pos-3):min(len(scores), pos+4)]))
            all_candidates.append((pos, local_conf))

        aligned_peaks = [pos for pos, conf in all_candidates if conf >= threshold]
        if aligned_peaks:
            peak_source = f"{len(aligned_peaks)}/{len(all_candidates)} DRACH sites above τ={threshold:.2f}"
        else:
            peak_source = f"No DRACH site above τ={threshold:.2f} (use lower threshold)"
    else:
        aligned_peaks = []
        peak_source = "No DRACH motif in sequence"
    aligned_peaks = [min(p, seq_len - 1) for p in aligned_peaks]
    peak_chars = [seq[p] if p < seq_len else '?' for p in aligned_peaks]

    # --- Noise gate (dynamic, based on slider) ---
    clean_scores = scores.copy()
    clean_scores[clean_scores < threshold] = 0.0

    # --- Build plot ---
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.plot(range(seq_len), clean_scores, color='#4f46e5', linewidth=2.0, marker='o', markersize=3)
    ax.fill_between(range(seq_len), clean_scores, color='#4f46e5', alpha=0.08)

    for i, target_pos in enumerate(aligned_peaks):
        ax.axvline(x=target_pos, color='red', linestyle='--', linewidth=2, alpha=0.8,
                   label='Aligned Target' if i == 0 else "")

    # --- Dynamic x‑axis ticks ---
    if seq_len > 60:
        tick_step = 100 if seq_len > 1000 else 50
        tick_positions = list(range(0, seq_len, tick_step))
        if (seq_len - 1) not in tick_positions:
            tick_positions.append(seq_len - 1)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(p) for p in tick_positions], fontsize=10)
    else:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(list(seq), fontsize=8, rotation=45 if seq_len <= 50 else 90)

    ax.set_xlabel("Spatial Nucleotide Resolution")
    ax.set_ylabel("Boundary Contrast Delta", color='#4f46e5')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_title(f"Genome‑Scale EBCS Profile – τ={threshold:.2f} | {peak_source}", fontweight='bold')
    if aligned_peaks:
        ax.legend(loc='upper right')

    gc_vals = calc_gc_content(seq)
    ax2 = ax.twinx()
    ax2.plot(range(seq_len), gc_vals, color='#9ca3af', linestyle='-', linewidth=2, alpha=0.4)
    ax2.set_ylabel("GC Content", color='#9ca3af')
    ax2.tick_params(axis='y', labelcolor='#9ca3af')
    plt.tight_layout()

    # --- HTML output ---
    target_html_list = []
    for c, p in zip(peak_chars, aligned_peaks):
        conf_score = float(np.max(scores[max(0, p-3):min(len(scores), p+4)]))
        html_segment = f"<span>{c}</span> (Pos <b>{p}</b>, Conf: <b>{conf_score:.4f}</b>)"
        target_html_list.append(html_segment)

    target_html = " &nbsp;|&nbsp; ".join(target_html_list)
    motifs_text, highlighted_seq = find_drach_motifs(seq)
    res = f"""
    <div style="color: #111827; font-size: 1.05rem;">
        <h3>🎯 Targets: {target_html}</h3>
        <p><b>Architecture:</b> Biophysical Tensor Fusion (variable‑length)</p>
        <p><b>Max Contrast:</b> {scores[raw_peak]:.4f}</p>
        <p><b>Sequence Map:</b> {highlighted_seq}</p>
    </div>
    """
    mot = f"<div><p><b>Canonical DRACH Motifs:</b> {motifs_text}</p></div>"
    return fig, res, mot

# ==========================================
# 5. BATCH PROCESSING
# ==========================================
def process_batch(file_obj, k_mask=6):
    if file_obj is None:
        return None, "<h3>❌ Upload a CSV or FASTA file.</h3>"
    sequences = []
    with open(file_obj.name) as f:
        for line in f:
            line = line.strip().upper()
            if line.startswith(">"):
                continue
            if len(line) >= 41:
                sequences.append(line)

    results = []
    for seq in sequences:
        if not re.fullmatch(r'[ACGTUN]+', seq):
            continue
        seq = seq.replace('T', 'U')
        seq_len = len(seq)
        global_raw_deltas = np.zeros(seq_len)
        counts = np.zeros(seq_len)
        mapping = {'A': 1, 'U': 2, 'C': 3, 'G': 4}
        for start in range(0, seq_len - 41 + 1):
            chunk = seq[start:start+41]
            tokens = torch.tensor([[mapping.get(b, 0) for b in chunk]], dtype=torch.long).to(device)
            with torch.no_grad():
                output = model(tokens).squeeze(0).cpu().numpy()
            global_raw_deltas[start:start+41] += output
            counts[start:start+41] += 1.0
        averaged_deltas = torch.tensor(global_raw_deltas / np.maximum(counts, 1.0), dtype=torch.float32)
        scores = compute_advanced_calibrated_profile(averaged_deltas)

        peak_idx = int(np.argmax(scores))
        motifs_text, _ = find_drach_motifs(seq)
        results.append({
            "Sequence": seq,
            "Peak_Position": peak_idx,
            "Peak_Base": seq[peak_idx] if peak_idx < len(seq) else '',
            "Max_EBCS_Score": round(scores[peak_idx], 4),
            "Length": len(seq),
            "DRACH_Motifs": re.sub(r'<.*?>', '', motifs_text)
        })

    if not results:
        return None, "<h3>❌ No valid sequences found.</h3>"

    df = pd.DataFrame(results)
    out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, "EpiRNA_Batch_Results.csv")
    df.to_csv(out_path, index=False)
    return out_path, f"<h3>✅ Processed {len(results)} sequences.</h3>"

# ==========================================
# 6. CAPTUM EXPLAINER
# ==========================================
def run_explainer(raw_seq):
    try:
        seq = raw_seq.upper().strip().replace('T', 'U')
        seq = re.sub(r'[^AUCG]', '', seq)
        if len(seq) < 41:
            seq = seq.ljust(41, 'G')
        window = seq[:41]
        mapping = {'A': 1, 'U': 2, 'C': 3, 'G': 4}
        tokens = torch.tensor([[mapping[base] for base in window]], dtype=torch.long).to(device)

        model.eval()
        lig = LayerIntegratedGradients(model, model.embedding)
        attributions = lig.attribute(tokens, target=0, n_steps=50, return_convergence_delta=False)

        attr_per_base = attributions.sum(dim=2).squeeze(0).detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#4f46e5' if v >= 0 else '#e11d48' for v in attr_per_base]
        ax.bar(range(41), attr_per_base, color=colors)
        ax.set_xticks(range(41))
        ax.set_xticklabels(list(window), fontsize=8, rotation=45)
        ax.set_title("Nucleotide‑level importance (Integrated Gradients)", fontweight='bold')
        ax.set_ylabel("Attribution score")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()

        res_html = """
        <p style='color:#111827;'><b>Explanation window:</b> first 41 bases of your input.</p>
        <p style='color:#111827;'>Positive bars (indigo) = increase catalytic boundary signal.<br>
        Negative bars (red) = decrease it.</p>
        """
        return fig, res_html
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'Captum error:\n{str(e)}', ha='center', va='center')
        ax.axis('off')
        return fig, f"<p style='color:#e11d48;'>Explainability failed: {e}</p>"

# ==========================================
# 7. GLASSMORPHISM FRONTEND THEME
# ==========================================
glass_theme = gr.themes.Soft(
    primary_hue="indigo", neutral_hue="slate"
).set(
    body_background_fill="#f8fafc", body_background_fill_dark="#f8fafc",
    background_fill_primary="rgba(255, 255, 255, 0.85)", background_fill_primary_dark="rgba(255, 255, 255, 0.85)",
    background_fill_secondary="rgba(255, 255, 255, 0.6)", background_fill_secondary_dark="rgba(255, 255, 255, 0.6)",
    border_color_primary="rgba(203, 213, 225, 0.6)", border_color_primary_dark="rgba(203, 213, 225, 0.6)",
    block_background_fill="rgba(255, 255, 255, 0.7)", block_background_fill_dark="rgba(255, 255, 255, 0.7)",
    block_title_text_color="#111827", block_title_text_color_dark="#111827",
    block_label_text_color="#374151", block_label_text_color_dark="#374151",
    body_text_color="#1f2937", body_text_color_dark="#1f2937",
    input_background_fill="#ffffff", input_background_fill_dark="#ffffff",
)

custom_css = """
/* ========== GLOBAL LAYOUT ========== */
:root {
    --font-mono: 'DM Serif Display', 'JetBrains Mono', 'Courier New', Courier, monospace;
}
.gradio-container {
    background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%) !important;
    font-family: var(--font-mono) !important;
    letter-spacing: -0.02em !important;
    min-height: 100vh !important;
    padding: 2rem !important;
}
.gradio-container h1, .gradio-container h2, .gradio-container h3, 
.gradio-container p, .gradio-container label, .gradio-container span,
.gradio-container button {
    font-family: var(--font-mono) !important;
    color: #1f2937 !important;
}
footer { display: none !important; }

/* ========== INPUTS & TEXTAREAS ========== */
.gradio-container textarea, 
.gradio-container input[type="text"], 
.gradio-container input[type="number"],
.gradio-container .block {
    background: rgba(255, 255, 255, 0.75) !important; 
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    color: #111827 !important; 
    border: 1px solid rgba(255, 255, 255, 0.6) !important; 
    border-radius: 14px !important; 
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.02), inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
    transition: all 0.25s ease !important;
}
.gradio-container textarea:focus, 
.gradio-container input:focus,
.gradio-container .block:focus-within {
    border-color: rgba(0, 0, 0, 0.2) !important; 
    background: rgba(255, 255, 255, 0.95) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.04) !important;
    outline: none !important;
}

/* ========== BUTTONS ========== */
.gradio-container button.primary,
.gradio-container .gr-button-primary { 
    background: #000000 !important; 
    color: #ffffff !important; 
    border-radius: 18px !important; 
    padding: 12px 24px !important; 
    font-weight: 500 !important; 
    border: none !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important; 
}
.gradio-container button.primary:hover,
.gradio-container .gr-button-primary:hover { 
    background: #ff3b30 !important; 
    transform: translateY(-1px) !important; 
    box-shadow: 0 6px 20px rgba(255, 59, 48, 0.2) !important; 
}
.gradio-container button.primary:active,
.gradio-container .gr-button-primary:active {
    transform: translateY(0px) !important;
}

/* ========== TABS ========== */
.gradio-container .tabs { 
    border: none !important; 
    background: transparent !important; 
}
.gradio-container .tab-nav { 
    border-bottom: 1px solid rgba(0, 0, 0, 0.05) !important; 
    padding-left: 0 !important; 
    gap: 8px !important; 
    display: flex !important;
}
.gradio-container .tab-nav button { 
    color: #86868b !important; 
    font-weight: 500 !important; 
    background: transparent !important; 
    font-size: 0.85rem !important; 
    padding: 12px 20px !important; 
    border-radius: 8px 8px 0 0 !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.gradio-container .tab-nav button:hover { 
    background: rgba(255, 255, 255, 0.4) !important; 
    color: #000000 !important; 
}
.gradio-container .tab-nav button.selected { 
    color: #000000 !important; 
    border-bottom: 2px solid #ff3b30 !important; 
    background: rgba(255, 255, 255, 0.8) !important;
}

/* ========== TOOLTIPS ========== */
.pro-tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
    border-bottom: 2px dotted #ff3b30; 
    font-weight: 600;
    color: #000000;
}
.pro-tooltip .tooltip-text {
    visibility: hidden;
    width: max-content;
    max-width: 300px;
    background: rgba(0, 0, 0, 0.95); 
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: #ffffff !important;
    padding: 12px 16px;
    border-radius: 12px;
    position: absolute;
    z-index: 999;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%) translateY(8px);
    opacity: 0;
    transition: all 0.2s ease;
    font-size: 0.8rem;
    font-weight: 400;
    line-height: 1.4;
    pointer-events: none;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
}
.pro-tooltip:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
    transform: translateX(-50%) translateY(0);
}

/* ========== TABLES ========== */
.gradio-container table { 
    border-radius: 12px !important; 
    border-collapse: collapse !important;
    overflow: hidden !important; 
    background: #ffffff !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
}
.gradio-container tbody tr:hover td {
    background-color: rgba(0, 0, 0, 0.03) !important; 
    transition: background-color 0.15s ease !important;
}

/* ========== INPUT LABEL FIX ========== */
.gradio-container label,
.gradio-container .label,
.gradio-container [data-testid="block-info"],
.gradio-container span[class*="label"] {
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    color: #1d1d1f !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
.gradio-container label span,
.gradio-container .label span {
    color: #1d1d1f !important;
    background: transparent !important;
    background-color: transparent !important;
}

/* ========== BLOCK GLASSMORPHISM ========== */
.gradio-container .block {
    background: rgba(255, 255, 255, 0.75) !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.02) !important;
}

/* ========== TABLE HOVER ========== */
table tr:hover td {
    background-color: #e0e7ff !important;
    transition: background-color 0.2s ease;
}

/* ========== RADIO BUTTONS ========== */
.gr-radio-group .gr-radio {
    background: rgba(255,255,255,0.7) !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    margin: 4px !important;
    border: 1px solid #cbd5e1 !important;
    transition: all 0.2s ease;
}
.gr-radio-group .gr-radio.selected {
    background: #4f46e5 !important;
    color: #ffffff !important;
    border-color: #4f46e5 !important;
}
.gr-radio-group label {
    cursor: pointer;
    font-size: 0.9rem !important;
}
/* NCBI fetch button – matching glassmorphism theme */
/* Center the NCBI fetch button */
#fetch-btn {
    display: block !important;
    margin: 0 auto !important;
}
"""

with gr.Blocks(title="EpiRNA-C") as app:
    with gr.Row():
        with gr.Column(scale=4):

            gr.HTML("""
                <div style="margin-bottom: 1.5rem; display: flex; flex-direction: column; gap: 0.25rem;">
                    <h1 style="font-size: 3.5rem; margin: 0; font-weight: 900; letter-spacing: -0.05em; background: linear-gradient(135deg, #4f46e5 0%, #ec4899 50%, #e11d48 100%); -webkit-background-clip: text; background-clip: text; color: transparent;">EpiRNA</h1>
                    <p style="font-size: 1.1rem; color: #4b5563; margin: 0; font-weight: 500; letter-spacing: -0.01em;">Decoding RNA Catalytic Boundaries at Single‑Nucleotide Resolution</p>
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0.5rem;">
    <div style="background: #e0e7ff; color: #4f46e5; font-size: 0.75rem; font-weight: 600; padding: 0.25rem 0.75rem; border-radius: 20px; letter-spacing: 0.02em;">
        ⚡ Live: Biophysical Tensor Fusion (CNN)
    </div>
    <div style="color: #6b7280; font-size: 0.75rem; font-weight: 500;">
    <a href="https://huggingface.co/spaces/supzammy/epiRNAT" style="color: #4f46e5; text-decoration: none; border-bottom: 1px dotted #4f46e5;">🔬 Advanced: Transformer variant</a>
    </div>
</div>
            """)

            # NCBI streaming – stacked layout, button centered via CSS
            ncbi_acc = gr.Textbox(
                label="Fetch from NCBI (accession)",
                lines=1,
                placeholder="e.g. NM_000558.5"
            )
            fetch_btn = gr.Button("Fetch", variant="primary", size="sm", elem_id="fetch-btn")
            fetch_status = gr.HTML(visible=False)
                      
            seq_input = gr.Textbox(label="RNA Sequence (≥41bp)", lines=3)
            threshold_radio = gr.Radio(
                choices=[
                    ("🔍 Discovery (τ=0.0)", 0.0),
                    ("⚖️ Standard (τ=0.45)", 0.45),
                    ("🔬 Strict (τ=0.7)", 0.7),
                    ("🏥 Clinical (τ=0.9)", 0.9)
                ],
                value=0.45,
                label="Detection Mode",
                info="Filter sites by confidence. Discovery shows all DRACH; Clinical only strongest."
            )
            run_btn = gr.Button("Analyze & Explain AI", variant="primary")            
        with gr.Column(scale=8):
            with gr.Tabs():
                with gr.Tab("EBCS Profile"):
                    out_plot = gr.Plot()
                    out_res = gr.HTML()
                    out_mot = gr.HTML()
                with gr.Tab("AI Attribution (Captum)"):
                    exp_plot = gr.Plot()
                    exp_res = gr.HTML()
                with gr.Tab("Batch Processing"):
                    batch_file = gr.File(label="Upload CSV/FASTA")
                    batch_btn = gr.Button("Run Batch")
                    batch_status = gr.HTML()
                    batch_download = gr.File(label="Download Results")
                with gr.Tab("Science & Architecture"):
                    gr.HTML("""
                    <div style="max-width: 900px; margin: 0 auto; color: #1f2937; font-family: system-ui, sans-serif;">
                        <h3 style="margin-top: 0; color: #111827; font-weight: 600;">The "Clever Hans" Effect in Epitranscriptomics</h3>
                        <p style="margin-top: 5px; color: #374151;">Traditional deep learning models for RNA modifications overfit to lab-specific technical noise (like <span class="pro-tooltip">GC-content bias<span class="tooltip-text">A common laboratory artifact where sequencing machines preferentially read sequences rich in Guanine (G) and Cytosine (C), tricking AI models into correlating GC% with RNA modifications.</span></span>). They fail to generalize across unseen datasets.</p>
                        <h3 style="margin-top: 25px; color: #111827; font-weight: 600;">The Zero-Shot Solution</h3>
                        <p style="margin-top: 5px; color: #374151;">EpiRNA leverages a <span class="pro-tooltip">DANN<span class="tooltip-text">Domain Adversarial Neural Network.</span></span> trained on <span class="pro-tooltip">SSB<span class="tooltip-text">Synthetic Sandbox Bootstrapping.</span></span>. By mathematically stripping away technical batch artifacts, it learns true causal biology.</p>
                        <h3 style="margin-top: 25px; color: #111827; font-weight: 600;">What is EBCS?</h3>
                        <p style="margin-top: 5px; color: #374151;">Epitranscriptomic Boundary Contrast Scoring (<span class="pro-tooltip">EBCS<span class="tooltip-text">A zero-shot mathematical probe that calculates the exact single-nucleotide derivative of an AI model's confidence.</span></span>) slides a synthetic mask across the sequence to calculate the mathematical derivative of the model's confidence. The <span class="pro-tooltip">peak contrast delta<span class="tooltip-text">The highest point on the blue graph line.</span></span> reveals the exact single-nucleotide catalytic boundary the AI relies upon.</p>
                        <hr style="margin: 30px 0; border-color: #e5e7eb;">
                        <h2 style="color: #4f46e5; margin-bottom: 16px;"> The Biophysical Tensor Fusion Paradigm</h2>
                        <p>
                            EpiRNA replaces traditional one‑hot nucleotide encoding with a <strong>3‑dimensional biophysical vector</strong>
                            for each base, directly embedding the chemical properties that govern RNA catalysis:
                        </p>
                        <table class="bio-table" style="width: 100%; border-collapse: collapse; margin: 16px 0;">
                            <tr style="background: #e0e7ff;">
                                <th style="padding: 8px; text-align: left;">Base</th>
                                <th style="padding: 8px; text-align: left;">H‑Bond Potential</th>
                                <th style="padding: 8px; text-align: left;">Stacking Energy</th>
                                <th style="padding: 8px; text-align: left;">Solvent Accessibility</th>
                            </tr>
                            <tr><td>A</td><td>+1.0</td><td>−1.0</td><td>+0.5</td></tr>
                            <tr><td>U/T</td><td>−1.0</td><td>−1.0</td><td>−0.5</td></tr>
                            <tr><td>C</td><td>−1.0</td><td>+1.0</td><td>+2.5</td></tr>
                            <tr><td>G</td><td>+1.0</td><td>+1.0</td><td>−1.0</td></tr>
                        </table>
                        <p>
                            This physical grounding allows the model to <strong>inherently discriminate</strong> functional
                            cytosine‑containing motifs (like DRACH) from inert decoys, without requiring explicit motif annotation.
                        </p>
                        <h3 style="color: #4f46e5; margin-top: 24px;"> Multi‑Path Dilated Convolution</h3>
                        <p>The sequence is processed by three parallel 1D‑convolutional arms:</p>
                        <ul>
                            <li><strong>Local Path</strong> (kernel=3) – captures immediate base‑pair interactions.</li>
                            <li><strong>Flank Path</strong> (kernel=5, dilation=2) – senses mid‑range structural context.</li>
                            <li><strong>Structure Path</strong> (kernel=5, dilation=4) – detects long‑range backbone curvature.</li>
                        </ul>
                        <p>
                            All arms use <code>MaxPool1d</code> to prevent background smearing at transition boundaries,
                            then are concatenated and normalised before the final contrast head.
                        </p>
                        <h3 style="color: #4f46e5; margin-top: 24px;"> Adaptive Calibration & Noise Gate</h3>
                        <p>
                            Raw delta scores are calibrated with a <strong>local‑global variance blender</strong>:
                            a Z‑score is computed using a blended standard deviation (30% local window, 70% global),
                            then mapped to [0,1] via a shifted sigmoid. This eliminates logit saturation and
                            ensures stable, comparable scores across sequences of any length.
                        </p>
                        <p>
                            A final production <strong>noise gate (threshold = 0.45)</strong> zeroes out low‑confidence
                            background fluctuations caused by abrupt GC‑content transitions, leaving only
                            genuine catalytic peaks in the visualisation.
                        </p>
                        <h3 style="color: #4f46e5; margin-top: 24px;"> Multi‑Target DRACH Alignment</h3>
                        <p>
                            Instead of simply reporting the highest score, the pipeline searches for canonical
                            <code>[AGU][AG]AC[ACU]</code> motifs and pinpoints the <strong>modifying adenosine</strong>
                            (position +2 from the motif start). If no DRACH motif is found, it falls back to
                            the centre of high‑score plateaus (≥0.85). This biologically informed peak‑picking
                            rejects false positives from non‑functional patterns.
                        </p>
                        <h3 style="color: #4f46e5; margin-top: 24px;"> Variable‑Length Capable</h3>
                        <p>
                            The model accepts <strong>any sequence ≥41 bp</strong> by sliding a 41‑nucleotide window
                            with overlapping averaging, making it suitable for full‑length transcripts,
                            genomic RNA fragments, and synthetic constructs.
                        </p>
                        <hr style="margin: 32px 0; border-color: #e5e7eb;">
                        <p style="font-size: 0.9rem; color: #6b7280;">
                            <em>Model weights pre‑trained on curated epi‑transcriptomic datasets.
                            For technical details and benchmarks, see the project repository.</em>

                        
                    """)
    # Unified logic: Single button triggers everything
    def run_all(seq, threshold):
        fig, res, mot = predict(seq, threshold)
        exp_fig, exp_text = run_explainer(seq)
        return fig, res, mot, exp_fig, exp_text

    run_btn.click(run_all,
                  inputs=[seq_input, threshold_radio],
                  outputs=[out_plot, out_res, out_mot, exp_plot, exp_res])

    batch_btn.click(process_batch, inputs=[batch_file], outputs=[batch_download, batch_status])

    fetch_btn.click(
        fn=fetch_ncbi,
        inputs=[ncbi_acc],
        outputs=[seq_input, fetch_status]
    )

app.queue().launch(theme=glass_theme, css=custom_css)