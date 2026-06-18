import os
import urllib.request
import streamlit as st
import numpy as np
import torch
import rasterio
from rasterio.warp import transform as warp_transform
from scipy.ndimage import distance_transform_edt, gaussian_filter
import pandas as pd
import pydeck as pdk
from pipeline.network import AttentionUNet
from pipeline.data_pipeline import ProductionInferencePipeline

# ==========================================
# 1. Page Configuration & Base Setup
# ==========================================
st.set_page_config(page_title="AI Site Planner", layout="wide", initial_sidebar_state="expanded")
st.title("🛰️ Geospatial Cellular Site Planning Dashboard")
st.write("---")

pipeline = ProductionInferencePipeline(patch_size=64)

MODEL_PATH = "models/unet_best.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/abc123xyz/unet_best.pth?rlkey=xyz123&dl=1"

@st.cache_resource
def download_model_from_dropbox():
    os.makedirs("models", exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 1000000:
        os.remove(MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Streaming trained model weights securely from Dropbox..."):
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(DROPBOX_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"Failed to stream from Dropbox. Error: {e}")
    return MODEL_PATH

download_model_from_dropbox()

@st.cache_resource
def load_model():
    model = AttentionUNet(in_channels=3, base=64, drop=0.2)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Attention U-Net Core Online")
except Exception as e:
    st.sidebar.error(f"❌ Model weight load error: {e}")

# ==========================================
# 2. Simplified Sidebar Controls
# ==========================================
st.sidebar.header("🧬 Optimization Settings")
num_candidates = st.sidebar.slider("Number of Sites to Allocate", 5, 50, 10, 5)

st.sidebar.write("---")
st.sidebar.markdown("🔬 **Genetic Algorithm Fine-Tuning**")
ga_pop_size = st.sidebar.slider("Population Size (Chromosomes)", 20, 100, 40, 10)
ga_generations = st.sidebar.slider("Evolution Generations", 10, 100, 30, 5)
ga_mutation_rate = st.sidebar.slider("Mutation Probability", 0.01, 0.30, 0.15, 0.05)

st.sidebar.write("---")
st.sidebar.header("📊 Multi-Tier Priority Weights")
w_prob = st.sidebar.slider("Model Confidence Weight (U-Net)", 0.0, 1.0, 0.35, 0.05)
w_pop = st.sidebar.slider("Population Demand Weight", 0.0, 1.0, 0.30, 0.05)
w_sinr = st.sidebar.slider("Baseline Service Gap Weight", 0.0, 1.0, 0.20, 0.05)
w_elev = st.sidebar.slider("Terrain Topography Weight", 0.0, 1.0, 0.15, 0.05)

# Enforce normalization of priority weights
total_w = w_prob + w_pop + w_sinr + w_elev
if total_w > 0:
    w_prob, w_pop, w_sinr, w_elev = w_prob/total_w, w_pop/total_w, w_sinr/total_w, w_elev/total_w

FIXED_ISD_M = 1500.0        
TARGET_RADIUS_M = 1500.0     # 🚀 Hardlocked to exactly 1.5 KM as requested

# ==========================================
# 3. Local RF Physics Propagation Engine
# ==========================================
def simulate_local_physics(r, c, shape, pixel_m):
    h, w = shape
    yx = np.indices((h, w))
    dist_m = np.sqrt((yx[0] - r)**2 + (yx[1] - c)**2) * pixel_m
    dist_m = np.maximum(dist_m, pixel_m) 
    
    simulated_rsrp = -50.0 - (44.9 - 6.55 * np.log10(30.0)) * np.log10(np.maximum(dist_m/1000.0, 0.001))
    simulated_rsrp = np.clip(simulated_rsrp, -130.0, -44.0)
    
    simulated_sinr = simulated_rsrp - (-95.0) - (dist_m / TARGET_RADIUS_M) * 12.0
    return float(simulated_rsrp[r, c]), float(simulated_sinr[r, c])

# ==========================================
# 4. Core Genetic Algorithm Optimization Tier
# ==========================================
class CellularGeneticOptimizer:
    def __init__(self, priority_surface, num_towers, pad_distance_px, pop_size=40, mutation_rate=0.15):
        self.surface = priority_surface
        self.num_towers = num_towers
        self.pad_px = pad_distance_px
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.height, self.width = priority_surface.shape

    def _generate_valid_chromosome(self):
        coords = []
        while len(coords) < self.num_towers:
            r = np.random.randint(self.pad_px, self.height - self.pad_px)
            c = np.random.randint(self.pad_px, self.width - self.pad_px)
            coords.append([r, c])
        return np.array(coords)

    def calculate_fitness(self, chromosome):
        score = 0.0
        for r, c in chromosome:
            score += self.surface[int(r), int(c)]
        
        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                dist = np.linalg.norm(chromosome[i] - chromosome[j])
                if dist < self.pad_px:
                    score *= 0.70  
        return max(0.001, float(score))

    def evolve(self, generations=30):
        population = [self._generate_valid_chromosome() for _ in range(self.pop_size)]
        best_chromosome = None
        best_fitness = -1.0

        for gen in range(generations):
            fitness_scores = np.array([self.calculate_fitness(chrom) for chrom in population])
            
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_chromosome = population[max_idx].copy()

            prob_distribution = fitness_scores / fitness_scores.sum()
            selected_indices = np.random.choice(self.pop_size, size=self.pop_size, p=prob_distribution)
            population = [population[idx].copy() for idx in selected_indices]

            next_generation = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = population[i], population[min(i+1, self.pop_size-1)]
                mask = np.random.rand(self.num_towers) > 0.5
                c1 = np.where(mask[:, None], p1, p2)
                c2 = np.where(~mask[:, None], p1, p2)
                
                for child in [c1, c2]:
                    if np.random.rand() < self.mutation_rate:
                        mutate_idx = np.random.randint(0, self.num_towers)
                        child[mutate_idx, 0] = np.clip(child[mutate_idx, 0] + np.random.randint(-15, 16), self.pad_px, self.height - self.pad_px)
                        child[mutate_idx, 1] = np.clip(child[mutate_idx, 1] + np.random.randint(-15, 16), self.pad_px, self.width - self.pad_px)
                    next_generation.append(child)
            population = next_generation[:self.pop_size]

        return best_chromosome

# ==========================================
# 5. Data Ingestion & Inference Loop
# ==========================================
st.subheader("🌐 Step 1: Regional Environment Ingestion")
cx1, cx2, cx3 = st.columns(3)
with cx1: cov_file = st.file_uploader("Upload Baseline Coverage (.tif)", type=["tif", "tiff"])
with cx2: pop_file = st.file_uploader("Upload Population Density (.tif)", type=["tif", "tiff"])
with cx3: elev_file = st.file_uploader("Upload Terrain Topography (.tif)", type=["tif", "tiff"])

if cov_file and pop_file and elev_file:
    with st.spinner("Processing geospatial rasters and coregistering matrices..."):
        features_stack, meta = pipeline.extract_and_normalize(cov_file, pop_file, elev_file)
        transform = meta['transform']
        pixel_m = abs(transform.a)
        
        patches, coords, meta_shapes = pipeline.generate_gaussian_patches(features_stack)
        orig_h, orig_w, pad_h, pad_w = meta_shapes
        
    st.success(f"Geospatial arrays processed successfully! Active layout: {orig_h}x{orig_w} pixels at {pixel_m:.1f}m/px.")

    # Neural Network Spatial Prediction Pipeline
    with st.spinner("Executing Attention U-Net Inference across patch spaces..."):
        prob_acc = np.zeros((pad_h, pad_w), dtype=np.float64)
        wgt_acc = np.zeros((pad_h, pad_w), dtype=np.float64)
        
        g1d = np.exp(-0.5 * ((np.arange(64) - 32) / 16) ** 2)
        gauss_window = np.outer(g1d, g1d)
        
        for patch, (i, j) in zip(patches, coords):
            tensor_input = torch.tensor(patch).permute(2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                pred_patch = model(tensor_input).squeeze().numpy()
                
            prob_acc[i*64:(i+1)*64, j*64:(j+1)*64] += pred_patch * gauss_window
            wgt_acc[i*64:(i+1)*64, j*64:(j+1)*64] += gauss_window
            
        prob_map = np.where(wgt_acc > 0, prob_acc / wgt_acc, 0.0).astype(np.float32)
        prob_map = prob_map[0:orig_h, 0:orig_w]

    # Integrated Unified Priority Scoring Function
    pop_n = features_stack[:, :, 1]
    sinr_bad = 1.0 - features_stack[:, :, 0] 
    elev_bad = 1.0 - features_stack[:, :, 2]
    
    priority_raw = (w_prob * prob_map) + (w_pop * pop_n) + (w_sinr * sinr_bad) + (w_elev * elev_bad)
    priority_compressed = np.power(np.clip(priority_raw, 0, 1), 0.6)
    priority_base = gaussian_filter(priority_compressed, sigma=5).astype(np.float32)
    p_min, p_max = float(priority_base.min()), float(priority_base.max())
    p_range = p_max - p_min if (p_max - p_min) > 1e-5 else 1.0
    priority_base = (priority_base - p_min) / p_range

    # ==========================================
    # 6. Execute Genetic Optimization
    # ==========================================
    st.write("---")
    st.subheader("🧬 Step 2: Genetic Layout Evolution Engine")
    
    with st.spinner("Initializing population chromosomes and calculating evolutionary fitness..."):
        min_dist_px = max(5, int(FIXED_ISD_M / pixel_m))
        
        ga_engine = CellularGeneticOptimizer(
            priority_surface=priority_base, num_towers=num_candidates,
            pad_distance_px=min_dist_px, pop_size=ga_pop_size, mutation_rate=ga_mutation_rate
        )
        
        optimal_layout = ga_engine.evolve(generations=ga_generations)
        
        # Build candidate coordinates dataset
        candidates = []
        for step, (r, c) in enumerate(optimal_layout):
            native_lon, native_lat = rasterio.transform.xy(transform, r, c, offset="center")
            longitudes, latitudes = warp_transform(meta['crs'], 'EPSG:4326', [native_lon], [native_lat])
            site_rsrp, site_sinr = simulate_local_physics(r, c, (orig_h, orig_w), pixel_m)
            
            candidates.append({
                "rank": step+1, "lat": latitudes[0], "lon": longitudes[0],
                "native_lat": round(native_lat, 1), "native_lon": round(native_lon, 1),
                "rsrp": round(site_rsrp, 1), "sinr": round(site_sinr, 1)
            })
        df_candidates = pd.DataFrame(candidates)

    # Extract coordinates for mapping
    with st.spinner("Extracting map
