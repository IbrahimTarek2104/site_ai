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
# 📋 PASTE YOUR COPIED DROPBOX LINK DIRECTLY HERE (ENSURE IT ENDS WITH dl=1)
DROPBOX_URL = "https://www.dropbox.com/scl/fi/abc123xyz/unet_best.pth?rlkey=xyz123&dl=1"

model = None  # Global fallback definition to prevent initialization NameErrors
@st.cache_resource
def download_model_from_dropbox():
    os.makedirs("models", exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 1000000:
        os.remove(MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        with st.spinner(" Streaming trained model weights securely from Dropbox..."):
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
    model_obj = AttentionUNet(in_channels=3, base=64, drop=0.2)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    model_obj.load_state_dict(state_dict, strict=False)
    model_obj.eval()
    return model_obj

try:
    model = load_model()
    st.sidebar.success(" Attention U-Net Core Online")
except Exception as e:
    st.sidebar.error(f" Model weight load error: {e}")

# ==========================================
# 2. Simplified Sidebar Controls
# ==========================================
st.sidebar.header(" Optimization Settings")
num_candidates = st.sidebar.slider("Number of Sites to Allocate", 5, 50, 10, 5)

st.sidebar.write("---")
st.sidebar.markdown(" **Genetic Algorithm Fine-Tuning**")
ga_pop_size = st.sidebar.slider("Population Size (Chromosomes)", 20, 100, 40, 10)
ga_generations = st.sidebar.slider("Evolution Generations", 10, 100, 30, 5)
ga_mutation_rate = st.sidebar.slider("Mutation Probability", 0.01, 0.30, 0.15, 0.05)

st.sidebar.write("---")
st.sidebar.header(" Multi-Tier Priority Weights")
w_pop = st.sidebar.slider("Population Demand Weight", 0.0, 1.0, 0.70, 0.05)
w_elev = st.sidebar.slider("Terrain Topography Weight", 0.0, 1.0, 0.30, 0.05)

# Enforce normalization of priority weights
total_w = w_pop + w_elev
if total_w > 0:
    w_pop, w_elev = w_pop/total_w, w_elev/total_w

# Strict Operational Constants
REQUIRED_ISD_M = 1500.0        
TARGET_RADIUS_M = 1500.0     

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
    def __init__(self, priority_surface, dist_to_legacy_m, num_towers, pad_distance_px, pop_size=40, mutation_rate=0.15):
        self.surface = priority_surface
        self.dist_to_legacy_m = dist_to_legacy_m
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
            if self.dist_to_legacy_m[r, c] >= REQUIRED_ISD_M:
                coords.append([r, c])
        return np.array(coords)

    def calculate_fitness(self, chromosome):
        score = 0.0
        for r, c in chromosome:
            ir, ic = int(r), int(c)
            if self.dist_to_legacy_m[ir, ic] < REQUIRED_ISD_M:
                return 0.0
            score += self.surface[ir, ic]
        
        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                dist_px = np.linalg.norm(chromosome[i] - chromosome[j])
                if dist_px < self.pad_px:
                    return 0.0  
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

            if fitness_scores.sum() == 0:
                population = [self._generate_valid_chromosome() for _ in range(self.pop_size)]
                continue

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

        if best_chromosome is None or best_fitness <= 0.001:
            best_chromosome = population[0]

        return best_chromosome

# ==========================================
# 5. Data Ingestion & Inference Loop
# ==========================================
st.subheader("Step 1: Regional Environment Ingestion")
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

    # Neural Network Background Pipeline
    prob_map = np.zeros((orig_h, orig_w), dtype=np.float32)
    if model is not None:
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
                
            prob_map = np.where(wgt_acc > 0, prob_acc / wgt_acc, 0.0).astype(np.float32)[0:orig_h, 0:orig_w]

    # Hard Constraint Core Filtering Calculations
    is_covered = (features_stack[:, :, 0] > 0.4).astype(np.uint8)
    is_uncovered_mask = (1 - is_covered).astype(np.float32)
    
    distance_to_legacy_px = distance_transform_edt(is_uncovered_mask)
    distance_to_legacy_m = distance_to_legacy_px * pixel_m
    
    pop_n = features_stack[:, :, 1]
    elev_bad = 1.0 - features_stack[:, :, 2]  
    
    priority_raw = is_uncovered_mask * np.where(distance_to_legacy_m >= REQUIRED_ISD_M, 1.0, 0.0) * ((w_pop * pop_n) + (w_elev * elev_bad))
    priority_compressed = np.power(np.clip(priority_raw, 0, 1), 0.7)
    priority_base = gaussian_filter(priority_compressed, sigma=3).astype(np.float32)
    p_min, p_max = float(priority_base.min()), float(priority_base.max())
    p_range = p_max - p_min if (p_max - p_min) > 1e-5 else 1.0
    priority_base = (priority_base - p_min) / p_range

    # ==========================================
    # 6. Execute Genetic Optimization
    # ==========================================
    st.write("---")
    st.subheader(" Step 2: Genetic Layout Evolution Engine")
    
    with st.spinner("Initializing population chromosomes and calculating evolutionary fitness..."):
        min_dist_px = max(5, int(REQUIRED_ISD_M / pixel_m))
        radius_px = int(TARGET_RADIUS_M / pixel_m)
        total_map_pixels = orig_h * orig_w
        
        ga_engine = CellularGeneticOptimizer(
            priority_surface=priority_base, dist_to_legacy_m=distance_to_legacy_m,
            num_towers=num_candidates, pad_distance_px=min_dist_px, pop_size=ga_pop_size, mutation_rate=ga_mutation_rate
        )
        
        optimal_layout = ga_engine.evolve(generations=ga_generations)
        
        # Calculate strict spatial matrix coverage progression
        running_coverage_mask = is_covered.copy()
        initial_covered_pixels = np.sum(running_coverage_mask)
        baseline_percentage = (initial_covered_pixels / total_map_pixels) * 100.0
        
        candidates = []
        for step, (r, c) in enumerate(optimal_layout):
            ir, ic = int(r), int(c)
            native_lon, native_lat = rasterio.transform.xy(transform, ir, ic, offset="center")
            longitudes, latitudes = warp_transform(meta['crs'], 'EPSG:4326', [native_lon], [native_lat])
            site_rsrp, site_sinr = simulate_local_physics(ir, ic, (orig_h, orig_w), pixel_m)
            
            # Fetch the specific pixel priority score
            pixel_priority_score = float(priority_base[ir, ic])
            
            # Calculate standalone footprint gain
            y_indices, x_indices = np.ogrid[:orig_h, :orig_w]
            dist_from_tower = np.sqrt((y_indices - ir)**2 + (x_indices - ic)**2)
            tower_mask = (dist_from_tower <= radius_px).astype(np.uint8)
            
            # Extract unserved pixels captured exclusively by this tower
            new_uncovered_pixels = np.sum((tower_mask == 1) & (running_coverage_mask == 0))
            site_coverage_increase_pct = (new_uncovered_pixels / total_map_pixels) * 100.0
            
            # Update the global map context for next index evaluation
            running_coverage_mask = np.bitwise_or(running_coverage_mask, tower_mask)
            
            candidates.append({
                "rank": step+1, "lat": latitudes[0], "lon": longitudes[0],
                "native_lat": round(native_lat, 1), "native_lon": round(native_lon, 1),
                "rsrp": round(site_rsrp, 1), "sinr": round(site_sinr, 1),
                "pixel_priority": round(pixel_priority_score, 3),
                "coverage_gain": round(site_coverage_increase_pct, 2)
            })
            
        df_candidates = pd.DataFrame(candidates)
        final_covered_pixels = np.sum(running_coverage_mask)
        final_percentage = (final_covered_pixels / total_map_pixels) * 100.0

    # Extract downsampled coordinates for visualization
    with st.spinner("Extracting map visualization data layers..."):
        step_stride = max(1, int(max(orig_h, orig_w) / 150))
        legacy_cells = []
        heatmap_data = []
        
        for r in range(0, orig_h, step_stride):
            for c in range(0, orig_w, step_stride):
                n_lon, n_lat = rasterio.transform.xy(transform, r, c, offset="center")
                g_lons, g_lats = warp_transform(meta['crs'], 'EPSG:4326', [n_lon], [n_lat])
                
                if is_covered[r, c] > 0.4:
                    legacy_cells.append({"lon": g_lons[0], "lat": g_lats[0]})
                
                val = float(priority_base[r, c])
                if val > 0.03:
                    heatmap_data.append({"lon": g_lons[0], "lat": g_lats[0], "weight": val})
                    
        df_legacy = pd.DataFrame(legacy_cells)
        df_heatmap = pd.DataFrame(heatmap_data)

    # ==========================================
    # 7. Spatial Visualization & Reports
    # ==========================================
    st.write("---")
    
    HIGH_DENSITY_CMAP = [
        [0, 0, 30, 0], [30, 0, 100, 45], [0, 60, 200, 70], [0, 120, 255, 95],
        [0, 180, 220, 120], [0, 220, 150, 140], [0, 245, 80, 160], [100, 255, 0, 180],
        [190, 255, 0, 195], [255, 255, 0, 210], [255, 190, 0, 225], [255, 120, 0, 235],
        [255, 50, 0, 245], [220, 0, 40, 255], [160, 0, 80, 255]
    ]
    LEGACY_RED_CMAP = [
        [230, 50, 50, 0], [230, 50, 50, 50], [230, 50, 50, 95], 
        [230, 50, 50, 140], [230, 50, 50, 185], [230, 50, 50, 220]
    ]
    
    view_state = pdk.ViewState(
        latitude=df_candidates['lat'].mean(), longitude=df_candidates['lon'].mean(), zoom=11.6, pitch=0
    )
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### MAP 1: Boundary-Locked Legal Suitability Surface")
        layer_priority_heatmap = pdk.Layer(
            "HeatmapLayer", df_heatmap, get_position="[lon, lat]", get_weight="weight",
            radius_pixels=25, intensity=2.5, threshold=0.02, aggregation='"MEAN"', color_range=HIGH_DENSITY_CMAP
        )
        st.pydeck_chart(pdk.Deck(layers=[layer_priority_heatmap], initial_view_state=view_state))
        st.caption("The suitability surface zeroes out completely near old cells to enforce strict spacing constraints.")

    with col_right:
        st.markdown("####  MAP 2: Allocation Matrix Deployment Blueprint")
        layer_legacy_heatmap = pdk.Layer(
            "HeatmapLayer", df_legacy, get_position="[lon, lat]",
            radius_pixels=22, intensity=3.5, threshold=0.04, color_range=LEGACY_RED_CMAP, pickable=False
        )
        layer_new_footprints = pdk.Layer(
            "ScatterplotLayer", df_candidates, get_position="[lon, lat]", get_radius=TARGET_RADIUS_M,               
            get_fill_color=[40, 220, 100, 65], get_line_color=[0, 170, 60, 200], line_width_min_pixels=2
        )
        layer_tower_mast = pdk.Layer(
            "ColumnLayer", df_candidates, get_position="[lon, lat]", get_elevation=350,
            radius=60, get_fill_color=[0, 240, 255, 255], extruded=True, pickable=True
        )
        
        st.pydeck_chart(pdk.Deck(layers=[layer_legacy_heatmap, layer_new_footprints, layer_tower_mast], initial_view_state=view_state))
        
        st.markdown(
            """
            <div style="display: flex; gap: 20px; font-size: 13px; margin-top: 5px; justify-content: center;">
                <div>🔴 <span style="color:#e63232; font-weight:bold;">Already Covered</span></div>
                <div>🔷 <span style="color:#00f0ff; font-weight:bold;">New Masts</span></div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # Analytics Performance Report
    st.write("---")
    st.write("###  Evolved Candidate Site Allocation Metrics")
    
    # Structural Dashboard Metrics
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Baseline Regional Coverage", f"{baseline_percentage:.2f}%")
    with m2: st.metric("Post-Evolution Global Coverage", f"{final_percentage:.2f}%")
    with m3: st.metric("Net Coverage Increase Delta", f"+{(final_percentage - baseline_percentage):.2f}%", delta_color="normal")
    
    st.write("#### Detailed Individual Site Efficiency Log")
    st.dataframe(
        df_candidates[["rank", "native_lat", "native_lon", "pixel_priority", "coverage_gain", "rsrp"]].rename(
            columns={
                "native_lat": "Northing (m)", 
                "native_lon": "Easting (m)", 
                "pixel_priority": "Target Pixel Priority Score", 
                "coverage_gain": "Individual Coverage Increase (%)", 
                "rsrp": "Est Center RSRP (dBm)"
            }
        ), use_container_width=True, hide_index=True
    )
else:
    st.info("👈 Please upload all three foundational environment rasters in the main layout panel to initiate the spatial allocation engine.")
