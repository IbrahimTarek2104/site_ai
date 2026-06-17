import streamlit as st
import numpy as np
import torch
import rasterio
from scipy.ndimage import distance_transform_edt, gaussian_filter
import pandas as pd
import pydeck as pdk

from pipeline.network import AttentionUNet
from pipeline.data_pipeline import ProductionInferencePipeline
import os
import requests

# Put this near the top of your app.py file, right under your imports
MODEL_PATH = "models/unet_best.pth"
FILE_ID = "1aM1V2q5OO1DWa05aFiv6giPKFQ4UVKCJ"

@st.cache_resource
def download_large_google_drive_file():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Streaming trained model weights from Google Drive (bypassing size warnings)..."):
            session = requests.Session()
            URL = "https://docs.google.com/uc?export=download"
            
            # Step 1: Send an initial request to catch the confirmation token
            response = session.get(URL, params={'id': FILE_ID}, stream=True)
            
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            
            # Step 2: If a token exists, append it to bypass the warning page
            if token:
                response = session.get(URL, params={'id': FILE_ID, 'confirm': token}, stream=True)
            
            # Step 3: Stream the true binary file chunks directly to disk
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        
    return MODEL_PATH

# Run this robust downloader block
download_large_google_drive_file()

# ==========================================
# Page Config & Core Layout Initialization
# ==========================================
st.set_page_config(page_title="AI Site Planner", layout="wide", initial_sidebar_state="expanded")
st.title("🛰️ Deep RL & Vision Cellular Site Network Planning Engine")
st.write("---")

pipeline = ProductionInferencePipeline(patch_size=64)

@st.cache_resource
@st.cache_resource
def load_model():
    model = AttentionUNet(in_channels=3, base=64, drop=0.2)
    
    # ADD weights_only=False inside the torch.load function right here:
    state_dict = torch.load("models/unet_best.pth", map_location=torch.device('cpu'), weights_only=False)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Attention U-Net Core Online")
except Exception as e:
    st.sidebar.error(f"❌ Model weight load error: {e}")

# ==========================================
# Dynamic Interactive Sidebar Widgets
# ==========================================
st.sidebar.header("🎛️ Optimization Parameters")
isd_min_m = st.sidebar.slider("Minimum Inter-Site Distance (meters)", 500, 3000, 1500, 250)
coverage_rad_m = st.sidebar.slider("Idealized Tower Coverage Radius (meters)", 500, 4500, 2250, 250)

allocation_mode = st.sidebar.radio(
    "Tower Placement Strategy Mode:",
    ["Automated Greedy Search Optimization", "Manual Engineering Coordinate Placement Mode"]
)

if allocation_mode == "Automated Greedy Search Optimization":
    num_candidates = st.sidebar.slider("Number of Sites to Allocate Automatically", 5, 50, 15, 5)
    diversity_weight = st.sidebar.slider("Spatial Diversity Bias (Exploration)", 0.0, 0.5, 0.15, 0.05)

st.sidebar.write("---")
st.sidebar.header("📊 Priority Weights")
w_prob = st.sidebar.slider("Model Confidence Weight", 0.0, 1.0, 0.40, 0.05)
w_pop = st.sidebar.slider("Population Demand Weight", 0.0, 1.0, 0.30, 0.05)
w_sinr = st.sidebar.slider("Interference Mitigation Weight", 0.0, 1.0, 0.20, 0.05)
w_elev = st.sidebar.slider("Terrain Disadvantage Weight", 0.0, 1.0, 0.10, 0.05)

total_w = w_prob + w_pop + w_sinr + w_elev
if total_w > 0:
    w_prob, w_pop, w_sinr, w_elev = w_prob/total_w, w_pop/total_w, w_sinr/total_w, w_elev/total_w

# ==========================================
# Real-Time Physics Propagation Simulator
# ==========================================
def simulate_local_physics(r, c, shape, pixel_m, radius_m):
    """Simulates localized path loss arrays (RSRP & SINR) around selected pixels."""
    h, w = shape
    yx = np.indices((h, w))
    dist_m = np.sqrt((yx[0] - r)**2 + (yx[1] - c)**2) * pixel_m
    dist_m = np.maximum(dist_m, pixel_m)
    
    # COST-231 Hata empirical degradation approximation curve logic
    simulated_rsrp = 46.0 - (44.9 - 6.55 * np.log10(30.0)) * np.log10(np.maximum(dist_m/1000.0, 0.001))
    simulated_rsrp = np.clip(simulated_rsrp, -140.0, -44.0)
    
    simulated_sinr = simulated_rsrp - (-97.0) - (dist_m / radius_m) * 15.0
    simulated_sinr = np.clip(simulated_sinr, -10.0, 30.0)
    
    return float(simulated_rsrp[r, c]), float(simulated_sinr[r, c])

# ==========================================
# Data Ingestion Layer Pipeline Execution
# ==========================================
st.subheader("🌐 Step 1: Regional Environment Ingestion")
cx1, cx2, cx3 = st.columns(3)
with cx1: cov_file = st.file_uploader("Upload Baseline Coverage (.tif)", type=["tif", "tiff"])
with cx2: pop_file = st.file_uploader("Upload Population Density (.tif)", type=["tif", "tiff"])
with cx3: elev_file = st.file_uploader("Upload Terrain Topography (.tif)", type=["tif", "tiff"])

if cov_file and pop_file and elev_file:
    with st.spinner("Processing rasters through clean production pipeline..."):
        features_stack, meta = pipeline.extract_and_normalize(cov_file, pop_file, elev_file)
        transform = meta['transform']
        pixel_m = abs(transform.a)
        
        patches, coords, meta_shapes = pipeline.generate_gaussian_patches(features_stack)
        orig_h, orig_w, pad_h, pad_w = meta_shapes
        
    st.success(f"Environment Stack Ready! Matrix resolution: {orig_h}x{orig_w} pixels at {pixel_m:.1f}m pixels.")
    
    # Blended Patch space inference loop
    with st.spinner("Executing Attention U-Net Inference across patch space..."):
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
        pred_map = (prob_map >= 0.5).astype(np.uint8)

    # Calculate Optimization Surface Priorities
    pop_n = features_stack[:, :, 1]
    sinr_bad = 1.0 - features_stack[:, :, 0] 
    elev_bad = 1.0 - features_stack[:, :, 2]
    
    priority_raw = (w_prob * prob_map) + (w_pop * pop_n) + (w_sinr * sinr_bad) + (w_elev * elev_bad)
    priority_compressed = np.power(np.clip(priority_raw, 0, 1), 0.6)
    priority_base = gaussian_filter(priority_compressed, sigma=5).astype(np.float32)
    priority_base = (priority_base - priority_base.min()) / (priority_base.max() - priority_base.min() + 1e-8)
    
    yx = np.indices((orig_h, orig_w))
    coverage_rad_px = max(15, int(coverage_rad_m / pixel_m))

    st.write("---")
    st.subheader(f"🎯 Step 2: Optimization Engine Output — ({allocation_mode})")

    # Automated allocation search routing
    if allocation_mode == "Automated Greedy Search Optimization":
        with st.spinner("Running Greedy Spatial Search Optimizer..."):
            tower_radius_px = max(10, int(isd_min_m / pixel_m))
            priority_work = priority_base.copy()
            placed_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            candidates = []
            
            for step in range(num_candidates):
                if len(candidates) > 0:
                    placed_inv = 1 - placed_mask
                    dist_to_placed = distance_transform_edt(placed_inv).astype(np.float32)
                    dist_norm = dist_to_placed / (dist_to_placed.max() + 1e-8)
                    priority_current = np.clip(priority_work + diversity_weight * dist_norm, 0, None)
                else:
                    priority_current = priority_work.copy()
                    
                idx = np.argmax(priority_current)
                r, c = np.unravel_index(idx, priority_current.shape)
                
                if priority_current[r, c] < 0.05:
                    break
                    
                lon, lat = rasterio.transform.xy(transform, r, c, offset="center")
                site_rsrp, site_sinr = simulate_local_physics(r, c, (orig_h, orig_w), pixel_m, coverage_rad_m)
                
                disc = ((yx[0] - r)**2 + (yx[1] - c)**2) <= coverage_rad_px**2
                newly_covered = disc & (pred_map == 1)
                gain_pct = (newly_covered.sum() / (pred_map.sum() + 1e-8)) * 100.0
                
                candidates.append({
                    "rank": step+1, "lat": lat, "lon": lon, 
                    "score": float(priority_base[r, c]), "gain_pct": round(gain_pct, 2),
                    "rsrp": round(site_rsrp, 1), "sinr": round(site_sinr, 1)
                })
                
                placed_mask[r, c] = 1
                priority_work[disc] = 0.0

            df_candidates = pd.DataFrame(candidates)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.write("#### 🗺️ Interactive 3D Allocation Blueprint (Color = Target RSRP Level)")
            view_state = pdk.ViewState(latitude=df_candidates['lat'].mean(), longitude=df_candidates['lon'].mean(), zoom=13, pitch=45)
            df_candidates['color_r'] = np.where(df_candidates['rsrp'] > -85, 0, 255)
            df_candidates['color_g'] = np.where(df_candidates['rsrp'] > -85, 255, 100)
            
            tower_layer = pdk.Layer(
                "ColumnLayer", df_candidates, get_position="[lon, lat]",
                get_elevation=150, radius=40, get_fill_color="[color_r, color_g, 120, 230]", 
                pickable=True, extruded=True
            )
            st.pydeck_chart(pdk.Deck(layers=[tower_layer], initial_view_state=view_state, tooltip={"text": "Rank: {rank}\nEst RSRP: {rsrp} dBm\nEst SINR: {sinr} dB"}))
        with col2:
            st.write("#### 📈 Ranked Candidate Site Metrics")
            st.dataframe(df_candidates[["rank", "lat", "lon", "gain_pct", "rsrp", "sinr"]].rename(columns={"gain_pct": "Area Gain %", "rsrp": "Predicted RSRP (dBm)", "sinr": "Predicted SINR (dB)"}), use_container_width=True, hide_index=True)
            st.metric("Total Automated Allocations", f"{len(df_candidates)} Sites")

    # Manual structural engineering routing
    else:
        st.write("Input custom coordinate nodes to run simulated coverage footprints.")
        center_r, center_c = orig_h // 2, orig_w // 2
        def_lon, def_lat = rasterio.transform.xy(transform, center_r, center_c, offset="center")
        
        cx_lat, cx_lon = st.columns(2)
        with cx_lat: target_lat = st.number_input("Target Node Latitude:", value=float(def_lat), format="%.6f")
        with cx_lon: target_lon = st.number_input("Target Node Longitude:", value=float(def_lon), format="%.6f")
        
        target_r, target_c = rasterio.transform.rowcol(transform, target_lon, target_lat)
        
        if (0 <= target_r < orig_h) and (0 <= target_c < orig_w):
            manual_rsrp, manual_sinr = simulate_local_physics(target_r, target_c, (orig_h, orig_w), pixel_m, coverage_rad_m)
            
            disc_manual = ((yx[0] - target_r)**2 + (yx[1] - target_c)**2) <= coverage_rad_px**2
            newly_covered_manual = disc_manual & (pred_map == 1)
            manual_gain_pct = (newly_covered_manual.sum() / (pred_map.sum() + 1e-8)) * 100.0
            
            pop_newly = (pop_n * newly_covered_manual).sum()
            pop_total = (pop_n * pred_map).sum()
            manual_pop_gain = (pop_newly / (pop_total + 1e-8)) * 100.0
            
            col_m1, col_m2 = st.columns([3, 2])
            with col_m1:
                st.write("#### 🗺️ Live Target Site Signal Propagation Footprint")
                df_manual_site = pd.DataFrame([{"lat": target_lat, "lon": target_lon, "radius": coverage_rad_m}])
                
                coverage_footprint_layer = pdk.Layer(
                    "ScatterplotLayer", df_manual_site, get_position="[lon, lat]", get_radius="radius",
                    get_fill_color=[0, 255, 150, 60] if manual_rsrp > -90 else [255, 75, 75, 60], 
                    get_line_color=[0, 200, 100, 200] if manual_rsrp > -90 else [255, 0, 0, 200],
                    line_width_min_pixels=2,
                )
                node_mast_layer = pdk.Layer(
                    "ColumnLayer", df_manual_site, get_position="[lon, lat]", get_elevation=200, radius=25,
                    get_fill_color=[0, 255, 120, 255] if manual_rsrp > -90 else [255, 200, 0, 255], extruded=True
                )
                
                view_state_manual = pdk.ViewState(latitude=target_lat, longitude=target_lon, zoom=13, pitch=30)
                st.pydeck_chart(pdk.Deck(layers=[coverage_footprint_layer, node_mast_layer], initial_view_state=view_state_manual))
                
            with col_m2:
                st.write("#### 📊 Real-Time Simulation Telemetry Physics")
                pm1, pm2 = st.columns(2)
                with pm1: st.metric("Simulated Node RSRP", f"{manual_rsrp:.1f} dBm", delta="Excellent" if manual_rsrp > -85 else "Marginal")
                with pm2: st.metric("Simulated Node SINR", f"{manual_sinr:.1f} dB", delta="High Throughput" if manual_sinr > 12 else "Interference")
                
                st.write("---")
                st.write("#### 📈 Regional Performance Gains")
                st.metric("Total Regional Area Coverage Gain", f"{manual_gain_pct:.3f} %")
                st.metric("Inhabited Population Demand Satisfied", f"{manual_pop_gain:.3f} %")
        else:
            st.error("❌ Out-of-Bounds Error: Coordinates fall outside current raster extents.")
else:
    st.info("👈 Please upload all three foundational environment rasters in the main layout panel to initiate the site allocation search engine.")