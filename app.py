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
# 1. Page Config & Core Layout Setup
# ==========================================
st.set_page_config(page_title="AI Site Planner", layout="wide", initial_sidebar_state="expanded")
st.title("🛰️ Deep RL & Vision Cellular Site Network Planning Engine")
st.write("---")

pipeline = ProductionInferencePipeline(patch_size=64)

MODEL_PATH = "models/unet_best.pth"
# 📋 PASTE YOUR COPIED DROPBOX LINK DIRECTLY HERE (ENSURE IT ENDS WITH dl=1)
DROPBOX_URL = "https://www.dropbox.com/scl/fi/abc123xyz/unet_best.pth?rlkey=xyz123&dl=1"

@st.cache_resource
def download_model_from_dropbox():
    os.makedirs("models", exist_ok=True)
    
    # Force flush any cached LFS metadata text pointers under 1MB
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

# Run download first
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
# 2. Interactive Sidebar Sliders
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
# 3. Local RF Physics Propagation Engine
# ==========================================
def simulate_local_physics(r, c, shape, pixel_m, radius_m):
    """Computes dynamic, relative path loss parameters at target indices."""
    h, w = shape
    yx = np.indices((h, w))
    
    # Calculate pure internal matrix distance offsets
    dist_px = np.sqrt((yx[0] - r)**2 + (yx[1] - c)**2)
    dist_m = dist_px * pixel_m
    dist_m = np.maximum(dist_m, pixel_m) # clip floor bound
    
    # COST-231 Hata empirical path loss model implementation
    simulated_rsrp = -50.0 - (44.9 - 6.55 * np.log10(30.0)) * np.log10(np.maximum(dist_m/1000.0, 0.001))
    simulated_rsrp = np.clip(simulated_rsrp, -130.0, -44.0)
    
    # SINR variation depending directly on target exclusion boundaries
    simulated_sinr = simulated_rsrp - (-95.0) - (dist_m / radius_m) * 12.0
    simulated_sinr = np.clip(simulated_sinr, -8.0, 28.0)
    
    return float(simulated_rsrp[r, c]), float(simulated_sinr[r, c])

# ==========================================
# 4. Raster Ingestion Data Loop
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
        
    st.success(f"Geospatial arrays processed! Canvas: {orig_h}x{orig_w} pixels at {pixel_m:.1f}m/px native resolution.")
    
    # Sequential Patch-Space Inference
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

    # Compile priority arrays
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

    # ==========================================
    # MODE A: AUTOMATED GREEDY SEARCH OPTIMIZATION
    # ==========================================
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
                    
                # Extract native file projected coordinates (e.g. Easting/Northing meters)
                native_lon, native_lat = rasterio.transform.xy(transform, r, c, offset="center")
                
                # 💥 CRUCIAL CRS CONVERSION: Translate local projection to WGS84 global coordinates
                longitudes, latitudes = warp_transform(meta['crs'], 'EPSG:4326', [native_lon], [native_lat])
                global_lon, global_lat = longitudes[0], latitudes[0]
                
                # Run relative math for signal telemetry calculations
                site_rsrp, site_sinr = simulate_local_physics(r, c, (orig_h, orig_w), pixel_m, coverage_rad_m)
                
                disc = ((yx[0] - r)**2 + (yx[1] - c)**2) <= coverage_rad_px**2
                newly_covered = disc & (pred_map == 1)
                gain_pct = (newly_covered.sum() / (pred_map.sum() + 1e-8)) * 100.0
                
                candidates.append({
                    "rank": step+1, 
                    "lat": global_lat, 
                    "lon": global_lon, 
                    "native_lat": round(native_lat, 1),
                    "native_lon": round(native_lon, 1),
                    "gain_pct": round(gain_pct, 2),
                    "rsrp": round(site_rsrp, 1), 
                    "sinr": round(site_sinr, 1)
                })
                
                placed_mask[r, c] = 1
                priority_work[disc] = 0.0

            df_candidates = pd.DataFrame(candidates)
        
        # ========================================================
        # 🗺️ VISUALIZATION MAP GENERATION: 3D TOWERS ON HEATMAP
        # ========================================================
        col1, col2 = st.columns([3, 2])
        with col1:
            st.write("#### 🗺️ Interactive 3D Blueprint over AI Suitability Heatmap")
            
            # 1. Convert the entire dense prob_map matrix into flat coordinates for the heatmap layer
            # To prevent performance lag on massive images, we downsample the background grid visualization slightly
            step_stride = max(1, int(max(orig_h, orig_w) / 150)) 
            
            heatmap_data = []
            for r in range(0, orig_h, step_stride):
                for c in range(0, orig_w, step_stride):
                    prob_val = float(prob_map[r, c])
                    if prob_val > 0.1:  # Only map pixels with a meaningful suitability score
                        # Translate current pixel to native meters, then to WGS84 degrees
                        n_lon, n_lat = rasterio.transform.xy(transform, r, c, offset="center")
                        g_lons, g_lats = warp_transform(meta['crs'], 'EPSG:4326', [n_lon], [n_lat])
                        heatmap_data.append({
                            "lon": g_lons[0],
                            "lat": g_lats[0],
                            "weight": prob_val
                        })
            
            df_heatmap = pd.DataFrame(heatmap_data)
            
            # 2. Configure Pydeck view viewport
            view_state = pdk.ViewState(
                latitude=df_candidates['lat'].mean(), 
                longitude=df_candidates['lon'].mean(), 
                zoom=12.5, 
                pitch=45
            )
            HIGH_CONTRAST_CMAP = [
                [48, 18, 59, 0],       # Low baseline (Transparent)
                [70, 74, 180, 40],     # Dark Indigo
                [67, 126, 249, 70],    # Blue
                [40, 174, 253, 95],    # Light Blue
                [24, 215, 203, 120],   # Turquoise
                [54, 244, 139, 145],   # Spring Green
                [118, 254, 78, 170],   # Bright Green
                [181, 242, 53, 190],   # Yellow-Green
                [230, 208, 51, 210],   # Pale Yellow
                [254, 156, 42, 225],   # Orange
                [247, 95, 23, 240],    # Dark Orange
                [220, 44, 5, 250],     # Orange-Red
                [179, 11, 2, 255],     # Deep Red
                [136, 1, 10, 255],     # Maroon
                [90, 0, 15, 255]       # Dark Crimson Peak
                ]
            # 3. Layer A: The Continuous AI Suitability Grid Layer (The Heatmap)
            suitability_heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                df_heatmap,
                get_position="[lon, lat]",
                get_weight="weight",
                radius_pixels=30,
                intensity=1.2,
                threshold=0.05,
                aggregation='"MEAN"',
                color_range=HIGH_CONTRAST_CMAP,
                color_domain=[0.01, 0.8])
            
            
            # 4. Layer B: The 3D Tower Masts
            df_candidates['color_r'] = np.where(df_candidates['rsrp'] > -85, 0, 240)
            df_candidates['color_g'] = np.where(df_candidates['rsrp'] > -85, 230, 80)
            
            tower_layer = pdk.Layer(
                "ColumnLayer", 
                df_candidates, 
                get_position="[lon, lat]",
                get_elevation=250, 
                radius=45, 
                get_fill_color="[color_r, color_g, 100, 240]", 
                pickable=True, 
                extruded=True
            )
            
            # 5. Render the multi-layer map canvas
            st.pydeck_chart(pdk.Deck(
                layers=[suitability_heatmap_layer, tower_layer], 
                initial_view_state=view_state, 
                tooltip={"text": "Rank: {rank}\nEst RSRP: {rsrp} dBm\nEst SINR: {sinr} dB"}
            ))
            
        with col2:
            st.write("#### 📈 Ranked Candidate Site Metrics")
            st.dataframe(
                df_candidates[["rank", "native_lat", "native_lon", "gain_pct", "rsrp", "sinr"]].rename(
                    columns={"native_lat": "Northing (m)", "native_lon": "Easting (m)", "gain_pct": "Area Gain %", "rsrp": "RSRP (dBm)", "sinr": "SINR (dB)"}
                ), use_container_width=True, hide_index=True
            )
            st.metric("Total Automated Allocations", f"{len(df_candidates)} Sites")

    # ==========================================
    # MODE B: MANUAL ENGINEERING PLACEMENT
    # ==========================================
    else:
        st.write("Input structural metric coordinates below to deploy a simulated node trace.")
        
        center_r, center_c = orig_h // 2, orig_w // 2
        def_lon, def_lat = rasterio.transform.xy(transform, center_r, center_c, offset="center")
        
        cx_lat, cx_lon = st.columns(2)
        with cx_lat: target_lat_m = st.number_input("Target Northing Metric Coordinate (m):", value=float(def_lat), format="%.2f")
        with cx_lon: target_lon_m = st.number_input("Target Easting Metric Coordinate (m):", value=float(def_lon), format="%.2f")
        
        target_r, target_c = rasterio.transform.rowcol(transform, target_lon_m, target_lat_m)
        
        if (0 <= target_r < orig_h) and (0 <= target_c < orig_w):
            # Translate specific target inputs to WGS84 for Pydeck positioning
            gl_lons, gl_lats = warp_transform(meta['crs'], 'EPSG:4326', [target_lon_m], [target_lat_m])
            m_lon, m_lat = gl_lons[0], gl_lats[0]
            
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
                df_manual_site = pd.DataFrame([{"lat": m_lat, "lon": m_lon, "radius": coverage_rad_m}])
                
                coverage_footprint_layer = pdk.Layer(
                    "ScatterplotLayer", df_manual_site, get_position="[lon, lat]", get_radius="radius",
                    get_fill_color=[40, 220, 130, 65] if manual_rsrp > -85 else [230, 70, 70, 65], 
                    get_line_color=[40, 200, 100, 200] if manual_rsrp > -85 else [210, 50, 50, 200],
                    line_width_min_pixels=2,
                )
                node_mast_layer = pdk.Layer(
                    "ColumnLayer", df_manual_site, get_position="[lon, lat]", get_elevation=300, radius=30,
                    get_fill_color=[40, 220, 130, 255] if manual_rsrp > -85 else [230, 180, 0, 255], extruded=True
                )
                
                view_state_manual = pdk.ViewState(latitude=m_lat, longitude=m_lon, zoom=13, pitch=30)
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
            st.error("❌ Out-of-Bounds Error: Specified target grid metrics lie outside the asset canvas matrix dimensions.")
else:
    st.info("👈 Please upload all three foundational environment rasters in the main layout panel to initiate the site allocation search engine.")
