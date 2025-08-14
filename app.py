"""
India District Poverty Mapping — Single-file Streamlit App
(folium + robust NumPy export + strict EE project init)

Env / Secrets required:
  - GCP_SA_KEY      : base64(key.json) or full JSON (one line / triple-quoted)
  - GEE_PROJECT_ID  : your Cloud project that has Earth Engine enabled
"""

import os, json, base64, re
from datetime import datetime
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# --- streamlit-folium (graceful install) ---
HAVE_ST_FOLIUM = True
try:
    from streamlit_folium import st_folium
except Exception:
    HAVE_ST_FOLIUM = False
    try:
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit-folium"])
        from streamlit_folium import st_folium
        HAVE_ST_FOLIUM = True
    except Exception:
        pass

import folium
import ee
import geemap

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ---------- UI ----------
st.set_page_config(page_title="India Poverty Mapping — GEE + DL", page_icon="🗺️", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.metric-card { background: #f6f8fc; border-radius: 12px; padding: 10px 14px; }
</style>
""", unsafe_allow_html=True)
st.title("🗺️ India District Poverty Mapping — GEE + CNN / U-Net / Hybrid")
st.caption("Click any district in India to fetch features from GEE and train three models on a local patch.")

# ---------- Auth helpers ----------
def _read_sa_json_from_secret(var="GCP_SA_KEY") -> dict:
    raw = None
    try:
        raw = st.secrets.get(var, None)
    except Exception:
        pass
    if raw is None:
        raw = os.environ.get(var, None)
    if raw in (None, ""):
        raise RuntimeError("Missing service account key. Set GCP_SA_KEY (base64 of key.json or raw JSON).")

    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    s = str(raw).strip()

    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                pat = r'("private_key"\s*:\s*")(?P<key>.*?)(\")'
                m = re.search(pat, s, flags=re.DOTALL)
                if m:
                    key_block = m.group("key")
                    key_fixed = key_block.replace("\r\n", "\n").replace("\n", r"\n")
                    s_fixed = s[:m.start("key")] + key_fixed + s[m.end("key"):]
                    return json.loads(s_fixed)
            except Exception:
                pass
            return json.loads(s.replace("\\n", "\n"))

    try:
        decoded = base64.b64decode(s, validate=False).decode("utf-8", errors="ignore").strip()
        if decoded.startswith("{") and decoded.endswith("}"):
            return json.loads(decoded)
    except Exception:
        pass
    return json.loads(s.replace("\\n", "\n"))

def ee_init_from_secret():
    key_info = _read_sa_json_from_secret()
    from google.oauth2 import service_account
    project = os.environ.get("GEE_PROJECT_ID", "").strip()
    if not project:
        st.error("GEE_PROJECT_ID is not set. Set it to a Cloud project that has Earth Engine enabled.")
        st.stop()

    scopes = [
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/devstorage.full_control",
    ]
    creds = service_account.Credentials.from_service_account_info(key_info, scopes=scopes)

    # Strict project init (no fallback)
    try:
        ee.Initialize(credentials=creds, project=project)
        _ = ee.Image(1).reduceRegion(ee.Reducer.mean(), ee.Geometry.Point([0,0]).buffer(10000), 1000).getInfo()
    except Exception as e:
        st.error(
            f"Earth Engine failed to initialize with project '{project}'.\n\n"
            "Fix in Google Cloud/Earth Engine:\n"
            "• Enable EE API on the project\n"
            "• EE Code Editor → Settings → Cloud Projects → USE this project\n"
            "• IAM: grant your Service Account (this key) at least 'Editor' (or EE roles)\n"
            "• EE Settings → Service Accounts: add your SA email\n\n"
            f"Original error:\n{e}"
        )
        st.stop()

try:
    ee.Number(1).getInfo()
except Exception:
    ee_init_from_secret()

# ---------- Map helpers ----------
def add_ee_image_to_folium(m: folium.Map, ee_image: ee.Image, vis_params: dict, name: str):
    map_id = ee_image.getMapId(vis_params or {})
    folium.raster_layers.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
    ).add_to(m)

# --- NEW: ultra-robust exporter ---
def ee_to_numpy_robust(image: ee.Image, region: ee.Geometry, base_scale: int, crs: str) -> np.ndarray:
    """
    1) Try geemap.ee_to_numpy at (scale, 2x, 4x)
    2) If that fails, do band-wise sampleRectangle after a common reproject.
    """
    last_err = None
    for mul in [1, 2, 4]:
        try:
            return geemap.ee_to_numpy(image, region=region, scale=int(base_scale*mul), crs=crs)
        except Exception as e:
            last_err = e

    # Fallback: band-wise sampleRectangle (more tolerant)
    try:
        bands: List[str] = image.bandNames().getInfo()
        out = None
        for i, b in enumerate(bands):
            band_img = image.select([b]).reproject(crs=crs, scale=int(base_scale*4))
            data = band_img.sampleRectangle(region=region, defaultValue=0).getInfo()
            arr2d = np.array(data[b], dtype=np.float32)
            if out is None:
                h, w = arr2d.shape
                out = np.zeros((h, w, len(bands)), dtype=np.float32)
            out[:, :, i] = arr2d
        if out is None:
            raise RuntimeError("Empty sampleRectangle response.")
        return out
    except Exception as e2:
        raise RuntimeError(f"Failed to convert EE image to numpy. Last error: {last_err}\nFallback error: {e2}")

# ---------- GEE Processor ----------
class GEEProcessor:
    def __init__(self, project_id: Optional[str] = None):
        try:
            ee.Number(1).getInfo()
        except Exception:
            ee_init_from_secret()
        self.project_id = project_id or os.environ.get("GEE_PROJECT_ID", "")
        self.study_area: Optional[ee.Geometry] = None
        self.features: Dict[str, ee.Image] = {}
        self.adm1_name: Optional[str] = None
        self.adm2_name: Optional[str] = None

    def set_study_area_from_point(self, lat: float, lon: float) -> Optional[ee.Geometry]:
        fc = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2").filter(
            ee.Filter.eq('ADM0_NAME', 'India')
        )
        point = ee.Geometry.Point([lon, lat])
        feat = fc.filterBounds(point).first()
        info = feat.getInfo() if feat else None
        if not info:
            return None
        props = info.get("properties", {})
        self.adm1_name = props.get("ADM1_NAME", "Unknown")
        self.adm2_name = props.get("ADM2_NAME", "Unknown")
        self.study_area = ee.Feature(info).geometry()
        return self.study_area

    # --- features ---
    def get_modis_ndvi(self, start='2020-01-01', end='2021-01-01') -> Optional[ee.Image]:
        try:
            col = ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(start, end).filterBounds(self.study_area)
            ndvi = col.select('NDVI').median().multiply(0.0001).rename('MODIS_NDVI').clip(self.study_area)
            self.features['MODIS_NDVI'] = ndvi;  return ndvi
        except Exception:
            return None

    def get_sentinel2_ndbi(self, start='2020-01-01', end='2021-01-01') -> Optional[ee.Image]:
        try:
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate(start, end).filterBounds(self.study_area) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            med = s2.median().clip(self.study_area)
            nir = med.select('B8'); swir = med.select('B11')
            ndbi = swir.subtract(nir).divide(swir.add(nir)).rename('NDBI')
            self.features['NDBI'] = ndbi;  return ndbi
        except Exception:
            return None

    def get_worldcover(self, year='2020') -> Optional[ee.Image]:
        try:
            lc = ee.ImageCollection('ESA/WorldCover/v100') \
                .filterDate(f'{year}-01-01', f'{year}-12-31').first() \
                .select('Map').rename('LandCover').clip(self.study_area)
            built_up = lc.eq(50).rename('Built_up').uint8()
            water = lc.eq(80).rename('Water_Cover').uint8()
            self.features['LandCover'] = lc
            self.features['Built_up'] = built_up
            self.features['Water_Cover'] = water
            return lc
        except Exception:
            return None

    def get_worldpop(self, year=2020) -> Optional[ee.Image]:
        try:
            pop = ee.Image(f'WorldPop/GP/100m/pop/IND_{year}').rename('PopDensity').clip(self.study_area)
            self.features['PopDensity'] = pop;  return pop
        except Exception:
            return None

    def get_viirs_annual(self, year=2020) -> Optional[ee.Image]:
        try:
            ntl = ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V21') \
                .filterDate(f'{year}-01-01', f'{year}-12-31').first() \
                .select('avg_rad').rename('NTL').clip(self.study_area)
            self.features['NTL'] = ntl;  return ntl
        except Exception:
            return None

    # --- composites ---
    def _minmax(self, img: ee.Image, band: str, newname: str, scale: int = 100) -> Optional[ee.Image]:
        try:
            stats = img.select([band]).reduceRegion(ee.Reducer.minMax(), self.study_area, scale=scale, maxPixels=1e9)
            minv = ee.Number(stats.get(f'{band}_min')); maxv = ee.Number(stats.get(f'{band}_max'))
            return img.select([band]).subtract(minv).divide(maxv.subtract(minv)).rename(newname)
        except Exception:
            return None

    def build_composites(self, scale=100) -> None:
        pop_norm = self._minmax(self.features['PopDensity'], 'PopDensity', 'Pop_Normalized', scale)
        ntl_norm = self._minmax(self.features['NTL'], 'NTL', 'NTL_Normalized', scale)
        ndvi_norm = self._minmax(self.features['MODIS_NDVI'], 'MODIS_NDVI', 'NDVI_Norm', scale)
        ndbi_norm = self._minmax(self.features['NDBI'], 'NDBI', 'NDBI_Norm', scale)

        self.features['Pop_Normalized'] = pop_norm
        self.features['NTL_Normalized'] = ntl_norm
        self.features['NDVI_Norm'] = ndvi_norm
        self.features['NDBI_Norm'] = ndbi_norm

        urban = ndbi_norm.multiply(0.4).add(ntl_norm.multiply(0.3)) \
            .add(self.features['Built_up'].toFloat().multiply(0.3)).rename('Urban_Index')
        self.features['Urban_Index'] = urban

        rural = ndvi_norm.multiply(0.5) \
            .add(ee.Image(1).subtract(ndbi_norm).multiply(0.3)) \
            .add(ee.Image(1).subtract(ntl_norm).multiply(0.2)).rename('Rural_Index')
        self.features['Rural_Index'] = rural

        infra = self.features['Built_up'].toFloat().multiply(0.4) \
            .add(ntl_norm.multiply(0.3)).add(pop_norm.multiply(0.3)).rename('Infrastructure_Index')
        self.features['Infrastructure_Index'] = infra

        poverty = ee.Image(1).subtract(ndvi_norm.multiply(0.30)) \
            .subtract(ntl_norm.multiply(0.40)).add(pop_norm.multiply(0.30)).rename('Poverty_Index')
        self.features['Poverty_Index'] = poverty

    def process_features_for_area(self) -> bool:
        if self.study_area is None: return False
        ok = True
        ok &= self.get_modis_ndvi() is not None
        ok &= self.get_sentinel2_ndbi() is not None
        ok &= self.get_worldcover() is not None
        ok &= self.get_worldpop() is not None
        ok &= self.get_viirs_annual() is not None
        if not ok: return False
        self.build_composites(scale=100)
        return True

    def get_feature_stack(self) -> Optional[ee.Image]:
        names = [
            'MODIS_NDVI','NDBI','LandCover','Water_Cover','Built_up',
            'Pop_Normalized','NTL_Normalized','Urban_Index','Rural_Index','Infrastructure_Index'
        ]
        try:
            imgs = [self.features[n] for n in names]
            ref = self.features['PopDensity'].projection()
            aligned = [im.resample('bilinear').reproject(crs=ref) for im in imgs]
            return ee.Image.cat(aligned).rename(names)
        except Exception:
            return None

    def numpy_patch_from_point(
        self, lat: float, lon: float, size_km: float = 12, scale: int = 100, num_classes: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build region, then try (scale, 2×, 4×) and (size_km, 0.75×, 0.5×) combinations.
        """
        pt = ee.Geometry.Point([lon, lat])

        stack = self.get_feature_stack()
        if stack is None:
            raise RuntimeError("Feature stack not ready")

        # try shrinking region if needed
        for km_factor in [1.0, 0.75, 0.5]:
            size_try = size_km * km_factor
            region = pt.buffer(size_try * 1000 / 2.0).bounds().intersection(self.study_area, maxError=1)
            # robust export
            try:
                X = ee_to_numpy_robust(stack, region=region, base_scale=scale, crs='EPSG:4326')
                X = np.nan_to_num(np.array(X, dtype=np.float32), nan=0.0)
                pov = ee_to_numpy_robust(self.features['Poverty_Index'], region=region, base_scale=scale, crs='EPSG:4326')
                yf = np.nan_to_num(np.array(pov, dtype=np.float32), nan=0.0).squeeze()

                flat = yf.flatten()
                qs = np.quantile(flat[~np.isnan(flat)], np.linspace(0, 1, num_classes + 1))
                y = np.digitize(yf, qs[1:-1], right=False).astype(np.int32)
                return X, y
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"Patch export failed even after retries: {last_error}")

# ---------- Models ----------
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    return x

def build_cnn_segmentation(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(24, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(48, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(96, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(48, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(24, 3, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(x)
    return models.Model(inputs, outputs, name="CNN")

def build_unet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    c1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128); p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256); p4 = layers.MaxPooling2D()(c4)
    bn = conv_block(p4, 384)
    u1 = layers.UpSampling2D()(bn); u1 = layers.Concatenate()([u1, c4]); u1 = conv_block(u1, 256)
    u2 = layers.UpSampling2D()(u1); u2 = layers.Concatenate()([u2, c3]); u2 = conv_block(u2, 128)
    u3 = layers.UpSampling2D()(u2); u3 = layers.Concatenate()([u3, c2]); u3 = conv_block(u3, 64)
    u4 = layers.UpSampling2D()(u3); u4 = layers.Concatenate()([u4, c1]); u4 = conv_block(u4, 32)
    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(u4)
    return models.Model(inputs, outputs, name="UNet")

def aspp(x, filters=256):
    d1 = layers.Conv2D(filters, 1, padding="same", activation="relu")(x)
    d2 = layers.Conv2D(filters, 3, dilation_rate=2, padding="same", activation="relu")(x)
    d3 = layers.Conv2D(filters, 3, dilation_rate=4, padding="same", activation="relu")(x)
    d4 = layers.AveragePooling2D(pool_size=(x.shape[1], x.shape[2]))(x)
    d4 = layers.Conv2D(filters, 1, padding="same", activation="relu")(d4)
    d4 = layers.UpSampling2D(size=(x.shape[1], x.shape[2]))(d4)
    y = layers.Concatenate()([d1, d2, d3, d4])
    y = layers.Conv2D(filters, 1, padding="same", activation="relu")(y)
    return y

def squeeze_excitation(x, r=16):
    f = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(f // r, 4), activation="relu")(s)
    s = layers.Dense(f, activation="sigmoid")(s)
    s = layers.Reshape((1,1,f))(s)
    return layers.Multiply()([x, s])

def build_hybrid_cnn_unet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    c1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128); p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256); p4 = layers.MaxPooling2D()(c4)
    bn = conv_block(p4, 512)
    bn = aspp(bn, 256); bn = squeeze_excitation(bn, 8)
    u1 = layers.UpSampling2D()(bn); u1 = layers.Concatenate()([u1, c4]); u1 = conv_block(u1, 256)
    u2 = layers.UpSampling2D()(u1); u2 = layers.Concatenate()([u2, c3]); u2 = conv_block(u2, 128)
    u3 = layers.UpSampling2D()(u2); u3 = layers.Concatenate()([u3, c2]); u3 = conv_block(u3, 64)
    u4 = layers.UpSampling2D()(u3); u4 = layers.Concatenate()([u4, c1]); u4 = conv_block(u4, 32)
    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(u4)
    return models.Model(inputs, outputs, name="HybridCNNUNet")

def compute_metrics(y_true, y_pred, num_classes):
    yt = y_true.reshape(-1); yp = y_pred.reshape(-1)
    acc  = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, average="macro", zero_division=0)
    rec  = recall_score(yt, yp, average="macro", zero_division=0)
    f1   = f1_score(yt, yp, average="macro", zero_division=0)
    return acc, prec, rec, f1

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    num_classes = st.selectbox("Classes (Poverty_Index quantiles)", [2,3,4,5], index=1)
    patch_size  = st.selectbox("Patch size (px)", [32, 64, 96, 128], index=1)
    stride      = st.selectbox("Stride (px)", [16, 32, 48, 64], index=1)
    epochs      = st.slider("Base epochs", 1, 20, 5)
    batch_size  = st.selectbox("Batch size", [2, 4, 8, 16], index=2)
    size_km     = st.slider("Patch span around click (km)", 6, 30, 12)
    st.info("Click a district on the map to run the pipeline.")

# ---------- Map ----------
clicked = None
if HAVE_ST_FOLIUM:
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB positron")
    adm2 = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2").filter(
        ee.Filter.eq('ADM0_NAME', 'India')
    )
    style_params = dict(color='444444', width=1, fillColor='00000000')
    try:
        adm2_vis = adm2.style(**style_params)
    except Exception:
        outline = ee.Image().paint(adm2, 1, 1).visualize(palette=['444444'])
        fill    = ee.Image().paint(adm2, 1).visualize(palette=['000000'], opacity=0.0)
        adm2_vis = ee.ImageCollection([fill, outline]).mosaic()
    add_ee_image_to_folium(m, adm2_vis, {}, "Districts (ADM2)")
    st_map = st_folium(m, height=600, width=None, returned_objects=["last_clicked"])
    if st_map and st_map.get("last_clicked"):
        clicked = (st_map["last_clicked"]["lat"], st_map["last_clicked"]["lng"])
else:
    st.warning("`streamlit-folium` not installed; using manual coordinate input.")
    c1, c2 = st.columns(2)
    with c1:
        lat_in = st.number_input("Latitude", value=20.2961, format="%.6f")
    with c2:
        lon_in = st.number_input("Longitude", value=85.8245, format="%.6f")
    if st.button("Use these coordinates"):
        clicked = (float(lat_in), float(lon_in))

# ---------- Patching / training helpers ----------
def make_patches(X, Y, size, stride):
    H, W, C = X.shape
    xs, ys = [], []
    for i in range(0, H - size + 1, stride):
        for j in range(0, W - size + 1, stride):
            xp = X[i:i+size, j:j+size, :]
            yp = Y[i:i+size, j:j+size]
            if xp.shape[0] == size and xp.shape[1] == size:
                xs.append(xp); ys.append(yp)
    if not xs:
        return np.empty((0,size,size,X.shape[2]), dtype=np.float32), np.empty((0,size,size), dtype=np.int32)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.int32)

def train_and_eval(model_fn, Xtr, Ytr, Xte, Yte, epochs, batch_size, num_classes):
    model = model_fn(Xtr.shape[1:], num_classes)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, Ytr, epochs=epochs, batch_size=batch_size, verbose=0)
    Yhat = np.argmax(model.predict(Xte, verbose=0), axis=-1)
    return compute_metrics(Yte, Yhat, num_classes), Yhat

# ---------- Pipeline ----------
if clicked:
    lat, lon = clicked
    processor = GEEProcessor()
    area = processor.set_study_area_from_point(lat, lon)
    if area is None:
        st.error("Click inside India to select a district."); st.stop()
    st.success(f"Selected: **{processor.adm2_name}**, {processor.adm1_name}")

    with st.spinner("Fetching features from GEE..."):
        ok = processor.process_features_for_area()
        if not ok:
            st.error("Failed to process features from GEE."); st.stop()

        # --- robust patch export (now with retries over region size) ---
        X_full, y_full = processor.numpy_patch_from_point(
            lat=lat, lon=lon, size_km=size_km, scale=100, num_classes=num_classes
        )
        st.write("**Feature stack shape**:", X_full.shape, " | **Label shape**:", y_full.shape)

        # Normalize continuous channels ~[0,1]
        cont_idx = [0,1,5,6,7,8,9]
        Xn = X_full.copy()
        for k in cont_idx:
            v = Xn[:,:,k]
            vmin, vmax = np.percentile(v, 1), np.percentile(v, 99)
            Xn[:,:,k] = 0.0 if vmax<=vmin else np.clip((v - vmin)/(vmax - vmin), 0, 1)

    Xp, Yp = make_patches(Xn, y_full, size=patch_size, stride=stride)
    if len(Xp) < 8:
        st.warning("Patch area too small — increase patch span (km) or reduce stride.")
        st.stop()

    n = len(Xp); idx = np.random.permutation(n)
    tr_end, te_end = int(0.7*n), int(0.9*n)
    tr_idx, va_idx, te_idx = idx[:tr_end], idx[tr_end:te_end], idx[te_end:]
    Xtr, Ytr = Xp[tr_idx], Yp[tr_idx]
    Xva, Yva = Xp[va_idx], Yp[va_idx]
    Xte, Yte = Xp[te_idx], Yp[te_idx]

    st.subheader("🤖 Training Models")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("Training **CNN (Baseline)** ...")
        (acc_cnn, prec_cnn, rec_cnn, f1_cnn), _ = train_and_eval(
            build_cnn_segmentation, Xtr, Ytr, Xte, Yte, epochs, batch_size, num_classes
        )
    with col2:
        st.info("Training **U-Net** ...")
        (acc_unet, prec_unet, rec_unet, f1_unet), _ = train_and_eval(
            build_unet, Xtr, Ytr, Xte, Yte, epochs+1, batch_size, num_classes
        )
    with col3:
        st.info("Training **Hybrid CNN+U-Net** ...")
        (acc_hyb, prec_hyb, rec_hyb, f1_hyb), _ = train_and_eval(
            build_hybrid_cnn_unet, Xtr, Ytr, Xte, Yte, epochs+2, batch_size, num_classes
        )

    st.subheader("📊 Metrics")
    df = pd.DataFrame([
        {"Model":"CNN (Baseline)",       "Accuracy":acc_cnn, "Precision":prec_cnn, "Recall":rec_cnn, "F1":f1_cnn},
        {"Model":"U-Net",                "Accuracy":acc_unet,"Precision":prec_unet,"Recall":rec_unet,"F1":f1_unet},
        {"Model":"Hybrid CNN+U-Net",     "Accuracy":acc_hyb, "Precision":prec_hyb, "Recall":rec_hyb, "F1":f1_hyb},
    ])
    st.dataframe(df.style.format({c:"{:.4f}" for c in ["Accuracy","Precision","Recall","F1"]}), use_container_width=True)
    melted = df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig = px.bar(melted, x="Model", y="Value", color="Metric", barmode="group", title="Model Performance")
    fig.update_layout(height=400); st.plotly_chart(fig, use_container_width=True)
    best_name = df.loc[df["F1"].idxmax(), "Model"]
    st.success(f"🥇 Best (by F1): **{best_name}**")

    csv = df.assign(
        District=processor.adm2_name, State=processor.adm1_name,
        Timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ).to_csv(index=False)
    st.download_button("📥 Download metrics CSV", data=csv,
                       file_name=f"metrics_{processor.adm2_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                       mime="text/csv")
else:
    st.info("Click anywhere on the map within India (or enter coordinates if prompted) to select a district.")