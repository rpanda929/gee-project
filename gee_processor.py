"""
Google Earth Engine Data Processing Module
Handles all GEE operations for poverty mapping features
"""

import ee
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import streamlit as st

class GEEProcessor:
    """Handles Google Earth Engine data processing for poverty mapping"""
    
    def __init__(self, project_id: str = None):
        """Initialize GEE with authentication"""
        self.project_id = project_id
        self.study_area = None
        self.features = {}


    def authenticate_and_initialize(self) -> bool:
    try:
        # If already initialized, a trivial server call will succeed
        ee.Number(1).getInfo()
        return True
    except Exception:
        pass

    # 1) Service Account (recommended for Streamlit deployments)
    try:
        sa_email = None
        sa_key = None
        gee_project = self.project_id
        if hasattr(st, "secrets"):
            sa_email = st.secrets.get("gee-project@weighty-time-440511-h3.iam.gserviceaccount.com")
            sa_key = st.secrets.get("{
  "type": "service_account",
  "project_id": "weighty-time-440511-h3",
  "private_key_id": "af579377ec14a4daed11b7c0c828ac588ec2af93",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCnq1i3HDsTjlf+\nATXiqv3t4cWDOg42I7+GxI8xdkx2+qC1YAxpoxH5IFJuPyXXDngY63j5kFfFENjh\nxjwV403UJFUlpDQqMjX9wsMy+k/Rbf6b+GFeTF9VWeSAIUqTw5n5sMvMm2qA6LWl\nIpbO46WVJ8SDfqf6N1MhAnqTOheq4p9G1s9dlAvDk97XlI97EQkZn+vn1XTo8hxj\noHe+h+eUNTgx4Ur247r/lFU0OznuWfsQq8+0goNHUTUuoAU3utpzAsocfRIpKFnD\nN0ZsBr0J9JeMTSJu7YN5DfUmspCW8tQsLXzwzvSD4DEo8rlO8/FEZOiELsoB8hov\nYeiQm9OvAgMBAAECggEAE8kK63Sn3ftmk8El0C0YUNla+sy/f5EBWVBZi18nz8Rs\ncWq3nEGTbd/sCjlmaDW7Y0lqGZz4VndR+HZxga20ceulpo71XuNU7rBsY1gZUh/W\nMyaAquV9PG+ioKINEFm9EjNUIT1XuIV9ZdKqlBhV4j9yl4e7H7Imm5cUysnIoC/z\n5Ik3PybZPMrKiFAiqCdyVc7gfwgEnhK99Gd4nDQ82W0612bZXMVp0Q7VozGqwi8Y\njxRuyiCTrhkWh7sBvlY3aurqLH1ANtnuUMrQKUagkX1VutwRMpVsTRUa4viDb6KY\nsuUzsvdRRxheT+OtRKLfrXwGUbRN6HzMCOcVmMRXHQKBgQDYvOruQ1NKnRCpsGOm\nEjnBi7Z/aXz1Ytgzi7BmR9zD4iqZ5fn0qkGdpWxkpi4HfMe9jTiLk5v/M7Xa7N+n\n0S+9p0XYFvQomL4OYhhPirhSYAvutb01/0hueOQ8/lTKOcm31D6ov3M543sR6b1e\n3RDl6FeMsMDtDxRyigJiNru90wKBgQDGCucR7vfMu1MubgZ92WMF5zdJtKI6xDTj\nKVs843c3Tavek+EWp8XDbfaHAYD5teIxIJKaKRkJfcqitUuYhvgHUwSX3peJeuTE\n4CYAECuUyHdCQ0t7VeuHSaM5Vq4q/cYjl9/H0siDWQ3EWtVAN5aZil9tR66PA+Px\nkwfAv8z9NQKBgQCypz6ruXk9pqwmg4wQRzOuc0CyU7y59IksK/fyx0eVe6cMBoJ8\nB59gmAv6BvUoHNX0TCURAFJ1ESXU2K4fAZJtrSUoUvtdP1JiPr+1SS5YUG1ljg7y\nJISK80GHeUlhDDNXQS+JH46WaAl5IYeEW4rjSBpqkQPmJCkBpPsEBAPLGwKBgQDC\n2sNTz84csEs9qZ/Vf2iAzGiHqqTcMWTgBTbyB9Sqo09xpgqX4echTDZ9yyr9hsnR\nEH1uFPW/cvdHdB23K0Uq37HrQ2XSLQqd8vUwprhaoYtFtTS1W2psKDXjGrgvMJYd\nOzdBQtFq+toi0kRz9L3GwHD38sl7iZZjgAjmy1CD6QKBgCJFqiI40n3wNVFKwdOb\ndzX5dGLbxnug2zzccaR4ScfjkeqmNjEp1qZQKib4m+rIHmKn1PT4e8p/6plNIa2c\nOplT+c8TyUsBcN1fbwkwWEffgWDhhAQkyzAHGCTjDubEW2AU8KtBoCcWGpS9JHjB\nOpOY3DukJPu7qxS6iBmK1Uwm\n-----END PRIVATE KEY-----\n",
  "client_email": "gee-project@weighty-time-440511-h3.iam.gserviceaccount.com",
  "client_id": "105577984323234023130",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/gee-project%40weighty-time-440511-h3.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
")  # contents of JSON key
            gee_project = gee_project or st.secrets.get("weighty-time-440511-h3")

        if sa_email and sa_key:
            credentials = ee.ServiceAccountCredentials(sa_email, key_data=sa_key)
            ee.Initialize(credentials, project=gee_project)
            return True
    except Exception as e:
        st.warning(f"Service account init failed: {e}")

    # 2) User OAuth (good for local development)
    try:
        ee.Authenticate()  # interactive flow
        ee.Initialize(project=self.project_id)
        return True
    except Exception as e:
        st.error(
            "GEE Authentication failed. Make sure your Cloud project is "
            "registered for Earth Engine and the API is enabled."
        )
        st.exception(e)
        return False

     def define_study_area(self, district_names: List[str] = None) -> ee.Geometry:
        """Define study area for the three districts in Odisha"""
        if district_names is None:
            district_names = ['Koraput', 'Rayagada', 'Malkangiri']
        
        try:
            admin2 = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
            districts = (admin2
                        .filter(ee.Filter.eq('ADM1_NAME', 'Orissa'))
                        .filter(ee.Filter.inList('ADM2_NAME', district_names)))
            self.study_area = districts.geometry()
            return self.study_area
        except Exception as e:
            st.error(f"Error defining study area: {str(e)}")
            return None
    
    def get_modis_ndvi(self, start_date: str = '2020-01-01', 
                       end_date: str = '2021-01-01') -> ee.Image:
        """Get MODIS NDVI data"""
        try:
            modis = (ee.ImageCollection('MODIS/061/MOD13Q1')
                    .filterDate(start_date, end_date)
                    .filterBounds(self.study_area))
            
            modis_ndvi = (modis.select('NDVI')
                         .median()
                         .multiply(0.0001)  # Scale factor for MODIS NDVI
                         .rename('MODIS_NDVI')
                         .clip(self.study_area))
            
            self.features['MODIS_NDVI'] = modis_ndvi
            return modis_ndvi
        except Exception as e:
            st.error(f"Error getting MODIS NDVI: {str(e)}")
            return None
    
    def get_sentinel2_ndbi(self, start_date: str = '2020-01-01', 
                          end_date: str = '2021-01-01') -> ee.Image:
        """Get Sentinel-2 NDBI (Normalized Difference Built-up Index)"""
        try:
            s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterDate(start_date, end_date)
                 .filterBounds(self.study_area)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
            
            s2_med = s2.median().clip(self.study_area)
            nir = s2_med.select('B8')   # NIR band
            swir = s2_med.select('B11') # SWIR band
            
            ndbi = (swir.subtract(nir)).divide(swir.add(nir)).rename('NDBI')
            self.features['NDBI'] = ndbi
            return ndbi
        except Exception as e:
            st.error(f"Error getting Sentinel-2 NDBI: {str(e)}")
            return None
    
    def get_worldcover_landcover(self, year: str = '2020') -> ee.Image:
        """Get ESA WorldCover land cover data"""
        try:
            landcover = (ee.ImageCollection('ESA/WorldCover/v100')
                        .filterDate(f'{year}-01-01', f'{year}-12-31')
                        .first()
                        .clip(self.study_area)
                        .select('Map')
                        .rename('LandCover'))
            
            # Create binary masks for specific land cover types
            built_up = landcover.eq(50).rename('Built_up').uint8()  # Built-up areas
            water = landcover.eq(80).rename('Water').uint8()        # Water bodies
            
            self.features['LandCover'] = landcover
            self.features['Built_up'] = built_up
            self.features['Water'] = water
            return landcover
        except Exception as e:
            st.error(f"Error getting WorldCover data: {str(e)}")
            return None
    
    def get_population_density(self, year: int = 2020) -> ee.Image:
        """Get WorldPop population density data"""
        try:
            pop_density = (ee.Image(f'WorldPop/GP/100m/pop/IND_{year}')
                          .clip(self.study_area)
                          .rename('PopDensity'))
            
            self.features['PopDensity'] = pop_density
            return pop_density
        except Exception as e:
            st.error(f"Error getting population density: {str(e)}")
            return None
    
    def get_nighttime_lights(self, year: int = 2020) -> ee.Image:
        """Get VIIRS nighttime lights data"""
        try:
            ntl = (ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V21')
                  .filterDate(f'{year}-01-01', f'{year}-12-31')
                  .first()
                  .select('average')
                  .clip(self.study_area)
                  .rename('NTL'))
            
            self.features['NTL'] = ntl
            return ntl
        except Exception as e:
            st.error(f"Error getting nighttime lights: {str(e)}")
            return None
    
    def minmax_normalize(self, image: ee.Image, band_name: str, 
                        new_name: str, scale: int = 100) -> ee.Image:
        """Min-max normalize an image"""
        try:
            stats = image.select([band_name]).reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=self.study_area,
                scale=scale,
                maxPixels=1e9
            )
            
            min_val = stats.get(f'{band_name}_min')
            max_val = stats.get(f'{band_name}_max')
            
            normalized = (image.select([band_name])
                         .subtract(ee.Number(min_val))
                         .divide(ee.Number(max_val).subtract(ee.Number(min_val)))
                         .rename(new_name))
            
            return normalized
        except Exception as e:
            st.error(f"Error normalizing {band_name}: {str(e)}")
            return None
    
    def create_poverty_index(self) -> ee.Image:
        """Create composite poverty index from all features"""
        try:
            # Normalize key features
            pop_norm = self.minmax_normalize(
                self.features['PopDensity'], 'PopDensity', 'Pop_Normalized'
            )
            ndvi_norm = self.minmax_normalize(
                self.features['MODIS_NDVI'], 'MODIS_NDVI', 'NDVI_Norm'
            )
            ntl_norm = self.minmax_normalize(
                self.features['NTL'], 'NTL', 'NTL_Norm'
            )
            
            # Create composite poverty index
            # Higher values indicate higher poverty likelihood
            poverty_index = (ee.Image(1)
                           .subtract(ndvi_norm.multiply(0.3))    # Low vegetation = higher poverty
                           .subtract(ntl_norm.multiply(0.4))     # Low lights = higher poverty
                           .add(pop_norm.multiply(0.3))          # High density can indicate poverty
                           .rename('Poverty_Index'))
            
            self.features['Poverty_Index'] = poverty_index
            return poverty_index
        except Exception as e:
            st.error(f"Error creating poverty index: {str(e)}")
            return None
    
    def align_and_stack_features(self, target_scale: int = 100) -> ee.Image:
        """Align all features to same projection and stack them"""
        try:
            # Use population density as reference projection
            target_proj = self.features['PopDensity'].projection()
            
            # Align all features to target projection
            aligned_features = []
            feature_names = []
            
            for name, image in self.features.items():
                if image is not None:
                    aligned = image.resample('bilinear').reproject(
                        crs=target_proj,
                        scale=target_scale
                    )
                    aligned_features.append(aligned)
                    feature_names.append(name)
            
            # Stack all features
            stacked = ee.Image.cat(aligned_features)
            return stacked
        except Exception as e:
            st.error(f"Error stacking features: {str(e)}")
            return None
    
    def export_to_drive(self, image: ee.Image, description: str, 
                       folder: str = 'GEE_Poverty_Mapping') -> ee.batch.Task:
        """Export image to Google Drive"""
        try:
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=description,
                folder=folder,
                scale=100,
                region=self.study_area,
                maxPixels=1e9,
                crs='EPSG:4326'
            )
            task.start()
            return task
        except Exception as e:
            st.error(f"Error exporting {description}: {str(e)}")
            return None
    
    def get_feature_info(self) -> Dict:
        """Get information about processed features"""
        info = {}
        for name, image in self.features.items():
            if image is not None:
                try:
                    # Get basic info
                    proj_info = image.projection().getInfo()
                    band_names = image.bandNames().getInfo()
                    info[name] = {
                        'bands': band_names,
                        'projection': proj_info['crs'],
                        'scale': proj_info['transform'][0] if 'transform' in proj_info else 'Unknown'
                    }
                except:
                    info[name] = {'status': 'Error getting info'}
        return info
    
    def process_all_features(self) -> bool:
        """Process all features for poverty mapping"""
        try:
            # Define study area
            if self.define_study_area() is None:
                return False
            
            # Get all features
            st.info("Processing MODIS NDVI...")
            self.get_modis_ndvi()
            
            st.info("Processing Sentinel-2 NDBI...")
            self.get_sentinel2_ndbi()
            
            st.info("Processing WorldCover land cover...")
            self.get_worldcover_landcover()
            
            st.info("Processing population density...")
            self.get_population_density()
            
            st.info("Processing nighttime lights...")
            self.get_nighttime_lights()
            
            st.info("Creating poverty index...")
            self.create_poverty_index()
            
            st.success("All features processed successfully!")
            return True
        except Exception as e:
            st.error(f"Error processing features: {str(e)}")
            return False

   
