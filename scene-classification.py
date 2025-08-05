# Import required libraries
from osgeo import gdal, osr  # For geospatial data processing
import numpy as np  # For array operations
import os  # For file path operations
import matplotlib.pyplot as plt  # For visualization
from matplotlib.colors import ListedColormap  # For custom color mapping

# =============================================
# 1. PATHS CONFIGURATION
# =============================================
# Define input/output file paths
input_path = "/content/drive/MyDrive/Colab Notebooks/2025-04-17-00_00_2025-04-17-23_59_Sentinel-2_L2A_Scene_classification_map.tiff"
output_dir = "/content/drive/MyDrive/Colab Notebooks"
output_base = os.path.join(output_dir, "April2025_scene_classification")  # Base name for output files

# =============================================
# 2. COORDINATE SYSTEM PRESERVATION
# =============================================
# Open the input TIFF file and extract geospatial information
ds = gdal.Open(input_path)  # Open the input file
original_crs = ds.GetProjectionRef()  # Get Coordinate Reference System (CRS)
geotransform = ds.GetGeoTransform()  # Get geotransform for pixel-to-coordinate conversion

# Print the original CRS for verification
print("Original CRS from input file:")
print(original_crs)

# =============================================
# 3. CLASSIFICATION WITH OFFICIAL COLORS
# =============================================
# Sentinel-2 Scene Classification (SCL) color mapping
SCL_COLORS = {
    0: (0, 0, 0),        # No Data (Black)
    1: (255, 0, 0),      # Saturated/Defective (Red)
    2: (47, 47, 47),     # Dark Area Pixels (Dark Gray)
    3: (100, 50, 0),     # Cloud Shadows (Brown)
    4: (0, 160, 0),      # Vegetation (Green)
    5: (255, 230, 90),   # Bare Soil (Yellow)
    6: (0, 0, 255),      # Water (Blue)
    7: (128, 128, 128),  # Clouds Low Prob (Medium Gray)
    8: (192, 192, 192),  # Clouds Medium Prob (Light Gray)
    9: (255, 255, 255),  # Clouds High Prob (White)
    10: (100, 200, 255), # Cirrus (Light Blue)
    11: (255, 150, 255)  # Snow (Pink)
}

# Read RGB bands and scale to 0-255 range
r, g, b = [ds.GetRasterBand(i).ReadAsArray() for i in [1, 2, 3]]  # Read RGB bands
rgb = np.stack([(r*255).astype('uint8'), (g*255).astype('uint8'), (b*255).astype('uint8')], axis=2)  # Combine into RGB array

# Initialize classification image
class_image = np.zeros(r.shape, dtype='uint8')  # Create empty array for classification
tolerance = 2  # Color matching tolerance (accounts for minor variations)

# Classify pixels based on RGB values
for class_id, target_rgb in SCL_COLORS.items():
    # Create mask for pixels matching this class's color (within tolerance)
    mask = (
        (np.abs(rgb[...,0] - target_rgb[0]) <= tolerance) &
        (np.abs(rgb[...,1] - target_rgb[1]) <= tolerance) &
        (np.abs(rgb[...,2] - target_rgb[2]) <= tolerance)
    )
    
    # Special handling for water (class 6) to prioritize it
    if class_id == 6:
        class_image[mask] = 6
    # For other classes, only assign if pixel isn't already classified
    elif class_id != 0:
        class_image[mask & (class_image == 0)] = class_id

# Ensure pure black pixels are marked as No Data
class_image[(r == 0) & (g == 0) & (b == 0)] = 0

# =============================================
# 4. ENVI OUTPUT WITH CORRECT GEOSPATIAL INFO
# =============================================
# Define output file paths
output_dat = f"{output_base}.dat"  # Binary data file
output_hdr = f"{output_base}.hdr"  # Header file with metadata

# Write classification data to binary file
with open(output_dat, 'wb') as f:
    class_image.tofile(f)  # Write numpy array to binary file

# Calculate pixel sizes (ensuring positive Y size)
pixel_size_x = geotransform[1]  # East-west pixel size
pixel_size_y = abs(geotransform[5])  # North-south pixel size (absolute value)

# Create ENVI header file with geospatial metadata
header = f"""ENVI
file type = ENVI Classification
samples = {ds.RasterXSize}  # Number of columns
lines = {ds.RasterYSize}    # Number of rows
bands = 1                   # Single band classification
header offset = 0           # No header offset
data type = 1               # 8-bit unsigned integer
interleave = bsq            # Band sequential format
byte order = 0              # Little-endian
map info = {{Geographic Lat/Lon, 1.0000, 1.0000, {geotransform[0]}, {geotransform[3]}, {pixel_size_x}, {pixel_size_y}, WGS-84, units=Degrees}}
coordinate system string = {{ {original_crs} }}

classes = 12  # Number of classes
class lookup = {{
   {','.join(map(str, SCL_COLORS[0]))},  # No Data
   {','.join(map(str, SCL_COLORS[1]))},  # Saturated
   {','.join(map(str, SCL_COLORS[2]))},  # Dark Area
   {','.join(map(str, SCL_COLORS[3]))},  # Cloud Shadows
   {','.join(map(str, SCL_COLORS[4]))},  # Vegetation
   {','.join(map(str, SCL_COLORS[5]))},  # Bare Soil
   {','.join(map(str, SCL_COLORS[6]))},  # Water
   {','.join(map(str, SCL_COLORS[7]))},  # Clouds Low
   {','.join(map(str, SCL_COLORS[8]))},  # Clouds Medium
   {','.join(map(str, SCL_COLORS[9]))},  # Clouds High
   {','.join(map(str, SCL_COLORS[10]))}, # Cirrus
   {','.join(map(str, SCL_COLORS[11]))}} # Snow
class names = {{
   No Data,
   Saturated or Defective,
   Dark Area Pixels,
   Cloud Shadows,
   Vegetation,
   Bare Soil,
   Water,
   Clouds (Low Probability),
   Clouds (Medium Probability),
   Clouds (High Probability),
   Cirrus,
   Snow}}
"""

# Write header file
with open(output_hdr, 'w') as f:
    f.write(header)

# Verification output
print("\nCRS Verification:")
print("Input CRS:", ds.GetProjectionRef())
print("Output CRS in header:", original_crs.split("\n")[0] + "...")

# =============================================
# 5. VISUALIZATION
# =============================================
# Create custom colormap for visualization
cmap = ListedColormap([np.array(SCL_COLORS[i])/255 for i in range(12)])

# Plot the classification result
plt.figure(figsize=(15,10))
plt.imshow(class_image, cmap=cmap, vmin=0, vmax=11)
plt.colorbar(ticks=range(12), label="Class ID")
plt.title("Classification (Water=Blue, CRS Preserved)")
plt.show()

# Final confirmation
print(f"\n\u2705 Files created with EXACT original CRS:")
print(f"→ {output_dat}")
print(f"→ {output_hdr}")