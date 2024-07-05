require(raster)

# Path to your raster file
raster_path <- "/Users/tnye/bayarea_path/files/vs30/California_vs30_Wills15_hybrid.tif"

# List of coordinates (longitude, latitude)
coordinates <- data.frame(x = c(-122,-122.5), y = c(36.5,37))

# Load the raster
raster_data <- raster(raster_path)

# Extract values at specified coordinates
values <- extract(raster_data, coordinates)

# Print the extracted values
print(values)
