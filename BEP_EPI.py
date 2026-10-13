# =================================================================
# GENERAL SET-UP
# =================================================================

# Import of necessary libraries from eduMRIsim
from simulator.model import load_model_from_nifti, Model
from simulator.scanlist import ScanItem
from scipy.ndimage import gaussian_filter

# Import other necessary libraries
import logging
import numpy as np
import matplotlib.pyplot as plt

# Suppress matplotlib debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Loading of digital patient (a.k.a. anatomical model)
digital_patient_path = "repository/models/MRSTAT_for_eduMRIsim_T2star_pseudo.nii"
digital_patient = load_model_from_nifti("Realistic Brain Model", digital_patient_path) 

# =================================================================
# CALCULATE SIGNAL INTENSITY FOR EPI-GRADIENT ECHO SEQUENCE
# =================================================================

scan_item_exercise_1 = ScanItem("exercise 1: T1-weighted spin echo")
"""
print("\n" + "="*50 + "\n") # print separator for better readability
print("Scan item for exercise 1 information:") # print to check your changes
print("="*50) # print separator for better readability
"""
scan_item_exercise_1.edit_parameters({"TE_ms": 50, "TR_ms": 400, "TI_ms": 0, "ReconMatrixSizeFE": 440, "ReconMatrixSizePE": 440})
scan_item_exercise_1.print()

import nibabel as nib
from nibabel.processing import resample_from_to
from simulator.model import new_load_model_from_nifti
from scipy.signal import windows

# visualise the ADC map using matplotlib
import matplotlib.pyplot as plt
import logging

# Suppress matplotlib debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# TODO: finish the calculate_SE_signal function that takes the digital patient and scan item as input and calculates the signal intensity for a spin echo sequence using the SE signal equation. The function should return the signal intensity map as a numpy array with the same shape as the tissue property maps. 

############################################################################################
# DATA PROCESSING PART
############################################################################################
#Matrix size
rows = 240
cols = 210

T1_map_path = "repository/models/BEP_practice_data/T1_map.nii"
T2_map_path = "repository/models/BEP_practice_data/T2_map.nii"
T2STAR_map_path = "repository/models/BEP_practice_data/T2s_map.nii"
PD_map_path = "repository/models/BEP_practice_data/PD_map.nii"
water_dixon_path = "repository/models/BEP_practice_data/water_dixon.nii"
fat_dixon_path = "repository/models/BEP_practice_data/fat_dixon.nii"
ADC_map_path = "repository/models/BEP_practice_data/ADC_map.nii"
B0_shimmed_path = "repository/models/BEP_practice_data/B0_shim_map.nii"
B0_unshimmed_path = "repository/models/BEP_practice_data/B0_unshimmed_map.nii"

model = new_load_model_from_nifti(
    "BEP practice scan",
    T1_file = T1_map_path,
    T2_file = T2_map_path,
    T2s_file = T2STAR_map_path,
    PD_file = PD_map_path,
    water_dixon_file = water_dixon_path,
    fat_dixon_file = fat_dixon_path,
    ADC_file = ADC_map_path,
    B0_shim_file = B0_shimmed_path,
    B0_unshimmed_file = B0_unshimmed_path
)
PD = model.PDmap
T1 = model.T1map_msec
T2 = model.T2map_msec
T2s = model.T2smap_msec
dB0 = model.B0map_unshimmed_Hz 


PDw = PD[:, :, :, 0]
T1w = T1[:, :, :, 0]
T2w = T2[:, :, :, 0]
T2sw = T2s[:, :, :, 0]
dB0w = dB0[:, : ,: ,0]
PDf = PD[:, :, :, 1]
T1f = T1[:, :, :, 1]
T2f = T2[:, :, :, 1]
T2sf = T2s[:, :, :, 1]
dB0f = dB0[:, :, :, 1]
#Without water and fat separation
PD = PD[:, :, :, 0]
T1 = T1[:, :, :, 0]
T2 = T2[:, :, :, 0]
T2s = T2s[:, :, :, 0]
dB0 = dB0[:, : ,: ,0]

def get_slice(volume):

    volume = np.squeeze(volume)

    z = volume.shape[2] // 2

    return volume[:, :, z]

PD = get_slice(PD)
T2s = get_slice(T2s)
T2 = get_slice(T2)
PDf = get_slice(PDf)
PDw = get_slice(PDw)
T2sw = get_slice(T2sw)
T2w = get_slice(T2w)
T2sf = get_slice(T2sf)
T2f = get_slice(T2f)
dB0 = get_slice(dB0)
dB0f = get_slice(dB0f)
dB0w = get_slice(dB0w)
plt.imshow(PD, cmap="gray")
plt.title("PD map")
plt.axis("off")
plt.show()
plt.imshow(T2s, cmap="gray")
plt.title("T2s map")
plt.axis("off")
plt.show()
plt.imshow(dB0, cmap="gray")
plt.title("B0 map")
plt.axis("off")
plt.show()


EPI_T2s = r"C:\BEP2047896\eduMRIsim_v1\repository\models\BEP_scanner_data\T2s_map.nii"
EPI_shim = r"C:\BEP2047896\eduMRIsim_v1\centered.nii\shiftishim.nii"
EPI_unshim = r"C:\B0_shim_Hz_after_EPI\shift_down.nii"
EPI_PD = r"C:\BEP2047896\eduMRIsim_v1\repository\models\BEP_scanner_data\PD_map.nii"
EPI_noshim = r"C:\Users\20234417\Downloads\20260617_scan_data\te_20260617_182453_20_1_EPInoshimV4.nii"
EPI_wdixon = r"C:\BEP2047896\eduMRIsim_v1\repository\models\BEP_scanner_data\water_dixon.nii"
EPI_fdixon = r"C:\BEP2047896\eduMRIsim_v1\repository\models\BEP_scanner_data\fat_dixon.nii"

import nibabel as nib
import matplotlib.pyplot as plt

# Load the NIfTI file

# Get the image data as a NumPy array
"""
data = img.get_fdata()
data1 = img1.get_fdata()
data2 = img2.get_fdata()
data3 = img3.get_fdata()
print(f"Scan shape:{data.shape}")
print(f"Scan shape:{data1.shape}")
print(f"Scan shape:{data2.shape}")
print(f"Scan shape:{data3.shape}")
data = data[:,:,100]
scan_T2s = np.squeeze(data)
# Check the dimensions
print(f"Scan shape:{data.shape}")
"""
def get_slice_scan(map_slice, slice_index=135):
    return map_slice[:,:,slice_index]
import nibabel as nib

# Load the NIfTI files
EPI_T2s_data = nib.load(EPI_T2s).get_fdata()
EPI_shim_data = nib.load(EPI_shim).get_fdata()
EPI_unshim_data = nib.load(EPI_unshim).get_fdata()
EPI_PD_data = nib.load(EPI_PD).get_fdata()
EPI_NOSHIM_epi = nib.load(EPI_noshim).get_fdata()
EPI_NOSHIM_epi = np.squeeze(EPI_NOSHIM_epi)
EPI_wdixon_data = nib.load(EPI_wdixon).get_fdata()
EPI_fdixon_data = nib.load(EPI_fdixon).get_fdata()
# Check shapes
print(EPI_T2s_data.shape)
print(EPI_shim_data.shape)
print(EPI_unshim_data.shape)
print(EPI_PD_data.shape)
print(EPI_NOSHIM_epi.shape)
# Extract middle slices
T2s_scan = get_slice_scan(EPI_T2s_data)
shim_scan = EPI_shim_data[:,:,114]
unshim_scan = EPI_unshim_data[:,:,129]
PD_scan = get_slice_scan(EPI_PD_data)
wdixon_scan = get_slice_scan(EPI_wdixon_data)
fdixon_scan = get_slice_scan(EPI_fdixon_data)
# Display the middle slice
import numpy as np
import matplotlib.pyplot as plt

def rotate_and_show(slice_2d, title="", show=True):
    """
    Rotate a 2D slice 90° clockwise and optionally display it.
    """

    rotated = np.rot90(slice_2d, k=1)

    if show:
        plt.imshow(rotated, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()

    return rotated
"""
Data from scan session
"""
rotate_T2s = rotate_and_show(T2s_scan, "T2s")
rotate_PD = rotate_and_show(PD_scan, "PD")
rotate_shim = rotate_and_show(shim_scan, "Shim")
rotate_unshim = np.rot90(unshim_scan, k=2)
rotate_wdixon = rotate_and_show(wdixon_scan, "wdixon")
rotate_fdixon = rotate_and_show(fdixon_scan, "fdixon")


###############################################################
# Digital phantom creation
###############################################################
"""
    Different maps applied in Phantom
    Gaussian map
    """
rows = 240
cols = 210
#Gaussian filter matrix
sigma = 30# Adjust this to change how wide the central hotspot is
A = 200

# Generate coordinate grids for rows and columns
y_indices = np.arange(rows)
x_indices = np.arange(cols)
x_grid, y_grid = np.meshgrid(x_indices, y_indices)

# Calculate centers
y0 = (rows - 1) / 2.0
x0 = (cols - 1) / 2.0

# Compute centered Gaussian formula
# Peak value in the exact middle will be 1.0
gaussian_matrix = A*np.exp(-((x_grid - x0)**2 + (y_grid - y0)**2) / (2 * sigma**2))
inv_gaussianmatrix = -gaussian_matrix
print("Matrix Shape:", gaussian_matrix.shape)
print("Value at exact center (119, 104):", gaussian_matrix[119, 104])

plt.figure(figsize=(6,6))
plt.imshow(gaussian_matrix, cmap='seismic')
plt.colorbar(label='Hz')
plt.title('Artificial B0 map gaussian')
plt.show()

BW_PE = 30

dy = gaussian_matrix / BW_PE
print("mean dy")
print(np.mean(dy))
print("max dy")
print(dy.max())    
plt.figure(figsize=(6,6))
plt.imshow(dy)
plt.colorbar(label='pixels')
plt.title('Predicted displacement')
plt.show()

"""
Artificial Phantom
"""

# -----------------------
# Image size
# -----------------------


x = np.linspace(-1, 1, cols)
y = np.linspace(-1, 1, rows)
X, Y = np.meshgrid(x,y)

# -----------------------
# Main circular object
# -----------------------
radius = 0.6
circle = (X**2 + Y**2) <= radius**2

# -----------------------
# Initialize maps
# -----------------------
phantom_PD = np.zeros((rows, cols))
phantom_T2s = np.zeros((rows, cols))
phantom_B0 = np.zeros((rows, cols))
# -----------------------
# Base of phantom
# -----------------------
fat_region = (X**2 + Y**2) <= 0.4**2
water_region = circle & (~fat_region)

"""
Parameters to test B0 inhomogeneity without chemical shift artifact
"""
phantom_PD[water_region] = 200
phantom_PD[fat_region] = 100

# Base T2* values (ms)
phantom_T2s[water_region] = 60
phantom_T2s[fat_region] = 40
phantom_B0 = 50
# -----------------------
# Add inner circles with different T2*
# -----------------------
inner_centers = [
    (0.2, 0.2),
    (-0.3, 0.1),
    (0.0, -0.3),
]

inner_radii = [0.12, 0.10, 0.08]

t2_values = [30, 60, 90]  # different T2* contrasts
pd_values = [200, 80, 150]

for (cx, cy), r, t2, pd in zip(inner_centers, inner_radii, t2_values, pd_values):
    mask = ((X - cx)**2 + (Y - cy)**2) <= r**2
    
    phantom_PD[mask] = pd
    phantom_T2s[mask] = t2

# -----------------------
# Display
# -----------------------

plt.figure(figsize=(6,5))
plt.imshow(phantom_PD, cmap="gray")
plt.title("Phantom PD map")
plt.colorbar()
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(phantom_T2s, cmap="magma")
plt.title("Phantom T2* map")
plt.colorbar()
plt.show()
# Test chemical shift


# Create coordinate grid
x = np.linspace(-1, 1, cols)
y = np.linspace(-1, 1, rows)
X, Y = np.meshgrid( x,y)

# Circular mask (object)
radius = 0.8
circle = (X**2 + Y**2) <= radius**2

# Initialize matrix
phantom = np.zeros((rows, cols))

# Define inner fat core + outer water ring
fat_radius = 0.4

fat_region = (X**2 + Y**2) <= fat_radius**2
water_region = circle & (~fat_region)

# Assign intensities
"""
phantom[water_region] = 100   # water signal
phantom[fat_region] = 300     # fat signal (slightly lower)
"""
x = np.linspace(-1,1, cols)
y = np.linspace(-1,1,rows)
X,Y = np.meshgrid(x,y)
outer_radius = 0.7
inner_radius=0.35
water_regioni = (X**2 + Y**2 <= outer_radius**2) & (X**2 + Y**2 > inner_radius**2)
fat_regioni = (X**2 + Y**2 <= inner_radius**2)
phantom_PD_water = np.zeros((rows, cols))
phantom_PD_fat = np.zeros((rows, cols))
phantom_T2s_fat = np.zeros((rows, cols))
phantom_T2s_water = np.zeros((rows,cols))
phantom_B0 = np.zeros((rows, cols))


"""
Parameters to test Chemical shift
"""

# Water-region (outer - ring)
phantom_PD_water[water_regioni] = 200    
phantom_T2s_water[water_regioni] = 60           # T2* of water in ms

# Vet-regio (binnenste kern)
phantom_PD_fat[fat_regioni] = 100    
phantom_T2s_fat[fat_regioni] = 40             # T2* of fat in ms

# Constant B0 inhomogeneity
phantom_B0[water_regioni | fat_regioni] = 50 
"""
matrix_size = 64
row_start = rows//2 - matrix_size//2
row_stop  = rows//2 + matrix_size//2
"""

x = np.linspace(-1, 1, cols)
y = np.linspace(-1, 1, rows)
X, Y = np.meshgrid(x, y)
linear_gradient = 0
quadratic_drop = 1/ (X**2 + Y**2)

dB0i = (linear_gradient + quadratic_drop)
b0_g = np.zeros((rows, cols), dtype=float)
rows = 240
cols = 210
dB0s = 50
rowhalf = rows //  2
dB0u = 200
col_indices = np.arange(cols)
for y in range(rows):
    # Calculate the field strength for this specific row
    k = dB0s * 0.025 * (y-rowhalf)
    
    # Assign this value 'k' to the entire row (all columns at row y)
    b0_g[y, :] = k
plt.figure(figsize=(6,5))
plt.imshow(b0_g, cmap="bwr")
plt.title("Compressing B0 map")
plt.colorbar()
plt.show()

###############################################################
# Calculation of SINGLE SHOT GE-EPI
###############################################################

def calculate_GE_EPI_signal(digital_patient : Model, scan_item : ScanItem) -> np.ndarray:
    rows = 240
    cols = 210
    
    signal_csaw = np.zeros_like(phantom)
    signal_csaf = np.zeros_like(phantom)    
    final_k_space = np.zeros((rows, cols), dtype=complex)
    final_k_spacewo = np.zeros((rows, cols), dtype=complex)
    final_k_spacep = np.zeros((rows, cols), dtype = complex)
    final_k_spacecs = np.zeros((rows, cols), dtype = complex)
    matrix_sizepe = 151
    deviatie_size = matrix_sizepe // 2
    row_start = rows//2 - matrix_sizepe//2
    row_stop  = rows//2 + matrix_sizepe//2
    matrix_sizefe = 110
    col_start = cols//2 - matrix_sizefe//2
    col_stop = cols//2 + matrix_sizefe//2

    for ky in range(int(row_start), int(row_stop)):
        ESP = 0.83
        C = 5
        t = ky*ESP + C
        t_sec = t/1000
        
   
        """
        These formulas calculate separate for water and fat signal
        """
        signal_arrayw = rotate_wdixon*np.exp(-np.abs(t_sec)/np.maximum(rotate_T2s, 1e-9))*np.exp(-rotate_unshim*2j*np.pi*t_sec)
        signal_arrayf= rotate_fdixon*np.exp(-np.abs(t_sec)/np.maximum(rotate_T2s, 1e-9))*np.exp(-(-440 + rotate_unshim)*2j*np.pi*t_sec)
        """
        Here the water and fat signal get added together
        """
        signal_arraycs = signal_arrayf + signal_arrayw
      

        """
        Calculate the signal for B0 inhomogeneities
        """
        signal_array = rotate_PD*np.exp(-np.abs(t_sec)/np.maximum(rotate_T2s, 1e-9))*np.exp((rotate_shim*2j*np.pi*t_sec))

        
        """
        Calculates signal only including T2* decay
        """
        signal_arraywo = rotate_PD*np.exp(-np.abs(t_sec)/np.maximum(rotate_T2s, 1e-9))


        k_full = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(signal_array)
  
            )
        )
        k_fullwo = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(signal_arraywo)
            )
        )
        k_fullcs = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(signal_arraycs)
            )
        )

        final_k_space[ky,:] = k_full[ky,:].ravel()
        final_k_spacewo[ky, :] = k_fullwo[ky, :].ravel()
        final_k_spacecs[ky, :] = k_fullcs[ky, :].ravel()
    
    return final_k_space, final_k_spacewo, final_k_spacecs


final_k_space, final_k_spacewo, final_k_spacecs = calculate_GE_EPI_signal(digital_patient, scan_item_exercise_1 )

y_size, x_size = final_k_spacewo.shape  # 240, 210
data_size = 151

# Create the Tukey window for the actual data portion
taper_window_inner = windows.tukey(data_size, alpha=0.4667)

# Create a full-size window of zeros (size 240)
full_window = np.zeros(y_size)

#  Center the Tukey window inside the full 240-sized array
start_idx = (y_size - data_size) // 2  
end_idx = start_idx + data_size        
full_window[start_idx:end_idx] = taper_window_inner

# Reshape full_window to (240, 1) so it broadcasts correctly across all 210 columns
final_k_space *= full_window[:, np.newaxis]
final_k_spacewo *= full_window[:, np.newaxis]
final_k_spacecs *= full_window[:, np.newaxis]
Final_k_space_wa = np.array(final_k_space)
reconstructed_image = np.fft.fftshift(
    np.fft.ifft2(
        np.fft.ifftshift(Final_k_space_wa)
    )
)
Final_k_space_wo = np.array(final_k_spacewo)
reconstructed_imagewo = np.fft.fftshift(
    np.fft.ifft2(
        np.fft.ifftshift(Final_k_space_wo)
    )
)
Final_k_space_cs = np.array(final_k_spacecs)
reconstructed_imagecs = np.fft.fftshift(
    np.fft.ifft2(
        np.fft.ifftshift(Final_k_space_cs)
    )
)
plt.figure(figsize=(6,6))
plt.imshow(np.abs(reconstructed_image), cmap="gray")
plt.title("Reconstructed Image EPI with B0 inhomogeneity from code")
plt.axis("off")
plt.show()
# Plot the image
plt.figure(figsize=(6,6))
plt.imshow(np.abs(reconstructed_imagewo), cmap="gray")
plt.title("Reconstructed Image EPI without gaussian B0 inhomogeneity - 128 ky lines ESP 0.6 ms ")
plt.axis("off")
plt.show()
plt.figure(figsize=(6,6))
plt.imshow(np.abs(reconstructed_imagecs), cmap="gray")
plt.title("Reconstructed Image EPI with Chemical Shift Artifact - without T2* decay 0.55 ms and 92 ky")
plt.axis("off")
plt.show()

####################################################
# Creation digital phantom map  Nyquist ghosting
####################################################


PD = digital_patient.PDmap
T1 = digital_patient.T1map_msec
T2 = digital_patient.T2map_msec
B0 = digital_patient.B0map_shim_Hz


rows = 240
cols = 210
#Gaussische filter matrix

sigma = 30# Adjust this to change how wide the central hotspot is

# Generate coordinate grids for rows and columns
y_indices = np.arange(rows)
x_indices = np.arange(cols)
x_grid, y_grid = np.meshgrid(x_indices, y_indices)

# Calculate centers
y0 = (rows - 1) / 2.0
x0 = (cols - 1) / 2.0
A = 200
# Compute centered Gaussian formula
# Peak value in the exact middle will be 1.0
gaussian_matrix = A*np.exp(-((x_grid - x0)**2 + (y_grid - y0)**2) / (2 * sigma**2))

plt.figure(figsize=(6,6))
plt.imshow(gaussian_matrix, cmap='seismic')
plt.colorbar(label='Hz')
plt.title('Artificial B0 map')
plt.show()

BW_PE = 30

dy = gaussian_matrix / BW_PE
print("mean dy")
print(np.mean(dy))
print("max dy")
print(dy.max())    
plt.figure(figsize=(6,6))
plt.imshow(dy)
plt.colorbar(label='pixels')
plt.title('Predicted displacement')
plt.show()

# -----------------------
# Image size
# -----------------------


x = np.linspace(-1, 1, cols)
y = np.linspace(-1, 1, rows)
X, Y = np.meshgrid(x, y)

# -----------------------
# Main circular object
# -----------------------
radius = 0.5
circle = (X**2 + Y**2) <= radius**2

# -----------------------
# Initialize maps
# -----------------------
phantom_PD = np.zeros((rows, cols))
phantom_T2s = np.zeros((rows, cols))
phantom_B0 = np.zeros((rows, cols))
# -----------------------
# Base of the phantom
# -----------------------
fat_region = (X**2 + Y**2) <= 0.3**2
water_region = circle & (~fat_region)

phantom_PD[water_region] = 200
phantom_PD[fat_region] = 100



# Base T2* values (ms)
phantom_T2s[water_region] = 60
phantom_T2s[fat_region] = 40

# -----------------------
# Add inner circles with different T2*
# -----------------------
inner_centers = [
    (0.2, 0.2),
    (-0.3, 0.1),
    (0.0, -0.3),
]

inner_radii = [0.12, 0.10, 0.08]

t2_values = [30, 60, 90]  # different T2* contrasts
pd_values = [200, 80, 150]

for (cx, cy), r, t2, pd in zip(inner_centers, inner_radii, t2_values, pd_values):
    mask = ((X - cx)**2 + (Y - cy)**2) <= r**2
    
    phantom_PD[mask] = pd
    phantom_T2s[mask] = t2

# -----------------------
# Display
# -----------------------

plt.figure(figsize=(6,5))
plt.imshow(phantom_PD, cmap="gray")
plt.title("Phantom PD map")
plt.colorbar()
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(phantom_T2s, cmap="magma")
plt.title("Phantom T2* map")
plt.colorbar()
plt.show()
# Test chemical shift

# Create coordinate grid
x = np.linspace(-1, 1, cols)
y = np.linspace(-1, 1, rows)
X, Y = np.meshgrid(x, y)

# Circular mask (object)
radius = 0.4
circle = (X**2 + Y**2) <= radius**2

# Initialize matrix
phantom = np.zeros((rows, cols))

# Define inner fat core + outer water ring
fat_radius = 0.4

fat_region = (X**2 + Y**2) <= fat_radius**2
water_region = circle & (~fat_region)


max_val = np.max(np.abs(gaussian_matrix))
im = plt.imshow(gaussian_matrix, cmap='seismic', vmin = -max_val, vmax=max_val)
plt.title("dB0 map")
#dB0 = 10

"""
Artificial B0 map
"""
x = np.linspace(-1, 1, cols)
y = np.linspace(-1, 1, rows)
X, Y = np.meshgrid(x, y)
R2 = X**2 + (Y*(rows/cols))**2
max_B0 = 20
dB0i = max_B0 * R2
max_vali = np.max(np.abs(dB0i))
plt.imshow(dB0i, cmap='seismic', vmin = -max_vali, vmax=max_vali)
plt.colorbar()
plt.title("B0 map")

############################################################
# CALCULATION OF SINGLE SHOT GE-EPI WITH NYQUIST GHOSTING
############################################################

def calculate_Nyquist_ghost_signal(digital_patient : Model, scan_item : ScanItem) -> np.ndarray:

    """
    Input Scan parameters
    """
    matrix_size = 128
    ESP = 0.6 # in ms
    C = 5 # Time between RF excitation pulse and echo train
    row_start = rows//2 - matrix_size//2
    row_stop  = rows//2 + matrix_size//2
    final_k_spaceg = np.zeros((rows, cols), dtype=complex) # final k-space matrix

    for ky in range(int(row_start), int(row_stop)):
        
        ESP_sec = ESP/1000 # in sec
        t = ky*ESP + C# in ms
        t_sec = t/1000 # in sec
        """
        Calculates Phase error
        """
        ghost_phase = 2*np.pi* rotate_unshim * ESP_sec * 20
        if ky % 2 == 0:
            # Forward readout direction
            sign = -1
            phase_error = ghost_phase
        else:
            # Reverse readout direction 
            sign = 1
            phase_error = ghost_phase

        """
        Calculates Signal including Nyquist Ghosting
        """

        signal_array = rotate_PD*np.exp(-(np.abs(t)/np.maximum(rotate_T2s, 1e-9)))*np.exp(-(rotate_unshim*2j*np.pi*t_sec))*np.exp(1j*sign*phase_error)
    
        k_full = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(signal_array)
            )
        )
        i = ky -2
    # stack only this acquired line
        final_k_spaceg[i+2:i+3, :] = k_full[ky,:].ravel()

    return final_k_spaceg

final_k_spaceg = calculate_Nyquist_ghost_signal(digital_patient, scan_item_exercise_1 )

y_size, x_size = final_k_spaceg.shape  # 240, 210
data_size = 128

# Create the Tukey window for the actual data portion
taper_window_inner = windows.tukey(data_size, alpha=0.4667)

# Create a full-size window of zeros (size 240)
full_window = np.zeros(y_size)

# Center the Tukey window inside the full 240-sized array
start_idx = (y_size - data_size) // 2  # (240 - 128) // 2 = 56
end_idx = start_idx + data_size        # 56 + 128 = 184
full_window[start_idx:end_idx] = taper_window_inner

# Reshape full_window to (240, 1) so it broadcasts correctly across all 210 columns
final_k_spaceg *= full_window[:, np.newaxis]

Final_k_space_g = np.array(final_k_spaceg)
reconstructed_imageg = np.fft.fftshift(
    np.fft.ifft2(
        np.fft.ifftshift(Final_k_space_g)
    )
)
plt.figure(figsize=(6,6))
plt.imshow(np.abs(reconstructed_imageg), cmap="gray")
plt.title("Nyquist ghosting- ESP = 0.6ms and 128 ky lines, 10 time intensivation")
plt.axis("off")
plt.show()