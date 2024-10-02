import numpy as np
import matplotlib.pyplot as plt

file_path = r"C:\Users\Utente\Clearbox_projects\PRISMS\dataset_management\mimic_cxr_iv_multimodal\dataset_ABIDE\data\extracted_abide\1 - barrow\ABIDEII-BNI_1\29006\session_1\anat_1\anat.nii.npy"

# Load the .npy file
data = np.load(file_path)

## Get the middle slice of the 3D data
#middle_slice = data.shape[2] // 2
#slice_data = data[:, :, middle_slice]

# Select the middle slice along the first dimension

print(data.shape[1])

middle_slice = data.shape[1] // 2
slice_data = data[middle_slice, :, :]

# Display the slice using the 'viridis' colormap
plt.imshow(slice_data, cmap='viridis')
plt.colorbar()  # Add a colorbar to the plot
plt.show()