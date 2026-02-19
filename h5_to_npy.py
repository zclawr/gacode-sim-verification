import h5py
import numpy as np
import torch

def h5_to_torch(file_path):
    print(f'Processing file: {file_path}')
    # getting keys from cfg
    input_keys = [
        "RLTS_3",
        "KAPPA_LOC",
        "ZETA_LOC",
        "TAUS_3",
        "VPAR_1",
        "Q_LOC",
        "RLNS_1",
        "TAUS_2",
        "Q_PRIME_LOC",
        "P_PRIME_LOC",
        "ZMAJ_LOC",
        "VPAR_SHEAR_1",
        "RLTS_2",
        "S_DELTA_LOC",
        "RLTS_1",
        "RMIN_LOC",
        "DRMAJDX_LOC",
        "AS_3",
        "RLNS_3",
        "DZMAJDX_LOC",
        "DELTA_LOC",
        "S_KAPPA_LOC",
        "ZEFF",
        "VEXB_SHEAR",
        "RMAJ_LOC",
        "AS_2",
        "RLNS_2",
        "S_ZETA_LOC",
        "BETAE_log10",
        "XNUE_log10",
        "DEBYE_log10"
    ]

    intermediate_target_keys = [
        "sumf"
    ]

    spectra_function_keys = [
        "ky"                     
    ]

    # creating lists to put the values
    input_list = []
    spectra_list = []
    intermediate_target_list = []

    with h5py.File(file_path, "r") as f:
        for key in input_keys:
            # Each of shape (N,)
            input_list.append(np.array(f[key]))
        for key in spectra_function_keys:
            spectra_list.append(np.array(f[key]))
        for key in intermediate_target_keys:
            flux_spectrum = np.array(f[key])    

            # assert flux_spectrum.shape[2] == 2 or flux_spectrum.shape[2] == 1, f"Unexpected shape at dim=2: {flux_spectrum.shape}"

            # Option 1: Select the first index at dim=2 (assuming it’s always the useful one)
            flux_spectrum = flux_spectrum[:, :, 0, :, :, :]  # (size, nky, nf, ns, 5)
            # 5 - channels ()
            # ns - num species (elec, ions)
            # nf - num fields? (should be 3 for cgyro, 1 for TGLF?) 

            # Option 2 (optional): Check if both are equal
            # assert np.allclose(flux_spectrum[:, :, 0], flux_spectrum[:, :, 1]), "Dim=2 entries differ"

            # Sum over nf
            summed_flux_spectrum = np.sum(flux_spectrum, axis=2)  # (size, nky, ns, 5)

            intermediate_target_list.append(summed_flux_spectrum)

    # Stack data and convert to tensors
    input_data = np.stack(input_list, axis=1)
    spectra_function_data = np.array(spectra_list[0])

    input_data_expanded = np.repeat(input_data[:, np.newaxis, :], spectra_function_data.shape[1], axis=1)
    spectra_function_data_expanded = spectra_function_data[:, :, np.newaxis]
    
    combined_matrix = np.concatenate((input_data_expanded, spectra_function_data_expanded), axis=2) # (size, nky, 32)

    flux_per_spicies_per_ky = torch.tensor(intermediate_target_list[0], dtype=torch.float32)

    # the sumf_tensor is of shape (size, nky, ns, 5)
    # now start converting to the interested fluxes: Ge,Qe,Qi,Pi, per wavenumber
    G_elec_per_ky = flux_per_spicies_per_ky[:, :, 0, 0]  # (size,nky,)
    Q_elec_per_ky = flux_per_spicies_per_ky[:, :, 0, 1]  # (size,nky,)
    Q_ions_per_ky = torch.sum(flux_per_spicies_per_ky[:, :, 1:, 1], dim=-1)  # (size,nky,)
    P_ions_per_ky = torch.sum(flux_per_spicies_per_ky[:, :, 1:, 2], dim=-1)  # (size,nky,)
    # cat them to be (size, nky, 4)
    target_flux_per_ky = torch.stack((G_elec_per_ky, Q_elec_per_ky, Q_ions_per_ky, P_ions_per_ky), dim=-1) # (size, nky, 4)

    return combined_matrix, target_flux_per_ky

def h5_to_npy(h5_path, input_path, output_path):
    input_npy, flux_tensor = h5_to_torch(h5_path)
    flux_npy = flux_tensor.cpu().detach().numpy()
    np.save(input_path, input_npy, allow_pickle=True)
    np.save(output_path, flux_npy, allow_pickle=True)
    print(f"Saved all H5 contents to {input_path} (inputs) and {output_path} (fluxes)")

# === Run (parallel) ===
if __name__ == "__main__":
    h5_to_npy("../../../data/lucas_work/tglf-sinn-data-subset/train/sampled_subset_40k2.h5", "reference_inputs.npy", "reference_fluxes.npy")