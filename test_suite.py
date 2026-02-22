import numpy as np
import os
import subprocess
import shutil
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

from format_tglf import process_tglf_batches
from h5_to_cgyro_input import convert_h5_to_batch_dir_parallel
from generate_ky_spectra import save_h5, compute_ky_matrix_skip_bad
from h5_to_npy import h5_to_npy
from parse_fluxes import process_all_batches

def create_test_inputs(physical_conditions, h5_path, tglf_dest_dir, reformat_dir):
    os.makedirs(tglf_dest_dir, exist_ok=True)
    os.makedirs(reformat_dir, exist_ok=True)

    # your grad_r0 (should be same one from cfg)
    grad_r0 = 1.23314445670738

    # Compute ky spectra
    ky_mat, inputs_kept, kept_idx, skipped_idx = compute_ky_matrix_skip_bad(physical_conditions, grad_r0)

    print(f"Computed ky for {ky_mat.shape[0]} valid samples (skipped {len(skipped_idx)})")
    print(f"ky_mat shape: {ky_mat.shape}")

    save_h5(
        out_path=h5_path,
        inputs_mat=inputs_kept,
        ky_mat=ky_mat,
        grad_r0=grad_r0,
        kept_idx=kept_idx,
        skipped_idx=skipped_idx,
    )
    print("✅ Saved new ky spectra to generated_candidates/ky_spectra_new.h5")

    convert_h5_to_batch_dir_parallel(h5_path, tglf_dest_dir)

    process_tglf_batches(tglf_dest_dir, reformat_dir)
    
def setup_tglf_tests(num_tests, reference_input_path, reference_flux_path, test_dir, sampled_input_path, sampled_flux_path):
    inputs = np.load(reference_input_path, allow_pickle=True)
    fluxes = np.load(reference_flux_path, allow_pickle=True)

    num_samples = inputs.shape[0]
    assert num_samples == fluxes.shape[0]

    indices = np.arange(0, num_samples, 1)
    random_indices = np.random.choice(indices, size=num_tests, replace=False)
    random_inputs = inputs[random_indices]
    random_fluxes = fluxes[random_indices]

    temp_dir = os.path.join(test_dir, '..', 'temp')
    h5_path = os.path.join(test_dir, '..', 'inputs.h5')
    create_test_inputs(random_inputs[:, 0, :-1], h5_path, temp_dir, test_dir) # this reshaping gives (num_tests, 31)

    np.save(sampled_input_path, random_inputs)
    np.save(sampled_flux_path, random_fluxes)
    return random_inputs, random_fluxes

def run_tglf(gacode_root, test_dir):
    my_env = os.environ.copy() 
    my_env["GACODE_PLATFORM"] = "MINT_OPENMPI"
    my_env["GACODE_ROOT"] = gacode_root
    my_env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    my_env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    subprocess.run(["chmod", "+x", "./run_simulation.sh"], env=my_env)
    subprocess.run(["bash", "run_simulation.sh", "tglf", f"{test_dir}"], env=my_env)

def parse_tglf_outputs(test_dir, out_h5, input_npy, fluxes_npy):
    # subprocess.run(["python", OUTPUT_PARSING_SCRIPT_PATH, test_dir, "-o", out_h5])
    process_all_batches(test_dir, out_h5)
    h5_to_npy(out_h5, input_npy, fluxes_npy)

def compute_ky_mae(gt_inputs, inputs, fig_path):
    mae = np.abs(gt_inputs - inputs)
    mae_ky = mae[:,:,-1].squeeze() # (num_tests, nky)
    print(f'KY Diffs: {mae_ky}')
    mae_ky_all = np.reshape(mae_ky, (mae_ky.shape[0] * mae_ky.shape[1]))
    percent_mae_ky_all = mae_ky_all / np.reshape(gt_inputs[:,:,-1], (mae_ky.shape[0] * mae_ky.shape[1]))

    plt.hist(percent_mae_ky_all, bins=30, color='red')
    plt.title(f'KY Value Percent MAE: TGLF Tests on {gt_inputs.shape[0]} Random Samples from SiNN Data')
    plt.ylabel('Frequency')
    plt.xlabel(f'KY Value Percent MAE from SiNN Data')
    plt.savefig(fig_path + f'ky_percent.png')
    plt.close('all')

def diagonal_plot_summed_fluxes(gt_input_path, gt_fluxes_path, sim_input_path, sim_flux_path, fig_path):
    gt_inputs = np.load(gt_input_path, allow_pickle=True)
    gt_fluxes = np.load(gt_fluxes_path, allow_pickle=True)
    inputs = np.load(sim_input_path, allow_pickle=True)
    fluxes = np.load(sim_flux_path, allow_pickle=True)
    
    gt_fluxes = np.sum(gt_fluxes, axis=1)
    fluxes = np.sum(fluxes, axis=1)

    print(gt_fluxes.shape) 

    labels = ["OUT_G_E", "OUT_Q_E", "OUT_Q_I", "OUT_P_I"]

    for i in range(len(labels)):
        plt.scatter(np.log10(gt_fluxes[:,i]), np.log10(fluxes[:,i]), alpha=0.4)
        plt.axline((0, 0), slope=1, color='red', linestyle='--')
        plt.xlabel(f"SiNN {labels[i]} Flux (Log)")
        plt.ylabel(f"Simulated {labels[i]} Flux (Log)")
        plt.savefig(fig_path + f'/diagonal_{labels[i]}_summed_fluxes.png')
        plt.show()
        plt.close('all')

def diagonal_plot(gt_input_path, gt_fluxes_path, sim_input_path, sim_flux_path, fig_path):
    gt_inputs = np.load(gt_input_path, allow_pickle=True)
    gt_fluxes = np.load(gt_fluxes_path, allow_pickle=True)
    inputs = np.load(sim_input_path, allow_pickle=True)
    fluxes = np.load(sim_flux_path, allow_pickle=True)
    
    gt_fluxes = np.reshape(gt_fluxes, (gt_fluxes.shape[0] * gt_fluxes.shape[1], gt_fluxes.shape[2]))
    fluxes = np.reshape(fluxes, (fluxes.shape[0] * fluxes.shape[1], fluxes.shape[2]))

    print(gt_fluxes.shape) 

    labels = ["OUT_G_E", "OUT_Q_E", "OUT_Q_I", "OUT_P_I"]

    for i in range(len(labels)):
        # plt.scatter(np.log10(gt_fluxes[:,i]), np.log10(fluxes[:,i]), alpha=0.4)
        # plt.axline((0, 0), slope=1, color='red', linestyle='--')
        # plt.xlabel(f"SiNN {labels[i]} Flux (Log)")
        # plt.ylabel(f"Simulated {labels[i]} Flux (Log)")
        # plt.savefig(fig_path + f'/diagonal_{labels[i]}.png')
        # plt.show()
        # plt.close('all')
        plt.figure(figsize=(6, 4))
        sns.kdeplot(x=np.log10(gt_fluxes[:,i]), y=np.log10(fluxes[:,i]), fill=True, cmap="crest")
        plt.axline((0, 0), slope=1, color='red', linestyle='--')
        plt.title(f'Density Plot of Re-Simulated and TGLF-SiNN {labels[i]} Fluxes')
        plt.xlabel('Ground Truth Flux (TGLF-SiNN)')
        plt.ylabel('Re-Simulated Flux')
        plt.savefig(fig_path + f'/diagonal_{labels[i]}.png')
        plt.show()
        plt.close('all')
        # seaborn.kdeplot(np.log10(gt_fluxes[:,i]), np.log10(fluxes[:,i]))

def compare_fluxes(gt_input_path, gt_fluxes_path, sim_input_path, sim_flux_path, fig_path):
    gt_inputs = np.load(gt_input_path, allow_pickle=True)
    gt_fluxes = np.load(gt_fluxes_path, allow_pickle=True)
    inputs = np.load(sim_input_path, allow_pickle=True)
    fluxes = np.load(sim_flux_path, allow_pickle=True)

    compute_ky_mae(gt_inputs, inputs, fig_path)
    print(f'Number of per-ky fluxes computed: {gt_fluxes.shape[0] * gt_fluxes.shape[1]}')
    print('=' * 50)
    # Compute Raw MAE
    print(f'COMPUTING RAW MAEs ACROSS FLUXES:')
    flux_mae = np.reshape(np.abs(gt_fluxes - fluxes), (gt_fluxes.shape[0] * gt_fluxes.shape[1], 4))
    out_g_e_avg_mae = np.average(flux_mae[:,0])
    out_p_i_avg_mae = np.average(flux_mae[:,1])
    out_q_e_avg_mae = np.average(flux_mae[:,2])
    out_q_i_avg_mae = np.average(flux_mae[:,3])
    print(f'Avg. MAE for OUT_G_e: {out_g_e_avg_mae}')
    print(f'Avg. MAE for OUT_P_i: {out_p_i_avg_mae}')
    print(f'Avg. MAE for OUT_Q_e: {out_q_e_avg_mae}')
    print(f'Avg. MAE for OUT_Q_i: {out_q_i_avg_mae}')
    print()
    # Compute Percentage MAE
    epsilon = 0.000001 # to avoid divide by 0
    flux_percent_mae = np.abs(flux_mae / (np.reshape(gt_fluxes, (gt_fluxes.shape[0] * gt_fluxes.shape[1], 4)) + epsilon))
    out_g_e_percent_mae = np.average(flux_percent_mae[:,0])
    out_p_i_percent_mae = np.average(flux_percent_mae[:,1])
    out_q_e_percent_mae = np.average(flux_percent_mae[:,2])
    out_q_i_percent_mae = np.average(flux_percent_mae[:,3])
    print(f'COMPUTING PERCENTAGE MAEs ACROSS FLUXES')
    print(f'Avg. Percent MAE for OUT_G_e: {out_g_e_percent_mae}')
    print(f'Avg. Percent MAE for OUT_P_i: {out_p_i_percent_mae}')
    print(f'Avg. Percent MAE for OUT_Q_e: {out_q_e_percent_mae}')
    print(f'Avg. Percent MAE for OUT_Q_i: {out_q_i_percent_mae}')
    print()
    outliers = np.where(flux_percent_mae > 0.30, 1, 0)
    outliers_out_g_e = np.sum(outliers[:,0])
    outliers_out_q_e = np.sum(outliers[:,1])
    outliers_out_q_i = np.sum(outliers[:,2])
    outliers_out_p_i = np.sum(outliers[:,3])
    print(f'COMPUTING OUTLIERS (> 30% MAE)')
    print(f'Number of outliers for OUT_G_e: {outliers_out_g_e}')
    print(f'Number of outliers for OUT_Q_e: {outliers_out_q_e}')
    print(f'Number of outliers for OUT_Q_i: {outliers_out_q_i}')
    print(f'Number of outliers for OUT_P_i: {outliers_out_p_i}')
    print()
    inlier_idxs = np.where(flux_percent_mae <= 0.30)
    gt_fluxes_all = np.reshape(gt_fluxes, (gt_fluxes.shape[0] * gt_fluxes.shape[1], 4))
    fluxes_all = np.reshape(fluxes, (gt_fluxes.shape[0] * gt_fluxes.shape[1], 4))
    flux_mae_inliers = np.abs(gt_fluxes_all[inlier_idxs, :] - fluxes_all[inlier_idxs, :])
    flux_mae_inliers_out_g_e = np.average(flux_mae_inliers[:,0])
    flux_mae_inliers_out_q_e = np.average(flux_mae_inliers[:,1])
    flux_mae_inliers_out_q_i = np.average(flux_mae_inliers[:,2])
    flux_mae_inliers_out_p_i = np.average(flux_mae_inliers[:,3])

    print(f'COMPUTING RAW MAE ACROSS FLUXES AFTER REMOVING OUTLIERS:')
    print(f'Avg. inlier-only MAE for OUT_G_e: {flux_mae_inliers_out_g_e}')
    print(f'Avg. inlier-only MAE for OUT_P_i: {flux_mae_inliers_out_q_e}')
    print(f'Avg. inlier-only MAE for OUT_Q_e: {flux_mae_inliers_out_q_i}')
    print(f'Avg. inlier-only MAE for OUT_Q_i: {flux_mae_inliers_out_p_i}')
    print()

    flux_percent_mae_inliers = np.abs(flux_mae_inliers / (gt_fluxes_all[inlier_idxs, :] + epsilon))
    percent_mae_inliers_out_g_e = np.average(flux_percent_mae_inliers[:,0])
    percent_mae_inliers_out_q_e = np.average(flux_percent_mae_inliers[:,1])
    percent_mae_inliers_out_q_i = np.average(flux_percent_mae_inliers[:,2])
    percent_mae_inliers_out_p_i = np.average(flux_percent_mae_inliers[:,3])

    print(f'COMPUTING PERCENT MAE ACROSS FLUXES AFTER REMOVING OUTLIERS:')
    print(f'Avg. inlier-only percent MAE for OUT_G_e: {percent_mae_inliers_out_g_e}')
    print(f'Avg. inlier-only percent MAE for OUT_Q_e: {percent_mae_inliers_out_q_e}')
    print(f'Avg. inlier-only percent MAE for OUT_Q_i: {percent_mae_inliers_out_q_i}')
    print(f'Avg. inlier-only percent MAE for OUT_P_i: {percent_mae_inliers_out_p_i}')
    print()

    out_g_e_valid = np.where(flux_percent_mae[:,0] <= 0.20)
    out_q_e_valid = np.where(flux_percent_mae[:,1] <= 0.20)
    out_q_i_valid = np.where(flux_percent_mae[:,2] <= 0.20)
    out_p_i_valid = np.where(flux_percent_mae[:,3] <= 0.20)
    
    valid_idxs = np.intersect1d(out_g_e_valid, out_g_e_valid)
    valid_idxs = np.intersect1d(valid_idxs, out_q_i_valid)
    valid_idxs = np.intersect1d(valid_idxs, out_p_i_valid)

    inputs_flattened = np.reshape(inputs, (inputs.shape[0] * inputs.shape[1], 32))       
    valid_kys = inputs_flattened[valid_idxs, -1]

    plt.hist(valid_kys, bins=30, color='blue')
    plt.title(f'Validation KY Values: TGLF Tests on {gt_inputs.shape[0]} Random Samples from SiNN Data')
    plt.ylabel('Frequency')
    plt.xlabel(f'KY Value from SiNN Data')
    plt.savefig(fig_path + f'ky_valid.png')
    plt.close('all')

    print(f'NUMBER OF VALIDATION SAMPLES (<= 20% MAE)')
    print(f'{len(valid_idxs)}')

    labels = ['OUT_G_e', 'OUT_P_i', 'OUT_Q_e', 'OUT_Q_i']
    colors = ['blue', 'red', 'green', 'orange']

    for i in range(len(labels)):
        plt.hist(flux_mae[:,i], bins=30, color=colors[i], label=labels[i])
        plt.legend()
        plt.title(f'TGLF Tests on {gt_inputs.shape[0]} Random Samples from SiNN Data')
        plt.ylabel('Frequency')
        plt.xlabel(f'{labels[i]} MAE from SiNN Data')
        plt.savefig(fig_path + f'{labels[i]}.png')
        plt.close('all')
    
    for i in range(len(labels)):
        plt.hist(flux_percent_mae[:,i], bins=30, color=colors[i], label=labels[i], range=(0, 1))
        plt.legend()
        plt.title(f'TGLF Tests on {gt_inputs.shape[0]} Random Samples from SiNN Data')
        plt.ylabel('Frequency')
        plt.xlabel(f'{labels[i]} Percent MAE from SiNN Data')
        plt.savefig(fig_path + f'percent_{labels[i]}.png')
        plt.close('all')
    
    for i in range(len(labels)):
        # plt.hist(np.reshape(gt_fluxes[:,:,i], (gt_fluxes.shape[0] * gt_fluxes.shape[1])), bins=30, color='blue', alpha=0.5, label=labels[i])
        plt.hist(np.reshape(fluxes[:,:,i], (fluxes.shape[0] * fluxes.shape[1])), bins=50, color='red', alpha = 0.5, range=(-10, 50), label=labels[i])
        plt.legend()
        plt.title(f'TGLF Tests on {gt_inputs.shape[0]} Random Samples from SiNN Data')
        plt.ylabel('Frequency')
        plt.xlabel(f'{labels[i]} Flux')
        plt.savefig(fig_path + f'sim_flux_{labels[i]}.png')
        plt.close('all')

def format_num(n):
    if n < 10:
        return f'00{n}'
    elif n < 100:
        return f'0{n}'
    else:
        return f'{n}'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", "-n")
    parser.add_argument("--output_dir", "-o")
    args = parser.parse_args()

    # SHOULD NOT CHANGE UNLESS FILE STRUCTURE CHANGED DURING SETUP
    OUTPUT_PARSING_SCRIPT_PATH = "./gacode-docker/output_parsing/batch_tglf_processor.py"
    # CAN CHANGE
    gacode_root = '/home/zach/local_gacode_abs/gacode'
    input_path = './reference_inputs.npy'
    flux_path = './reference_fluxes.npy'
    num_tests = int(args.num_samples)
    test_dir_root = args.output_dir
    os.makedirs(test_dir_root, exist_ok=True)
    test_dir_sim = os.path.join(test_dir_root, 'tglf_simready')
    test_sim_h5 = os.path.join(test_dir_root, 'results.h5')
    test_inputs_npy = os.path.join(test_dir_root, 'inputs.npy')
    test_fluxes_npy = os.path.join(test_dir_root, 'fluxes.npy')
    sampled_input_path = os.path.join(test_dir_root, f'ref_inputs_sampled_{num_tests}.npy')
    sampled_flux_path = os.path.join(test_dir_root, f'ref_fluxes_sampled_{num_tests}.npy')
    fig_path = os.path.join(test_dir_root, 'mae_') # NOTE: this is not a complete path, it is completed in the compare_fluxes function call
    diag_fig_path = test_dir_root
    # RUN TEST SUITE
    setup_tglf_tests(num_tests, input_path, flux_path, test_dir_sim, sampled_input_path, sampled_flux_path)
    run_tglf(gacode_root, test_dir_sim)
    parse_tglf_outputs(test_dir_sim, test_sim_h5, test_inputs_npy, test_fluxes_npy)
    compare_fluxes(sampled_input_path, sampled_flux_path, test_inputs_npy, test_fluxes_npy, fig_path)
    diagonal_plot(sampled_input_path, sampled_flux_path, test_inputs_npy, test_fluxes_npy, diag_fig_path)
    diagonal_plot_summed_fluxes(sampled_input_path, sampled_flux_path, test_inputs_npy, test_fluxes_npy, diag_fig_path)
