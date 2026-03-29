#!/usr/bin/env python3
"""
Refactored TGLF Saturation Rules using real PyroScan data.
This script reads simulation results from a directory rather than mocking them.
"""
import os
import re
import numpy as np
import h5py
import os
from qlgyro_and_tglf_flux_calculation import calculate_TGLF_flux
from omfit_classes.omfit_tglf import OMFITtglf

def apply_tglf_saturation(omfit_tglf, sat_rule, alpha_zf_in=-1):
   """
   Apply TGLF saturation rules based on OMFIT objects and implementations
   """
   inputs = omfit_tglf['input.tglf']
   tglf_kys = np.asarray(omfit_tglf['eigenvalue_spectrum']['ky'])
   tglf_gammas = np.asarray(omfit_tglf['eigenvalue_spectrum']['gamma']).T

   # Original: (species: 3, field: 2, mode_num: 5, ky: 24)
   # Expects shapes to be (ky, mode, species, field)
   particle_flux = np.asarray(omfit_tglf['QL_flux_spectrum']['particle'].astype(np.float32)).transpose((3, 2, 0, 1))
   energy_flux = np.asarray(omfit_tglf['QL_flux_spectrum']['energy'].astype(np.float32)).transpose((3, 2, 0, 1))
   toroidal_stress_flux = np.asarray(omfit_tglf['QL_flux_spectrum']['toroidal stress'].astype(np.float32)).transpose((3, 2, 0, 1))
   parallel_stress_flux = np.asarray(omfit_tglf['QL_flux_spectrum']['parallel stress'].astype(np.float32)).transpose((3, 2, 0, 1))
   exchange_flux = np.asarray(omfit_tglf['QL_flux_spectrum']['exchange'].astype(np.float32)).transpose((3, 2, 0, 1))

   tglf_ql = np.stack([particle_flux, energy_flux, toroidal_stress_flux, parallel_stress_flux, exchange_flux], axis=-1)
   tglf_sat = calculate_TGLF_flux(inputs, tglf_kys, tglf_gammas, tglf_ql, sat_rule, alpha_zf_in)
   sumf = tglf_sat['sum_flux_spectrum']
   return sumf, tglf_kys

def parse_tglf_dir(dir, sat_rule):
    # Ensure the directory exists
    if not os.path.isdir(dir):
        raise FileNotFoundError(f"TGLF output directory not found: {dir}")
    
    tglf = OMFITtglf(dir)
    # Both {sumf, ql_gt} have shape: (ky, mode, species, field, type)
    sumf, kys = apply_tglf_saturation(tglf, sat_rule)
    
    # Parse ground truth fluxes from out.tglf.sum_flux_spectrum
    particle_flux = np.asarray(tglf['sum_flux_spectrum']['particle'].astype(np.float32)).transpose((2, 0, 1))
    energy_flux = np.asarray(tglf['sum_flux_spectrum']['energy'].astype(np.float32)).transpose((2, 0, 1))
    toroidal_stress_flux = np.asarray(tglf['sum_flux_spectrum']['toroidal_stress'].astype(np.float32)).transpose((2, 0, 1))
    parallel_stress_flux = np.asarray(tglf['sum_flux_spectrum']['parallel_stress'].astype(np.float32)).transpose((2, 0, 1))
    exchange_flux = np.asarray(tglf['sum_flux_spectrum']['exchange'].astype(np.float32)).transpose((2, 0, 1))

    sumf_gt = np.stack([particle_flux, energy_flux, toroidal_stress_flux, parallel_stress_flux, exchange_flux], axis=-1)

    # The following are summed over fields and modes
    G_elec_summed = np.sum(np.sum(np.sum(sumf[:,:,0,:,0], axis=-1), axis=-1), axis=-1)
    Q_elec_summed = np.sum(np.sum(np.sum(sumf[:,:,0,:,1], axis=-1), axis=-1), axis=-1)
    Q_ions_summed = np.sum(np.sum(np.sum(np.sum(sumf[:,:,1:,:,1], axis=-1), axis=-1), axis=-1), axis=-1)
    P_ions_summed = np.sum(np.sum(np.sum(np.sum(sumf[:,:,1:,:,2], axis=-1), axis=-1), axis=-1), axis=-1)

    # The following only need to be summed over fields (out.tglf.sum_flux_spectrum already sums over modes)
    G_elec_gt_per_ky = np.sum(sumf_gt[:,0,:,0], axis=-1)
    Q_elec_gt_per_ky = np.sum(sumf_gt[:,0,:,1], axis=-1)
    Q_ion1_gt_per_ky = np.sum(sumf_gt[:,1,:,1], axis=-1)
    Q_ion2_gt_per_ky = np.sum(sumf_gt[:,2,:,1], axis=-1)
    P_ion1_gt_per_ky = np.sum(sumf_gt[:,1,:,2], axis=-1)
    P_ion2_gt_per_ky = np.sum(sumf_gt[:,2,:,2], axis=-1)

    fluxes = np.array([G_elec_summed, Q_elec_summed , Q_ions_summed, P_ions_summed], dtype=np.float32)
    file_fluxes = np.array([G_elec_gt_per_ky, Q_elec_gt_per_ky, Q_ion1_gt_per_ky, Q_ion2_gt_per_ky, P_ion1_gt_per_ky, P_ion2_gt_per_ky], dtype=np.float32)
    ky_array = kys.astype(np.float32)
    tglf_inputs = tglf['input.tglf']
    return fluxes, sumf, ky_array, tglf_inputs, file_fluxes

def find_batch_directories(root_dir):
   """
   Find all batch-XXX directories in root_dir.
   Returns sorted list of (batch_number, batch_path) tuples.
   """
   batch_pattern = re.compile(r'^batch-(\d+)$')
   batches = []
  
   for item in os.listdir(root_dir):
       item_path = os.path.join(root_dir, item)
       if os.path.isdir(item_path):
           match = batch_pattern.match(item)
           if match:
               batch_num = int(match.group(1))
               batches.append((batch_num, item_path))
  
   # Sort by batch number
   batches.sort(key=lambda x: x[0])
   return batches

def prepare_input_dict(in0):
   """
   Prepare input dictionary with required TGLF keys for HDF5 storage.
   """
   TGLF_KEYS = [
       "RLTS_3", "KAPPA_LOC", "ZETA_LOC", "TAUS_3", "VPAR_1", "Q_LOC", "RLNS_1", "TAUS_2",
       "Q_PRIME_LOC", "P_PRIME_LOC", "ZMAJ_LOC", "VPAR_SHEAR_1", "RLTS_2", "S_DELTA_LOC",
       "RLTS_1", "RMIN_LOC", "DRMAJDX_LOC", "AS_3", "RLNS_3", "DZMAJDX_LOC", "DELTA_LOC",
       "S_KAPPA_LOC", "ZEFF", "VEXB_SHEAR", "RMAJ_LOC", "AS_2", "RLNS_2", "S_ZETA_LOC",
       "BETAE_log10", "XNUE_log10", "DEBYE_log10"
   ]
  
   input_dict = {k: in0.get(k, np.nan) for k in TGLF_KEYS}
  
   # Compute log10 values if base values exist
   for key in ["BETAE", "XNUE", "DEBYE"]:
       if key in in0:
           input_dict[f"{key}_log10"] = np.log10(in0[key])
  
   return input_dict

def process_all_batches_tglf(root_dir, output_h5, output_gt_fluxes, sat_rule=2, start_batch=None, end_batch=None):
   """
   Process all batch directories and save to a single HDF5 file.
  
   Parameters:
   -----------
   root_dir : str
       Root directory containing batch-XXX subdirectories
   output_h5 : str
       Path to output HDF5 file
   sat_rule : int
       Saturation rule (1, 2, or 3)
   start_batch : int, optional
       Start processing from this batch number
   end_batch : int, optional
       Stop processing at this batch number (inclusive)
   """
   # Find all batch directories
   batches = find_batch_directories(root_dir)
  
   if not batches:
       print(f"❌ No batch directories found in {root_dir}")
       return
  
   print(f"📊 Found {len(batches)} batch directories")
  
   # Filter by start/end if specified
   if start_batch is not None:
       batches = [(n, p) for n, p in batches if n >= start_batch]
   if end_batch is not None:
       batches = [(n, p) for n, p in batches if n <= end_batch]
  
   print(f"📝 Processing {len(batches)} batches (SAT{sat_rule})")
  
   # Remove existing HDF5 file if it exists
   if os.path.exists(output_h5):
       print(f"⚠️  Removing existing file: {output_h5}")
       os.remove(output_h5)
  
   # Process each batch
   success_count = 0
   fail_count = 0
   gt_fluxes = []
   for batch_num, batch_path in batches:
       result = parse_tglf_dir(batch_path, sat_rule)
       if result is not None:
           fluxes, sumf, ky, in0, file_fluxes = result
           gt_fluxes.append(file_fluxes)
          
           # Prepare input dictionary
           input_dict = prepare_input_dict(in0)
          
           # Save to HDF5
           try:
               append_to_h5_individual_keys(
                   h5_path=output_h5,
                   input_dict=input_dict,
                   fluxes=fluxes,
                   sumf=sumf,
                   ky=ky,
                   meta={
                       'batch_num': batch_num,
                       'sat_rule': sat_rule,
                       'nky': len(ky),
                       'nspecies': sumf.shape[1],
                   }
               )
               success_count += 1
           except Exception as e:
               print(f"  ❌ Error saving to HDF5: {e}")
               fail_count += 1
       else:
           fail_count += 1
  
   # Summary
   gt_fluxes = np.stack(gt_fluxes, axis=0)
   np.save(output_gt_fluxes, gt_fluxes)
   print(f"\n{'='*60}")
   print(f"✅ Successfully processed: {success_count}/{len(batches)} batches")
   if fail_count > 0:
       print(f"❌ Failed: {fail_count}/{len(batches)} batches")
   print(f"💾 Results saved to: {output_h5}")
   print(f"{'='*60}\n")

def append_to_h5_individual_keys(h5_path, input_dict, fluxes, sumf, ky, meta=None):
   """
   Append data to HDF5 file using individual keys.
   Compatible with the batch processing workflow from parse_outputs.py
   """
  
   fluxes = np.atleast_1d(fluxes).astype(np.float32)
   sumf = np.asarray(sumf, dtype=np.float32)
   ky = np.asarray(ky, dtype=np.float32)
  
   with h5py.File(h5_path, 'a') as f:
       print("📥 Appending to HDF5:")
       print(f"  fluxes: {fluxes.shape}")
       print(f"  sumf: {sumf.shape}")
       print(f"  ky: {ky.shape}")
      
       # Save scalar input keys
       for name, data in input_dict.items():
           data = np.atleast_1d(data).astype(np.float32)
           if name not in f:
               f.create_dataset(name, data=data, maxshape=(None,), chunks=True)
           else:
               f[name].resize(f[name].shape[0] + 1, axis=0)
               f[name][-1] = data
      
       # Save flux vector (G_e, Q_e, Q_i, P_i)
       flux_names = ["OUT_G_e", "OUT_Q_e", "OUT_Q_i", "OUT_P_i"]
       for i, name in enumerate(flux_names):
           val = np.float32(fluxes[i])
           if name not in f:
               f.create_dataset(name, data=[val], maxshape=(None,), chunks=True)
           else:
               f[name].resize((f[name].shape[0] + 1), axis=0)
               f[name][-1] = val
      
       # Save sumf matrix: (1, nky, ns, nf, 5)
       if "sumf" not in f:
           f.create_dataset("sumf", data=sumf[None, ...], maxshape=(None,) + sumf.shape, chunks=True)
       else:
           f["sumf"].resize((f["sumf"].shape[0] + 1), axis=0)
           f["sumf"][-1] = sumf
      
       # Save ky array: (1, nky)
       if "ky" not in f:
           f.create_dataset("ky", data=ky[None, :], maxshape=(None, ky.shape[0]), chunks=True)
       else:
           f["ky"].resize((f["ky"].shape[0] + 1), axis=0)
           f["ky"][-1] = ky
      
       # Save meta info
       if meta:
           meta_grp = f.require_group("meta")
           for key, value in meta.items():
               value = np.asarray(value)
               if key not in meta_grp:
                   meta_grp.create_dataset(
                       key,
                       data=value[None, ...] if value.ndim > 0 else value[None],
                       maxshape=(None,) + value.shape if value.ndim > 0 else (None,),
                       chunks=True
                   )
               else:
                   meta_grp[key].resize(meta_grp[key].shape[0] + 1, axis=0)
                   meta_grp[key][-1] = value

# === Run (parallel) ===
if __name__ == "__main__":
    # process_all_batches("./test3/tglf_simready", "./pyro_parsing.h5")
    parse_tglf_dir("./results/pyrofix_commitfix_100_parsefix/tglf_simready/batch-000", 2)
    # parse_tglf_dir("./results/pyrofix_commitfix_100_parsefix/tglf_simready/batch-000", 2)