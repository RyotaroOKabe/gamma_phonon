#%%
# python==3.10
from mp_api.client import MPRester
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
import time, os, sys
from os.path import join as opj
import json
import pandas as pd
from tqdm import tqdm
import csv
import pickle as pkl
import pymatgen
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
import datetime
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
api_key = "PvfnzQv5PLh4Lzxz1pScKnAtcmmWVaeU"
generate_fig = False

#%%
"""Read phonon data (.json file)
data source
https://figshare.com/articles/dataset/Parsed_phonon_data/5649298?backTo=/collections/High-throughput_Density-Functional_Perturbation_Theory_phonons_for_inorganic_materials/3938023
"""
ph_dict = {}
mp_ids = []
ph_files = sorted(glob(opj('../phonon', '*.json')))
for file in ph_files:
    mpid = file[10:-5]
    f = open(file)
    ph_data = json.load(f)
    ph_dict[mpid] = ph_data
    mp_ids.append(mpid)
    if generate_fig:
        band = np.array(ph_data['phonon']['ph_bandstructure'])
        ph_dos = np.array(ph_data['phonon']['ph_dos'])
        freq = np.array(ph_data['phonon']['dos_frequencies'])
        ymin, ymax = np.min(freq), np.max(freq)
        fig = plt.figure(constrained_layout=True, figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax1.plot(band)
        ax1.set_ylim([ymin, ymax])
        ax2 = fig.add_subplot(122)
        ax2.plot(ph_dos, freq)
        ax2.set_ylim([ymin, ymax])
        fig.suptitle(mpid)
        fig.subplots_adjust(hspace=0.6)
        fig.patch.set_facecolor('white')
        fig.show()
        fig.savefig(f'bands_2/{mpid}.png') 
pkl.dump(ph_dict, open('data/phonon.pkl', 'wb'))

#%%
# [mp_id, formula] >> csv file (takes long)
id_formula = {}
with MPRester(api_key) as mpr:
    with open('data/mp_data.csv', 'w') as csvfile:
        for id in mp_ids:
            doc = mpr.summary.get_data_by_id(id)
            formula = doc.formula_pretty
            id_formula[id]=formula
            csvfile.write(f'{id}, {formula}')
            csvfile.write('\n')

#%%
# download MP structures and save as pymatgen (takes long)
structures_pm = {}
for id in mp_ids:
    print(id)
    with MPRester(api_key) as mpr:
        struct = mpr.get_structure_by_material_id(id)
        structures_pm[id] = struct
pkl.dump(structures_pm, open('data/structures_pm.pkl', 'wb'))

print(len(structures_pm))

#%%
# Get Pandas data frame

# structures_pm = pkl.load(open(f'data/structures_pm.pkl', 'rb'))
# ph_dict = pkl.load(open(f'data/phonon.pkl', 'rb'))
# mp_ids = sorted(list(structures_pm.keys()))

df = pd.DataFrame({})
for i, id in tqdm(enumerate(mp_ids), total=len(mp_ids), bar_format=bar_format):
    struct = structures_pm[id]
    struct_ase = Atoms(list(map(lambda x: x.symbol, structures_pm[id].species)) , # list of symbols got from pymatgen
            positions=structures_pm[id].cart_coords.copy(),
            cell=structures_pm[id].lattice.matrix.copy(), pbc=True) 
    freq = np.array(ph_dict[id]['phonon']['dos_frequencies'])
    qpts = np.array(ph_dict[id]['phonon']['qpts'])
    ph_dos = np.array(ph_dict[id]['phonon']['ph_dos'])
    band = np.array(ph_dict[id]['phonon']['ph_bandstructure'])
    g_pt_ = np.where(band[:, 0]==band[:, 0].min())[0]
    choice = int(g_pt_.shape[0] / 2)    # get the middle point
    g_pt = g_pt_[choice]
    g_phs = band[choice, :]
    print('Gamma point for use: ', g_pt)
    print('g_phs', g_phs)
    rrr = {"id": id,
           "formula": struct_ase.get_chemical_formula(),
           'sites': struct.num_sites,
           'species':[list(set(struct_ase.get_chemical_symbols()))],
           "structure_pm":[struct],
           "structure": [struct_ase],
           "phfreq": [freq],
           "phfreq_shape": [np.array(freq).shape],
           "qpts": [qpts],
           "qpts_shape": [np.array(qpts).shape],
           "phdos": [ph_dos],
           "pdos": [ph_dos.tolist()],
           "phdos_shape": [np.array(ph_dos).shape],
           "band":[band],
           "band_shape":[np.array(band).shape],
           "g_pts": [g_pt_],
           "g_pt_idx": [choice],
           "g_pt":[g_pt],
           "g_phs":[g_phs],
           }
    df0 = pd.DataFrame(data=rrr)
    df = pd.concat([df, df0], ignore_index=True)

df.to_pickle(f'data/df_struct_phonon.pkl')
#df = pd.read_pickle(f'data/df_struct_phonon.pkl'

species = sorted(list(set(df['species'].sum())))

# %%
