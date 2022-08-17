#%%
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
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
api_key = "PvfnzQv5PLh4Lzxz1pScKnAtcmmWVaeU"

icsd_pd = pd.read_csv('data/mp_data.csv', header=None)
mp_ids = icsd_pd.iloc[:, 0].to_list()

#%%
structures_pm = {}
#structures_pm2 = {}
structures_pm2 = {}
for id in mp_ids:
    print(id)
    with MPRester(api_key) as mpr:
        struct = mpr.get_structure_by_material_id(id)
        structures_pm[id] = struct
        # docs = mpr.summary.search(material_ids=[id], fields=["structure"])
        # struct2 = docs[0].structure
        # structures_pm2[id] = struct2
        doc = mpr.summary.get_data_by_id(id, fields=["structure"])
        struct2 = doc.structure
        structures_pm2[id] = struct2

#%%
pkl.dump(structures_pm, open('data/structures_pm.pkl', 'wb'))
pkl.dump(structures_pm2, open('data/structures_pm2.pkl', 'wb'))

#%%
print(structures_pm==structures_pm2)
# %%
