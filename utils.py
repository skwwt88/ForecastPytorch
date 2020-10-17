import random
import torch
import numpy as np
import os
import glob 

def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def model_name(name, score):
     return "{0}_{1:.8f}.pkl".format(name, score)

def find_best_model_file(name, model_folder, use_min = True):
     files = glob.glob(os.path.join(model_folder, "{}*.pkl".format(name)))
     files.sort()
     return files[0] if use_min else files[-1]

def clean_models(name, model_folder):
     for file in glob.glob(os.path.join(model_folder, "{}*.pkl".format(name))):
          os.remove(file)

def save_model(model, name, score, model_folder):
     if not os.path.exists(model_folder):
          os.makedirs(model_folder)

     name = model_name(name, score)
     torch.save(model.state_dict(), os.path.join(model_folder, name))