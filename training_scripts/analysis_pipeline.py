import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import voduct.save_io as io
import sys
import os

if __name__=="__main__":
    plot_keys = ['train_loss', 'train_acc',
                 'mask_train_loss', 'mask_train_acc', 
                 'val_loss', 'val_acc', 
                 'mask_val_loss', 'mask_val_acc'] 
    for exp_folder in sys.argv[1:]:
        df_path = os.path.join(exp_folder,"model_data.csv")
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
        else:
            df = None
        model_folders = io.get_model_folders(exp_folder)
        for model_folder in model_folders:
            table = {
                
            }
            path = os.path.join(exp_folder,model_folder)
            checkpt_files = io.get_checkpoints(path)
            plot_vals = {k:[] for k in plot_keys}
            epochs = []
            for checkpt_f in checkpt_files:
                checkpt = io.load_checkpoint(checkpt_f)
                epochs.append(checkpt['epoch'])
                for pk in plot_keys:
                    if pk in checkpt: 
                        if pk in plot_vals:
                            plot_vals[pk].append(checkpt[pk])
                        else:
                            plot_vals[pk] = [checkpt[pk]]
            for pk in plot_vals.keys():
                if len(plot_vals[pk]) > 0:
                    fig = plt.figure(figsize=(5,5), dpi=80)
                    plt.plot(epochs, plot_vals[pk])
                    plt.title(pk)
                    name = os.path.join(path, pk+".png")
                    plt.savefig(name)
                    name = name.replace("/", "-")
                    name = os.path.join("img_viewing", name)
                    plt.savefig(name)
                    plt.close()
                else:
                    print(pk, "is empty")


