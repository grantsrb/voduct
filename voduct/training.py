import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import PoissonNLLLoss, MSELoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR
import time
from tqdm import tqdm
import math
from queue import Queue
from collections import deque
import psutil
import gc
import resource
import json
import voduct.datas as datas
import voduct.models as models
import ml_utils.save_io as io
from ml_utils.utils import try_key
from ml_utils.training import get_exp_num,record_session,get_save_folder

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def train(hyps, verbose=True):
    """
    hyps: dict
        contains all relavent hyperparameters
    """
    # Set manual seed
    hyps['exp_num'] = get_exp_num(hyps['main_path'], hyps['exp_name'])
    hyps['save_folder'] = get_save_folder(hyps)
    if not os.path.exists(hyps['save_folder']):
        os.mkdir(hyps['save_folder'])
    hyps['seed'] = try_key(hyps,'seed', int(time.time()))
    torch.manual_seed(hyps['seed'])
    np.random.seed(hyps['seed'])
    model_class = hyps['model_class']
    hyps['model_type'] = models.TRANSFORMER_TYPE[model_class]
    if not hyps['init_decs'] and not hyps['gen_decs'] and\
                                not hyps['ordered_preds']:
        s = "WARNING!! You probably want to set ordered preds to True "
        s += "with your current configuration!!"
        print(s)

    if verbose:
        print("Retreiving Dataset")
    if "shuffle_split" not in hyps and hyps['shuffle']:
        hyps['shuffle_split'] = True
    train_data,val_data = datas.get_data(**hyps)
    hyps['enc_slen'] = train_data.X.shape[-1]
    hyps['dec_slen'] = train_data.Y.shape[-1]-1
    #if hyps[
    train_loader = torch.utils.data.DataLoader(train_data,
                                    batch_size=hyps['batch_size'],
                                    shuffle=hyps['shuffle'])
    val_loader = torch.utils.data.DataLoader(val_data,
                                  batch_size=hyps['batch_size'])
    hyps['n_vocab'] = len(train_data.word2idx.keys())

    if verbose:
        print("Making model")
    model = getattr(models,model_class)(**hyps)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'],
                                           weight_decay=hyps['l2'])
    init_checkpt = try_key(hyps,"init_checkpt", None)
    if init_checkpt is not None and init_checkpt != "":
        if verbose:
            print("Loading state dicts from", init_checkpt)
        checkpt = io.load_checkpoint(init_checkpt)
        model.load_state_dict(checkpt["state_dict"])
        optimizer.load_state_dict(checkpt["optim_dict"])
    lossfxn = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                    patience=6,
                                                    verbose=True)
    if model.transformer_type != models.DICTIONARY:
        hyps['emb_alpha'] = 0
    if verbose:
        print("Beginning training for {}".format(hyps['save_folder']))
        print("train shape:", train_data.X.shape)
        print("val shape:", val_data.X.shape)
        print("n_vocab:", hyps['n_vocab'])

    record_session(hyps,model)
    if hyps['dataset'] == "WordProblem":
        save_data_structs(train_data.samp_structs)

    if hyps['exp_name'] == "test":
        hyps['n_epochs'] = 2
    epoch = -1
    alpha = hyps['loss_alpha']
    emb_alpha = hyps['emb_alpha']
    print()
    idx2word = train_data.idx2word
    mask_idx = train_data.word2idx["<MASK>"]
    while epoch < hyps['n_epochs']:
        epoch += 1
        print("Epoch:{} | Model:{}".format(epoch,hyps['save_folder']))
        starttime = time.time()
        avg_loss = 0
        avg_acc = 0
        avg_indy_acc = 0
        avg_emb_loss = 0
        mask_avg_loss = 0
        mask_avg_acc = 0
        model.train()
        print("Training...")
        optimizer.zero_grad()
        for b,(x,y) in enumerate(train_loader):
            torch.cuda.empty_cache()
            if model.transformer_type == models.AUTOENCODER:
                targs = x.data[:,1:]
                y = x.data
            elif model.transformer_type == models.DICTIONARY:
                targs = x.data[:,1:]
                emb_targs = model.embeddings(y.to(DEVICE))
                word_to_define = idx2word[y.squeeze()[0].item()]
                y = x.data
            else:
                targs = y.data[:,1:]
            og_shape = targs.shape
            if hyps['masking_task']:
                x,y,mask = mask_words(x, y, mask_p=hyps['mask_p'])
            y = y[:,:-1]
            preds = model(x.to(DEVICE),y.to(DEVICE))

            tot_loss = 0
            if model.transformer_type == models.DICTIONARY:
                emb_preds = preds[1]
                preds = preds[0]
                emb_loss = F.mse_loss(emb_preds,emb_targs.data)
                tot_loss += (emb_alpha)*emb_loss
                avg_emb_loss += emb_loss.item()

            if epoch % 3 == 0 and b == 0:
                if model.transformer_type == models.DICTIONARY:
                    print("Word:", word_to_define)
                whr = torch.where(y[0]==mask_idx)[0]
                endx = y.shape[-1] if len(whr) == 0 else whr[0].item()
                print("y:",[idx2word[a.item()] for a in y[0,:endx]])
                print("t:", [idx2word[a.item()] for a in targs[0,:endx-1]])
                ms = torch.argmax(preds,dim=-1)
                print("p:", [idx2word[a.item()] for a in ms[0,:endx-1]])
                del ms
            targs = targs.reshape(-1)

            if hyps['masking_task']:
                print("masking!")
                # Mask loss and acc
                preds = preds.reshape(-1,preds.shape[-1])
                mask = mask.reshape(-1).bool()
                idxs = torch.arange(len(mask))[mask]
                mask_preds = preds[idxs]
                mask_targs = targs[idxs]
                mask_loss = lossfxn(mask_preds,mask_targs)
                mask_preds = torch.argmax(mask_preds,dim=-1)
                mask_acc = (mask_preds==mask_targs).sum().float()
                mask_acc = mask_acc/idxs.numel()

                mask_avg_acc  += mask_acc.item()
                mask_avg_loss += mask_loss.item()
            else:
                mask_loss = torch.zeros(1).to(DEVICE)
                mask_acc = torch.zeros(1).to(DEVICE)

                mask_avg_acc  += mask_acc.item()
                mask_avg_loss += mask_loss.item()

            # Tot loss and acc
            preds = preds.reshape(-1,preds.shape[-1])
            targs = targs.reshape(-1).to(DEVICE)
            if not hyps['masking_task']:
                bitmask = (targs!=mask_idx)
                loss = (1-emb_alpha)*lossfxn(preds[bitmask],
                                             targs[bitmask])
            else:
                loss = lossfxn(preds,targs)

            if hyps['masking_task']:
                temp = ((alpha)*loss + (1-alpha)*mask_loss)
                tot_loss += temp/hyps['n_loss_loops']
            else:
                tot_loss += loss/hyps['n_loss_loops']
            tot_loss.backward()

            if b % hyps['n_loss_loops'] == 0 or b==len(train_loader)-1:
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                preds = torch.argmax(preds,dim=-1)
                sl = og_shape[-1]
                if not hyps['masking_task']:
                    eq = (preds==targs).float()
                    indy_acc = eq[bitmask].mean()
                    eq[~bitmask] = 1
                    eq = eq.reshape(og_shape)
                    acc = (eq.sum(-1)==sl).float().mean()
                else:
                    eq = (preds==targs).float().reshape(og_shape)
                    acc = (eq.sum(-1)==sl).float().mean()
                    indy_acc = eq.mean()
                preds = preds.cpu()

            avg_acc += acc.item()
            avg_indy_acc += indy_acc.item()
            avg_loss += loss.item()


            if hyps["masking_task"]:
                s = "Mask Loss:{:.5f} | Acc:{:.5f} | {:.0f}%"
                s = s.format(mask_loss.item(), mask_acc.item(),
                                               b/len(train_loader)*100)
            elif model.transformer_type == models.DICTIONARY:
                s = "Loss:{:.5f} | Acc:{:.5f} | Emb:{:.5f} | {:.0f}%"
                s = s.format(loss.item(),acc.item(),emb_loss.item(),
                                          b/len(train_loader)*100)
            else:
                s = "Loss:{:.5f} | Acc:{:.5f} | {:.0f}%"
                s = s.format(loss.item(), acc.item(),
                                          b/len(train_loader)*100)
            print(s, end=len(s)*" " + "\r")
            if hyps['exp_name'] == "test" and b>5: break
        print()
        mask_train_loss = mask_avg_loss/len(train_loader)
        mask_train_acc = mask_avg_acc/len(train_loader)
        train_avg_loss = avg_loss/len(train_loader)
        train_avg_acc = avg_acc/len(train_loader)
        train_avg_indy = avg_indy_acc/len(train_loader)
        train_emb_loss = avg_emb_loss/len(train_loader)

        stats_string = "Train - Loss:{:.5f} | Acc:{:.5f} | Indy:{:.5f}\n"
        stats_string = stats_string.format(train_avg_loss,
                                           train_avg_acc,
                                           train_avg_indy)
        if hyps['masking_task']:
            stats_string+="Tr. Mask Loss:{:.5f} | Tr. Mask Acc:{:.5f}\n"
            stats_string = stats_string.format(mask_train_loss,
                                               mask_train_acc)
        elif model.transformer_type==models.DICTIONARY:
            stats_string+="Train Emb Loss:{:.5f}\n"
            stats_string = stats_string.format(train_emb_loss)
        model.eval()
        avg_loss = 0
        avg_acc = 0
        avg_indy_acc = 0
        avg_emb_loss = 0
        mask_avg_loss = 0
        mask_avg_acc = 0
        print("Validating...")
        words = None
        with torch.no_grad():
            rand_word_batch = int(np.random.randint(0,len(val_loader)))
            for b,(x,y) in enumerate(val_loader):
                torch.cuda.empty_cache()
                if model.transformer_type == models.AUTOENCODER:
                    targs = x.data[:,1:]
                    y = x.data
                elif model.transformer_type == models.DICTIONARY:
                    targs = x.data[:,1:]
                    emb_targs = model.embeddings(y.to(DEVICE))
                    if b == rand_word_batch or hyps['exp_name']=="test":
                        words = [idx2word[y.squeeze()[i].item()] for\
                                         i in range(len(y.squeeze()))]
                    y = x.data
                else:
                    targs = y.data[:,1:]
                og_shape = targs.shape

                if hyps['init_decs']:
                    y = train_data.inits.clone().repeat(len(x),1)
                if hyps['masking_task']:
                    x,y,mask = mask_words(x, y, mask_p=hyps['mask_p'])
                y = y[:,:-1]
                preds = model(x.to(DEVICE),y.to(DEVICE))

                tot_loss = 0
                if model.transformer_type == models.DICTIONARY:
                    emb_preds = preds[1]
                    preds = preds[0]
                    emb_loss = F.mse_loss(emb_preds,emb_targs)
                    avg_emb_loss += emb_loss.item()

                if hyps['masking_task']:
                    # Mask loss and acc
                    targs = targs.reshape(-1)
                    preds = preds.reshape(-1,preds.shape[-1])
                    mask = mask.reshape(-1).bool()
                    idxs = torch.arange(len(mask))[mask]
                    mask_preds = preds[idxs]
                    mask_targs = targs[idxs]
                    mask_loss = lossfxn(mask_preds,mask_targs)
                    mask_avg_loss += mask_loss.item()
                    mask_preds = torch.argmax(mask_preds,dim=-1)
                    mask_acc = (mask_preds==mask_targs).sum().float()
                    mask_acc = mask_acc/mask_preds.numel()
                    mask_avg_acc += mask_acc.item()
                else:
                    mask_acc = torch.zeros(1).to(DEVICE)
                    mask_loss = torch.zeros(1).to(DEVICE)

                # Tot loss and acc
                preds = preds.reshape(-1,preds.shape[-1])
                targs = targs.reshape(-1).to(DEVICE)
                if not hyps['masking_task']:
                    bitmask = (targs!=mask_idx)
                    loss = lossfxn(preds[bitmask], targs[bitmask])
                else:
                    loss = lossfxn(preds,targs)
                preds = torch.argmax(preds,dim=-1)
                sl = og_shape[-1]
                if not hyps['masking_task']:
                    eq = (preds==targs).float()
                    indy_acc = eq[bitmask].mean()
                    eq[~bitmask] = 1
                    eq = eq.reshape(og_shape)
                    acc = (eq.sum(-1)==sl).float().mean()
                else:
                    eq = (preds==targs).float().reshape(og_shape)
                    acc = (eq.sum(-1)==sl).float().mean()
                    indy_acc = eq.mean()
                preds = preds.cpu()

                if b == rand_word_batch or hyps['exp_name']=="test":
                    rand = int(np.random.randint(0,len(x)))
                    question = x[rand]
                    whr = torch.where(question==mask_idx)[0]
                    endx=len(question) if len(whr)==0 else whr[0].item()
                    question = question[:endx]
                    pred_samp = preds.reshape(og_shape)[rand]
                    targ_samp = targs.reshape(og_shape)[rand]
                    whr = torch.where(targ_samp==mask_idx)[0]
                    endx=len(targ_samp) if len(whr)==0 else whr[0].item()
                    targ_samp = targ_samp[:endx]
                    pred_samp = pred_samp[:endx]
                    idx2word = train_data.idx2word
                    question =  [idx2word[p.item()] for p in question]
                    pred_samp = [idx2word[p.item()] for p in pred_samp]
                    targ_samp = [idx2word[p.item()] for p in targ_samp]
                    question = " ".join(question)
                    pred_samp = " ".join(pred_samp)
                    targ_samp = " ".join(targ_samp)
                    if words is not None:
                        word_samp = str(words[rand])

                avg_acc += acc.item()
                avg_indy_acc += indy_acc.item()
                avg_loss += loss.item()
                if hyps["masking_task"]:
                    s = "Mask Loss:{:.5f} | Acc:{:.5f} | {:.0f}%"
                    s = s.format(mask_loss.item(), mask_acc.item(),
                                                   b/len(val_loader)*100)
                elif model.transformer_type == models.DICTIONARY:
                    s = "Loss:{:.5f} | Acc:{:.5f} | Emb:{:.5f} | {:.0f}%"
                    s = s.format(loss.item(),acc.item(),emb_loss.item(),
                                              b/len(val_loader)*100)
                else:
                    s = "Loss:{:.5f} | Acc:{:.5f} | {:.0f}%"
                    s = s.format(loss.item(), acc.item(),
                                              b/len(val_loader)*100)
                print(s, end=len(s)*" " + "\r")
                if hyps['exp_name']=="test" and b > 5: break
        print()
        del targs
        del x
        del y
        del eq
        if model.transformer_type == models.DICTIONARY:
            del emb_targs
        torch.cuda.empty_cache()
        mask_val_loss = mask_avg_loss/len(val_loader)
        mask_val_acc = mask_avg_acc/  len(val_loader)
        val_avg_loss = avg_loss/len(val_loader)
        val_avg_acc = avg_acc/len(val_loader)
        val_emb_loss = avg_emb_loss/len(val_loader)
        scheduler.step(val_avg_acc)
        val_avg_indy = avg_indy_acc/len(val_loader)
        stats_string += "Val - Loss:{:.5f} | Acc:{:.5f} | Indy:{:.5f}\n"
        stats_string = stats_string.format(val_avg_loss,val_avg_acc,
                                                        val_avg_indy)
        if hyps['masking_task']:
            stats_string+="Val Mask Loss:{:.5f} | Val Mask Acc:{:.5f}\n"
            stats_string=stats_string.format(mask_avg_loss,mask_avg_acc)
        elif model.transformer_type==models.DICTIONARY:
            stats_string+="Val Emb Loss:{:.5f}\n"
            stats_string = stats_string.format(val_emb_loss)
        if words is not None:
            stats_string += "Word: " + word_samp + "\n"
        stats_string += "Quest: " + question + "\n"
        stats_string += "Targ: " + targ_samp + "\n"
        stats_string += "Pred: " + pred_samp + "\n"
        optimizer.zero_grad()

        save_dict = {
            "epoch":epoch,
            "hyps":hyps,
            "train_loss":train_avg_loss,
            "train_acc":train_avg_acc,
            "train_indy":train_avg_indy,
            "mask_train_loss":mask_train_loss,
            "mask_train_acc":mask_train_acc,
            "val_loss":val_avg_loss,
            "val_acc":val_avg_acc,
            "val_indy":val_avg_indy,
            "mask_val_loss":mask_val_loss,
            "mask_val_acc":mask_val_acc,
            "state_dict":model.state_dict(),
            "optim_dict":optimizer.state_dict(),
            "word2idx":train_data.word2idx,
            "idx2word":train_data.idx2word,
            "sampled_types":train_data.sampled_types
        }
        save_name = "checkpt"
        save_name = os.path.join(hyps['save_folder'],save_name)
        io.save_checkpt(save_dict, save_name, epoch, ext=".pt",
                                del_prev_sd=hyps['del_prev_sd'])
        stats_string += "Exec time: {}\n".format(time.time()-starttime)
        print(stats_string)
        s = "Epoch:{} | Model:{}\n".format(epoch, hyps['save_folder'])
        stats_string = s + stats_string
        log_file = os.path.join(hyps['save_folder'],"training_log.txt")
        with open(log_file,'a') as f:
            f.write(str(stats_string)+'\n')
    del save_dict['state_dict']
    del save_dict['optim_dict']
    del save_dict['hyps']
    save_dict['save_folder'] = hyps['save_folder']
    return save_dict

def save_data_structs(hyps, structs):
    """
    Records a copy of the data structs that were used to create this
    dataset

    structs: list of samp_structs
        see the datas.WordProblem class for details on sample structs
    """
    sf = hyps['save_folder']
    if not os.path.exists(sf):
        os.mkdir(sf)
    h = "samp_structs.json"
    with open(os.path.join(sf,h),'w') as f:
        json.dump(structs, f)


def mask_words(x,y,mask_p=.15,mask_idx=0):
    """
    x: Long tensor (..., S)
        a torch tensor of token indices
    y: Long tensor (..., S)
        a torch tensor of token indices in which the sequence is
        offset forward by 1 position
    mask_p: float [0,1]
        the probability of masking a word
    mask_idx: int
        the index of the mask token
    """
    if mask_p == 0:
        return x,y,torch.zeros(x.shape).bool()
    probs = torch.rand(x.shape)
    mask = (probs<mask_p).bool()
    x[mask] = mask_idx
    postpender = torch.zeros(*x.shape[:-1],1).bool()
    mask = torch.cat([mask,postpender],dim=-1)[...,1:].bool()
    y[mask] = mask_idx
    return x,y,mask


