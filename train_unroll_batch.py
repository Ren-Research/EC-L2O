import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from utils.preprocess import sliding_windows, load_power_shortage
from utils.loss import object_loss_cost, object_loss_cr
from utils.model import LSTM_unroll
from utils.dataset import TrajectCR_Dataset
from tqdm import tqdm
import json 
import argparse

n_iter = 0
n_iter_val = 0
use_cuda = False

def train_cr(ml_model, optimizer, writer, train_dataloader, demand_validation, 
            num_epoch, switch_weight, min_cr, mtl_weight = 0.5, mute=True):

    global n_iter, n_iter_val, use_cuda
    
    if not mute:
        epoch_iter = tqdm(range(num_epoch))
    else:
        epoch_iter = range(num_epoch)
    
    for _ in epoch_iter:
        ml_model.train()
        for _, (demand,opt_cost) in enumerate(train_dataloader):
            demand = demand.float()
            if use_cuda: 
                demand = demand.cuda()
                opt_cost = opt_cost.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            action_ml = ml_model(demand, calib = False)
            
            
            if mtl_weight == 1.0:
                loss_calib = torch.zeros((1,1))
                
                loss_ml = object_loss_cost(demand, action_ml, c=switch_weight)
                # loss_ml = object_loss_cr(demand, action_ml, opt_cost, min_cr = min_cr, c=switch_weight)
                loss = loss_ml

            else:
                loss_ml = object_loss_cr(demand, action_ml, opt_cost, min_cr = min_cr, c=switch_weight)
                
                action_calib = ml_model(demand, calib = True)
                loss_calib = object_loss_cost(demand, action_calib, c = switch_weight)
                
                loss = mtl_weight*loss_ml + (1-mtl_weight)*loss_calib

            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss_train/no_calib', loss_ml.item(), n_iter)
            writer.add_scalar('Loss_train/with_calib', loss_calib.item(), n_iter)
            writer.add_scalar('Loss_train/overall', loss.item(), n_iter)
            n_iter += 1

        writer.flush()
        
        # Calculate evaluation cost
        ml_model.eval()

        with torch.no_grad():
            action_val_ml    = ml_model(demand_validation, mode="val", calib=False)
            action_val_calib = ml_model(demand_validation, mode="val", calib=True)

            loss_val_ml = object_loss_cost(demand_validation, action_val_ml, c = switch_weight)
            loss_val_calib = object_loss_cost(demand_validation, action_val_calib, c = switch_weight)

        writer.add_scalar('Loss_val/no_calib', loss_val_ml.item()/100, n_iter_val)
        writer.add_scalar('Loss_val/with_calib', loss_val_calib.item()/100, n_iter_val)
        n_iter_val += 1

    writer.close()

def single_experiment(writer, w, l_1, l_2, l_3, mtl_weight, min_cr, 
                        epoch_num, lr_list, batch_size, mute=True, 
                        csv_file = "data/solar_2015.csv"):
    
    global use_cuda, n_iter, n_iter_val
    
    n_iter = 0
    n_iter_val = 0
    print("Parameters")
    print("     w     l_1     l_2     l_3     mtl     ")
    print("  {:.3f}   {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(w, l_1, l_2, l_3, mtl_weight))

    hidden_size = 10
    num_classes = 1

    input_size = 2 * num_classes
    seq_length = 25
    num_layers = 3
    
    # df_header = pd.read_csv(csv_file, nrows=1) ## general information (e.g. time zone, elevation)
    df= pd.read_csv(csv_file, header = 2)

    data_raw = load_power_shortage(df)

    n_trian_step=24*60
    n_val_step=24*30
    # n_test_step=24*60

    ## Splitting training and testing dataset
    data_raw = data_raw.reshape([-1,1])
    train_raw=data_raw[:n_trian_step, :]
    val_raw=data_raw[n_trian_step:n_trian_step+n_val_step, :]
    
    train_seq = sliding_windows(train_raw, seq_length)
    
    
    traject_dataset_train = TrajectCR_Dataset(train_seq, w, mute=mute)
    train_dataloader = DataLoader(traject_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    val_seq = val_raw.reshape([1,-1,1])
    val_seq_tensor = torch.from_numpy(val_seq).float()
    if use_cuda: val_seq_tensor = val_seq_tensor.cuda()

    lstm = LSTM_unroll(num_classes, input_size, hidden_size, num_layers, 
                            seq_length, w, l_1, l_2, l_3)
        
    optimizer = optim.Adam(lstm.parameters(), lr=lr_list[0])
    if use_cuda: lstm = lstm.cuda()

    for lr in lr_list:
        train_cr(lstm, optimizer, writer, train_dataloader, val_seq_tensor, 
            epoch_num, w, min_cr, mtl_weight = mtl_weight, mute=mute)
        optimizer.param_groups[0]["lr"] = lr
    
    pth_path = writer.get_logdir() + "lstm_unroll.pth"
    torch.save(lstm.state_dict(), pth_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a L2O Model')
    parser.add_argument('config', help='train config file path')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with  open(args.config, "r") as f:
        config_data = json.load(f)

    # Problem definition
    w = config_data["w"]
    l_1 = config_data["l_1"]
    l_2 = config_data["l_2"]
    l_3 = config_data["l_3"]
    min_cr = config_data["min_cr"]

    # Traing parameters
    epoch_num = config_data["epoch_num"]
    lr_list = config_data["lr_list"]
    batch_size = config_data["batch_size"]
    
    # Experiment parameters
    base_log_dir = config_data["base_log_dir"]
    mtl_list = np.array(config_data["mtl_list"])

    for mtl_weight in mtl_list:
        writer_path = base_log_dir + "/mtl_{:.2f}/".format(mtl_weight)
        writer = SummaryWriter(writer_path)
        single_experiment(writer, w, l_1, l_2, l_3, mtl_weight, min_cr, 
                    epoch_num, lr_list, batch_size, mute=False)
    
    print('Finished Training')