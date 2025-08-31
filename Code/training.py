import torch
import random
import time
import torch.nn as nn
from torch_geometric.loader import DataLoader
from helpers import rmse, pearson, model_dict
from utils import GraphDataset, init_weights
import os
import pandas as pd
import argparse
import numpy as np
import pickle

# Change: Add more functions
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def predict(model, device, loader, y_scaler=None):
    model.eval()

    # Change: total_bind_logits is added
    total_pKd_preds = torch.Tensor()
    total_bind_logits = torch.Tensor()
    total_labels = torch.Tensor()
    
    # total_preds = torch.Tensor()
    # total_labels = torch.Tensor()

    
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Change: Change into two output nodes 
            pred_pKd, pred_logits = model(data) 
            
            # output = model(data)

            
            # total_preds = torch.cat((total_preds, output.cpu()), 0)
            # total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
           
            # Change: Add total_bind_logits
            total_pKd_preds = torch.cat((total_pKd_preds, pred_pKd.cpu()), 0)
            total_bind_logits = torch.cat((total_bind_logits, pred_logits.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0) 

            

     # Change: Return it to original labels 
    if y_scaler:
        true_pKd = y_scaler.inverse_transform(total_labels.numpy())
        pred_pKd = y_scaler.inverse_transform(total_pKd_preds.numpy())
    else:
        true_pKd = total_labels.numpy()
        pred_pKd = total_pKd_preds.numpy()
    

    # Change: Construct the true labels 
    true_bind = (true_pKd >= 4.0).astype(float)


    # Change: Using sigmoid
    pred_prob = torch.sigmoid(total_bind_logits).numpy()


    # Change: Return four values 
    return {
        "true_pKd": true_pKd.flatten(),
        "pred_pKd": pred_pKd.flatten(),
        "true_bind": true_bind.flatten(),
        "pred_prob": pred_prob.flatten()
    }
    
    # return y_scaler.inverse_transform(total_labels.numpy().flatten().reshape(-1,1)).flatten(), y_scaler.inverse_transform(total_preds.detach().numpy().flatten().reshape(-1,1)).flatten()

# Change: Add y_scaler
def train(model, device, train_loader, optimizer, epoch, loss_fn, y_scaler):

# def train(model, device, train_loader, optimizer, epoch, loss_fn):
    log_interval = 100
    model.train()
    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        
        # Change: Change into two output nodes
        pred_pKd, pred_logits = model(data)

        # Change: Calculated the regression loss  
        y_scaled = data.y.view(-1, 1).to(device)

        if y_scaler is not None:
            y_unscaled = torch.from_numpy(
                y_scaler.inverse_transform(y_scaled.detach().cpu().numpy())
            ).to(device)
        else:
            y_unscaled = y_scaled

        bind_label = (y_unscaled >= 4.0).float()

     
        # Change: Calculated the regression loss
      # y_true = data.y.view(-1, 1).to(device)
      # bind_label = (y_true >= 4.0).float()

#       loss_reg = loss_fn(pred_pKd, y_true)  

        # Change: Calculated the regression loss
        loss_reg = loss_fn(pred_pKd, y_scaled)  




        # Change: Calculated the classification loss
        loss_cls = nn.BCEWithLogitsLoss()(pred_logits, bind_label)  

        # Change: Calculated the total loss
        loss =loss_reg +  loss_cls


        
        loss.backward()
        optimizer.step()
        total_loss += (loss.item()*len(data.y))
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]'.format(epoch,
                                                             batch_idx * len(data.y),
                                                             len(train_loader.dataset),
                                                             100. * batch_idx / len(train_loader)))

    # Change: Change the print information 
    print(f"Epoch {epoch} Summary → Total Loss: {total_loss/len(train_loader.dataset):.4f} "
      f"| Last Reg Loss: {loss_reg.item():.4f} | Last Cls Loss: {loss_cls.item():.4f}")

    # print("Loss for epoch {}: {:.4f}".format(epoch, total_loss/len(train_loader.dataset)))
    return total_loss/len(train_loader.dataset)



def _train(model, device, loss_fn, train_loader, valid_loader, optimizer, n_epochs, y_scaler, model_output_dir, model_file_name):
    best_pc = -1.1
    pcs = []
    for epoch in range(n_epochs):
        # Change: Add y_scaler 
        _ = train(model, device, train_loader, optimizer, epoch + 1, loss_fn, y_scaler)

     #  _ = train(model, device, train_loader, optimizer, epoch + 1, loss_fn)
        
        # G, P = predict(model, device, valid_loader, y_scaler)
         # Change: Change into results form 
        results = predict(model, device, valid_loader, y_scaler)

        # true_pKd = results["true_pKd"]
        # pred_pKd = results["pred_pKd"]
        # true_cls = results["true_bind"]
        # pred_prob = results["pred_prob"]

        

        # current_pc = pearson(G, P)
        # pcs.append(current_pc)
        
        # Change: Regression evaluation
        current_pc = pearson(results["true_pKd"], results["pred_pKd"])
        pcs.append(current_pc)

        # Change: classification evaluation
        auc = roc_auc_score(results["true_bind"], results["pred_prob"])

        
        low = np.maximum(epoch-7,0)
        avg_pc = np.mean(pcs[low:epoch+1])
        if(avg_pc > best_pc):
            torch.save(model.state_dict(), os.path.join(model_output_dir, model_file_name))
            best_pc = avg_pc  
            
        # print('The current validation set Pearson correlation:', current_pc)
    # return
        #  Change: Print the results
        print(f"Epoch {epoch+1} → Validation Pearson: {current_pc:.4f} | AUC: {auc:.4f}")

    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GATv2Net')
    parser.add_argument('--dataset', type=str, default='pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark') # Change: Change into correct name 
    parser.add_argument('--batch_size', type=int, default=32)  # Change: Change the default batch_size into 32 for this project 
    parser.add_argument('--epochs', type=int, default=50) # Change: Change the default epochs into 50 for this project 
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--head', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00012291937615434127)
    parser.add_argument('--activation_function', type=str, default='leaky_relu')



    # Change: Add more arguments 
    parser.add_argument('--reg_loss_weight', type=float, default=1.0, help='Weight for pKd regression loss')
    parser.add_argument('--cls_loss_weight', type=float, default=1.0, help='Weight for classification loss')
    parser.add_argument('--cls_threshold', type=float, default=4.0, help='pKd threshold for bind classification')


    
    
    args = parser.parse_args()
    return args


def train_NN(args):
    modeling = model_dict[args.model]
    model_st = modeling.__name__
    
    batch_size = args.batch_size
    LR = args.lr
    n_epochs = args.epochs

    print('Train for {} epochs: '.format(n_epochs))

    dataset = args.dataset

    print('Running dataset {} on model {}.'.format(dataset, model_st))
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_output_dir = os.path.join("output", "trained_models")
    os.makedirs(model_output_dir, exist_ok=True) # Change: Add os.makedirs


    print("💡 Training Data：", dataset+'_train')

    train_data = GraphDataset(root='data', dataset=dataset+'_train', y_scaler=None)
    valid_data = GraphDataset(root='data', dataset=dataset+'_valid', y_scaler=train_data.y_scaler)
    test_data = GraphDataset(root='data', dataset=dataset+'_test', y_scaler=train_data.y_scaler)

    seeds = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]
    for i,seed in enumerate(seeds):
        random.seed(seed)
        torch.manual_seed(int(seed))
        
        model_file_name = timestr + '_model_' + model_st + '_' + dataset + '_' + str(i) + '.model'

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        

        if(torch.cuda.is_available()):
            print("GPU is available")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print('Device state:', device)

        

        model = modeling(node_feature_dim=train_data.num_node_features, edge_feature_dim=train_data.num_edge_features, config=args)
        model.apply(init_weights)
    
        print("The number of node features is ", train_data.num_node_features)
        print("The number of edge features is ", train_data.num_edge_features)
    
        weight_decay = 0
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    
        model.to(device)
        _train(model, device, loss_fn, train_loader, valid_loader, optimizer, n_epochs, train_data.y_scaler, model_output_dir, model_file_name)
        
        model.load_state_dict(torch.load(os.path.join(model_output_dir, model_file_name)))



        
        
        # G_test, P_test = predict(model, device, test_loader, train_data.y_scaler)
        # Change: Change into results form 
        results = predict(model, device, test_loader, train_data.y_scaler)
        G_test, P_test = results["true_pKd"], results["pred_pKd"]

        if(i == 0):
            df_test = pd.DataFrame(data=G_test, index=range(len(G_test)), columns=['truth'])
        
        col = 'preds_' + str(i)
        df_test[col] = P_test
    
    df_test['preds'] = df_test.iloc[:,1:].mean(axis=1)






#    scaler_file = timestr + '_model_' + model_st + '_' + dataset + '.pickle'
# Change: Change scaler_file
    scaler_file = os.path.join(model_output_dir, timestr + '_model_' + model_st + '_' + dataset + '.pickle')

    with open(scaler_file, 'wb') as f:
#    with open(scaler_file,'wb') as f:
        pickle.dump(train_data.y_scaler, f)
    
    test_preds = np.array(df_test['preds'])
    test_truth = np.array(df_test['truth'])
    test_ens_pc = pearson(test_truth, test_preds)
    test_ens_rmse = rmse(test_truth, test_preds)
    print("Ensemble test PC:", test_ens_pc)
    print("Ensemble test RMSE:", test_ens_rmse)

    # Change: Add test_auc
    test_auc = roc_auc_score(results["true_bind"], results["pred_prob"])
    print("Ensemble Test AUC:", test_auc)  

     
    # Change: Change the results into DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(model_output_dir, "test_results.csv"), index=False)

if __name__ == "__main__":
    start_time = time.time()
    
    args = parse_args()
    
    train_NN(args)
    
    print("Total time is %s seconds" % (time.time() - start_time))
    
    