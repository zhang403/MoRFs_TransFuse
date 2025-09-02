import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from torch.nn.utils.rnn import pad_sequence
from tape import ProteinBertModel, TAPETokenizer  # Import TAPE model and tokenizer
import torch.nn.functional as F
from myutils import *
import pickle
from sklearn.metrics import roc_curve


max_length = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1024
num_classes = 1  
# 多层感知机 (MLP) 类定义
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, output_dim) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x)) 
        x = self.relu(self.fc2(x))  
        x = self.fc3(x)
        return x
# 蛋白质数据集类定义
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  
        self.labels = labels  

    def __len__(self):
        return len(self.sequences) 

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]  


class PositionalEncoding(nn.Module):

    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None 

    def forward(self, x):
        seq_len = x.size(0)  
        if self.pe is None or self.pe.size(0) < seq_len:
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))

            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            pe[:, 0::2] = torch.sin(position * div_term) 
            pe[:, 1::2] = torch.cos(position * div_term) 
            self.pe = pe.unsqueeze(0).transpose(0, 1)  

        return x + self.pe[:seq_len, :] 

class SelfAttention(nn.Module):

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads 
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads 嵌入大小需要被头部整除"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size).to(device)  

    def forward(self, values, keys, query):
        N = query.shape[0] 
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim).to(device)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim).to(device)
        queries = query.reshape(N, query_len, self.heads, self.head_dim).to(device)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads).to(device)
        self.norm1 = nn.LayerNorm(embed_size).to(device)
        self.norm2 = nn.LayerNorm(embed_size).to(device)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size).to(device),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size).to(device)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x)) 
        return out

class Encoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        ).to(device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, x, x)
        return x

class ProteinTransformer(nn.Module):
    def __init__(self, embed_dim=1351, num_heads=8, num_layers=6, forward_expansion=4, dropout=0.1, max_length=max_length, num_classes=num_classes, hidden_dim=256, augment_eps=0.05):
        super(ProteinTransformer, self).__init__()
        self.input_block = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-6),
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.hidden_block = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6)
            , nn.Dropout(dropout)
            , nn.Linear(hidden_dim, hidden_dim)
            , nn.LeakyReLU()
            , nn.LayerNorm(hidden_dim, eps=1e-6)
        )

        self.augment_eps=augment_eps

        self.pos_encoder = PositionalEncoding(256).to(device)  
        self.encoder = Encoder(256, num_layers, num_heads, forward_expansion, dropout, max_length).to(device)
        # self.fc_out = nn.Linear(embed_dim, num_classes).to(device)  
        self.mlp = MLP(embed_dim, hidden_dim, num_classes).to(device)
        self.fc_out = nn.Linear(256, num_classes).to(device)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        if self.training and self.augment_eps > 0:
            src = src + self.augment_eps * torch.randn_like(src)


        src_input=self.input_block(src)

        src_hidden = self.pos_encoder(src_input)
        src_finish=self.hidden_block(src_hidden)

        output = self.encoder(src_finish)
        output = self.fc_out(output)
        # output = self.mlp(output)
        return output

def test_model(model, test_loader):
    model.eval()
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            outputs = model(inputs)
            probs = torch.sigmoid(outputs).view(-1)

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())

    true_labels = np.concatenate(all_labels)
    pred_scores = np.concatenate(all_outputs)
    return true_labels, pred_scores
def convert_to_list_of_tensors(data: np.ndarray):
    return [torch.tensor(row, dtype=torch.float32).unsqueeze(0) for row in data]
def save_roc_data(true_labels, pred_score, save_path):
    try:
        with open(save_path, "wb") as f:
            pickle.dump([true_labels, pred_score], f)
        print(f"ROC data saved to {save_path}")
    except Exception as e:
        print(f"Error saving ROC data: {e}")

###*********************************固定tpr*********************************
def get_fpr_at_fixed_tpr(true_labels, pred_scores, target_tprs=[0.2, 0.3, 0.4]):
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
    results = {}
    for target_tpr in target_tprs:
        idx = np.argmin(np.abs(tpr - target_tpr))
        achieved_tpr = tpr[idx]
        achieved_fpr = fpr[idx]
        threshold = thresholds[idx]
        
        results[target_tpr] = {
            "achieved_tpr": achieved_tpr,
            "fpr": achieved_fpr,
            "threshold": threshold
        }
        
        print(f"Target TPR: {target_tpr:.2f} → Achieved TPR: {achieved_tpr:.4f}, FPR: {achieved_fpr:.4f}, Threshold: {threshold:.4f}")
    
    return results

def save_fpr_results_to_txt(results, filename="fpr_results.txt"):
    with open(filename, 'w') as f:
        for target_tpr, metrics in results.items():
            line = (
                f"Target TPR: {target_tpr:.2f}\n"
                f"  Achieved TPR: {metrics['achieved_tpr']:.4f}\n"
                f"  FPR: {metrics['fpr']:.4f}\n"
                f"  Threshold: {metrics['threshold']:.4f}\n"
                "-----------------------------\n"
            )
            f.write(line)
    print(f"FPR results saved to {filename}")
##*************************************************************
if __name__ == "__main__":
    test1_path = "/root/autodl-tmp/MORF/Transformer/dataset/test1.txt"
    test2_path = "/root/autodl-tmp/MORF/Transformer/te/dataset/test2.txt"
    test1_encodings = np.loadtxt(test1_path) 
    test2_encodings = np.loadtxt(test2_path)
    test_encodings = np.vstack([test1_encodings, test2_encodings])
    test1_labels = np.loadtxt("/root/autodl-tmp/MORF/Transformer/dataset/X1_label.txt")  
    test2_labels = np.loadtxt("/root/autodl-tmp/MORF/Transformer/dataset/X2_label.txt") 
    test_labels = np.concatenate([test1_labels, test2_labels]) 
    
#     test_file = "/root/autodl-tmp/MORF/Transformer/dataset/test3.txt"
#     label_file = "/root/autodl-tmp/MORF/Transformer/dataset/X3_label.txt"
#     test_encodings = np.loadtxt(test_file)
#     test_labels = np.loadtxt(label_file)

    test_encodings = convert_to_list_of_tensors(test_encodings)
    test_labels = test_labels.tolist()
    test_dataset = ProteinDataset(test_encodings, test_labels)
    # test_dataset = BalanceProteinDataset(test_encodings, test_labels.tolist(), threshold=0, seq_len=64)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ProteinTransformer(
        embed_dim=test_encodings[0].shape[0] * test_encodings[0].shape[1], 
        num_heads=2,
        num_layers=1,
        forward_expansion=2,
        dropout=0.5,
    ).to(device)

    model_path = 'Model_parameter.pth'
    try:
        model.load_state_dict(torch.load(model_path))
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit()
    true_labels, pred_score = test_model(model, test_loader)
    auc_score = roc_auc_score(true_labels, pred_score)
    print(f"测试AUC值: {auc_score:.4f}")
    save_path = (
        "/root/autodl-tmp/MORF/Transformer/dataset/plot_data/Model_parameter_roc_data.pkl"
    )
    save_roc_data(true_labels, pred_score, save_path)
    target_tprs = [0.2, 0.3, 0.4]
    fpr_results = get_fpr_at_fixed_tpr(true_labels, pred_score, target_tprs)
    save_fpr_results_to_txt(fpr_results, filename="/root/autodl-tmp/MORF/Transformer/dataset/fpr/0.2-0.4.txt")
