import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from torch.nn.utils.rnn import pad_sequence
from tape import ProteinBertModel, TAPETokenizer  # Import TAPE model and tokenizer
import torch.nn.functional as F
from myutils import *
max_length = 64
huadong=48
num_classes = 1  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size =1

##################################################
def get_seqs_labels(dataset_path:str):

    seqs = []
    labels = []
    with open(dataset_path) as f:
        lines = f.readlines()
        group_num = len(lines) // 3
        assert group_num * 3 == len(lines)
        for i in range(group_num):
            seqs.append(lines[3 * i + 1].strip())
            labels.append(lines[3 * i + 2].strip())
    return seqs, labels
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


##################################################
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
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # Einstein求和约定
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.heads * self.head_dim) # 拼接多头输出
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

        self.pos_encoder = PositionalEncoding(hidden_dim).to(device)  #位置编码
        self.encoder = Encoder(hidden_dim, num_layers, num_heads, forward_expansion, dropout, max_length).to(device)
        # self.fc_out = nn.Linear(embed_dim, num_classes).to(device)  
        self.mlp = MLP(embed_dim, hidden_dim, num_classes).to(device)
        self.fc_out = nn.Linear(hidden_dim, num_classes).to(device)
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

def convert_to_list_of_tensors(data: np.ndarray):
    return [torch.tensor(row, dtype=torch.float32).unsqueeze(0) for row in data]

def find_regions(label_arr, label_value=1):
    regions = []
    start = None
    for i, val in enumerate(label_arr):
        if val == label_value:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i - 1
                regions.append((start, end, end - start + 1))
                start = None
    if start is not None:
        end = len(label_arr) - 1
        regions.append((start, end, end - start + 1))
    return regions
def filter_single_residues(pred_label_arr):
    filtered_arr = np.copy(pred_label_arr)
    seq_len = len(filtered_arr)
    i = 0
    
    while i < seq_len:
        if filtered_arr[i] == 1:
            start = i
            while i < seq_len and filtered_arr[i] == 1:
                i += 1
            end = i - 1
            region_length = end - start + 1
            if region_length == 1:
                filtered_arr[start] = 0
        else:
            i += 1
    
    return filtered_arr
def find_overlapping_regions(true_arr, pred_arr):
    overlapping = []
    start = None
    seq_len = len(true_arr)
    
    for i in range(seq_len):
        if true_arr[i] == 1 and pred_arr[i] == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i - 1
                overlapping.append((start, end, end - start + 1))
                start = None
    
    if start is not None:
        end = seq_len - 1
        overlapping.append((start, end, end - start + 1))
    
    return overlapping

def find_fn_regions(true_arr, pred_arr):
    fn_regions = []
    start = None
    seq_len = len(true_arr)
    
    for i in range(seq_len):
        if true_arr[i] == 1 and pred_arr[i] == 0:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i - 1
                fn_regions.append((start, end, end - start + 1))
                start = None
    
    if start is not None:
        end = seq_len - 1
        fn_regions.append((start, end, end - start + 1))
    
    return fn_regions

def find_fp_regions(true_arr, pred_arr):
    fp_regions = []
    start = None
    seq_len = len(true_arr)
    
    for i in range(seq_len):
        if true_arr[i] == 0 and pred_arr[i] == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i - 1
                fp_regions.append((start, end, end - start + 1))
                start = None
    
    if start is not None:
        end = seq_len - 1
        fp_regions.append((start, end, end - start + 1))
    
    return fp_regions

def find_sequence_position(sequence_file, target_seq):
    with open(sequence_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]  
    
    total_residues = 0  
    seq_index = 0      
    
    for i in range(0, len(lines), 3):  
        current_seq = lines[i+1]  
        if current_seq == target_seq:
            if seq_index == 0:
                return 0, len(target_seq)-1, len(target_seq)
            else:
                return total_residues, total_residues + len(target_seq)-1, len(target_seq)
        total_residues += len(current_seq)
        seq_index += 1
    
    raise ValueError(f"未找到目标序列：{target_seq[:20]}...")
def merge_and_convert2int(labels):
    res = []
    for label in labels:
        res.extend([int(item != "0") for item in label])
    return res


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequence_file_path = "/root/autodl-tmp/MORF/Transformer/raw_seqs/EXP53.txt"
    dataset_dir = get_sub_hj_dir(sequence_file_path) 
    seqs, labels = get_seqs_labels(dataset_dir)
    labels = merge_and_convert2int(labels)
    test_data_path = "/root/autodl-tmp/MORF/Transformer/dataset/test3.txt"
    true_label_path = "/root/autodl-tmp/MORF/Transformer/dataset/X3_label.txt"
    model_path = "Model_parameter.pth" 
    target_sequence = "MAQEQTKRGGGGGDDDDIAGSTAAGQERREKLTEETDDLLDEIDDVLEENAEDFVRAYVQKGGQ"  
    residue_start_row, residue_end_row, seq_length = find_sequence_position(
        sequence_file_path, 
        target_sequence
    )
    print(f"找到目标序列: 长度={seq_length}, 起始位置={residue_start_row}, 结束位置={residue_end_row}")
    test_encodings = np.loadtxt(test_data_path)
    test_encoding = test_encodings[residue_start_row:residue_end_row+1, :]
    test_encoding = convert_to_list_of_tensors(test_encoding) 
    
    true_labels = np.loadtxt(true_label_path)
    test_label = true_labels[residue_start_row:residue_end_row+1]
    test_label = test_label.tolist()
    seq_len = len(test_label)
    test_dataset = ProteinDataset(test_encoding, test_label)  # 假设已定义
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = ProteinTransformer(
        embed_dim= test_encoding[0].shape[0] * test_encoding[0].shape[1],
        num_heads=2,
        num_layers=1,
        forward_expansion=2,
        dropout=0.5,
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit()

    def predict(model, dataloader):
        model.eval()
        all_true = []
        all_pred_probs = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                a, b, c = inputs.shape
                inputs_reshaped = inputs.reshape(a, 1, b * c).to(device)
                
                outputs = model(inputs_reshaped)
                outputs = outputs.view(-1)
                pred_probs = torch.sigmoid(outputs).cpu().numpy()
                
                all_true.extend(labels.cpu().numpy())
                all_pred_probs.extend(pred_probs)
        return np.array(all_true), np.array(all_pred_probs)

    
    true_label_arr, pred_prob_arr = predict(model, test_loader)
    pred_label_arr = (pred_prob_arr >= 0.47).astype(int)
    pred_label_arr = filter_single_residues(pred_label_arr)
    auc=roc_auc_score(true_label_arr, pred_prob_arr)
    recall = recall_score(true_label_arr, pred_label_arr, zero_division=1)

    TP = np.sum((true_label_arr == 1) & (pred_label_arr == 1))
    FP = np.sum((true_label_arr == 0) & (pred_label_arr == 1))
    TN = np.sum((true_label_arr == 0) & (pred_label_arr == 0))
    FN = np.sum((true_label_arr == 1) & (pred_label_arr == 0))

    TPR = recall  
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0

    print("\n评估指标:")
    print(f"Recall (TPR):    {recall:.4f}")
    print(f"FPR:    {FPR:.4f}")
    print(f"AUC:  {auc:.4f}")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

 
    tp_regions = find_overlapping_regions(true_label_arr, pred_label_arr)  
    fn_regions = find_fn_regions(true_label_arr, pred_label_arr)          
    fp_regions = find_fp_regions(true_label_arr, pred_label_arr)         
    
    total_tp_residues = sum(region[2] for region in tp_regions)
    total_fp_residues = sum(region[2] for region in fp_regions)
    total_fn_residues = sum(region[2] for region in fn_regions)
    
    print(f"\n区域统计:")
    print(f"TP区域(红蓝重合): {len(tp_regions)} 个区域, 共 {total_tp_residues} 个残基")
    print(f"FP区域(只有蓝线): {len(fp_regions)} 个区域, 共 {total_fp_residues} 个残基")
    print(f"FN区域(只有红线): {len(fn_regions)} 个区域, 共 {total_fn_residues} 个残基")
    plt.figure(figsize=(15, 8))
    plt.plot(range(seq_len), true_label_arr, 'r-', label='True Labels', linewidth=2)
    plt.plot(range(seq_len), pred_label_arr, 'b--', label='Predicted Labels', linewidth=2)
    plt.title('MoRFs Prediction: True vs Predicted Labels')  
    plt.ylabel('Label (1 = MoRF, 0 = Non-MoRF)')
    plt.xlabel('Position (Residue Index)')
    plt.ylim(-0.1, 1.2)  
    plt.legend(loc='upper left')  

    plt.tight_layout()
    plt.savefig('MoRFs_prediction_simple.png', dpi=300, bbox_inches='tight')
    plt.show()