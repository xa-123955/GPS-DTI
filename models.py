import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from transformers import AutoTokenizer, EsmModel
from GPS import GPS
# Use CUDA if available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DynamicMaxPool1d(nn.Module):
    def __init__(self):
        super(DynamicMaxPool1d, self).__init__()
    def forward(self, x, pool_size):
        return F.max_pool1d(x, kernel_size=pool_size)

def edge_encoder_cls(_):
            def zero(_):
                return 0
            return zero

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class GPSDTI(nn.Module):
    def __init__(self, **config):
        super(GPSDTI, self).__init__()
        mlp_in_dim = config["DECODER"]["IN_DIM"]#256
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]#512
        mlp_out_dim = config["DECODER"]["OUT_DIM"]#128
        out_binary = config["DECODER"]["BINARY"]#2
        attn_kwargs = {'dropout': 0.5}
        self.gps = GPS(channels=128, pe_dim=8, num_layers=10, attn_type='multihead', attn_kwargs=attn_kwargs).to(device)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
        self.tokenizer = AutoTokenizer.from_pretrained("")   
        self.model = EsmModel.from_pretrained("").to(device)
        self.conv1 = nn.Conv1d(in_channels=640, out_channels=512, kernel_size=7,padding='same')
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5,padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3,padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.Drug_max_pool = DynamicMaxPool1d()
        self.mix_attention_layer = nn.MultiheadAttention(128, 4)
        self.Protein_max_pool = nn.MaxPool1d(1000)
        self.dropout1 = nn.Dropout(0.1)

    def dgl_split(self, bg, feats):
        max_num_nodes = int(bg.batch_num_nodes().max())
        batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in zip(bg.batch_num_nodes(), range(bg.batch_size))],
                       dim=1).reshape(-1).type(torch.long).to(bg.device)
        cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
        return out

    def forward(self, bg_d, v_p, mode="train"):
        # Extract graph data for drug
        x =bg_d.ndata['atom']
        pe=bg_d.ndata['lap_pos_enc']
        edge_index = torch.stack((bg_d.edges()[0], bg_d.edges()[1]), 0)
        edge_attr=bg_d.edata['bond']
        batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in zip(bg_d.batch_num_nodes(), range(bg_d.batch_size))],
                       dim=1).reshape(-1).type(torch.long).to(bg_d.device)
        # GPS model forward pass for drug features
        v_d = self.gps(x,pe,edge_index,edge_attr,batch)
        v_d = self.dgl_split(bg_d, v_d)
        # Pretrained protein model (ESM) to get protein embeddings
        with torch.no_grad():
            inputs = self.tokenizer(v_p,max_length=1000, padding="max_length",return_tensors="pt").to(device)
            outputs = self.model(**inputs)
        word_vectors = outputs['last_hidden_state']
        v=word_vectors.permute(0,2,1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v_p = v.permute(0,2,1)
        drugConv = v_d.permute(0, 2, 1)
        proteinConv = v_p.permute(0, 2, 1)
        drug_QKV = drugConv.permute(2, 0, 1)
        protein_QKV = proteinConv.permute(2, 0, 1)
        drug_att, _ = self.mix_attention_layer(drug_QKV, protein_QKV, protein_QKV)
        protein_att, _ = self.mix_attention_layer(protein_QKV, drug_QKV, drug_QKV)
        drug_att = drug_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)
        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5
        drugConv = self.Drug_max_pool(drugConv,drugConv.size(2)).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)
        att = torch.cat((drug_att, protein_att), dim=-1)
        f = torch.cat([drugConv, proteinConv], dim=1)
        f = self.dropout1(f)
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, f, score 


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

