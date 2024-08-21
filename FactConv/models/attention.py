import torch
import torch.nn as nn
import torch.nn.functional as F

import copy 

def resample(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            resample(m1)
        if isinstance(m1,nn.Conv2d):
            m1.resample()

class CrossAttentionNetwork(nn.Module):
    def __init__(self, net, num_samples=10, 
            nets=[], q=0, k=0, v=0, pre_processing="",
            post_processing="", num_heads=1):
        super(CrossAttentionNetwork, self).__init__()
        self.net_classifier = copy.deepcopy(net.linear)
        net.linear = nn.Identity()
        self.net = net
        self.nets = nets
        self.q = q
        self.k = k
        self.v = v

        q_shape=10 if q == 0 else 2048
        k_shape=10 if k == 0 else 2048
        v_shape=10 if v == 0 else 2048

        is_true = ("mean" in pre_processing or "mean" in post_processing or "sum" in pre_processing or "sum" in post_processing)

        self.pre_processing = pre_processing
        self.post_processing = post_processing

        self.num_samples = num_samples
        input_shape = q_shape * (num_samples if not is_true else 1)
        self.classifier = nn.Linear(input_shape, 10)

        self.attn =  nn.MultiheadAttention(q_shape, num_heads, kdim=k_shape, vdim=v_shape, batch_first=True)

    def forward(self, x):
        features=[]
        logits=[]

        with torch.no_grad():
            for i in range(0, self.num_samples):
                net = self.return_network(i)
                feature = net(x)
                logit = self.net_classifier(feature)
                features.append(feature)
                logits.append(logit)

        features = torch.stack(features, dim=1)
        logits = torch.stack(logits, dim=1)

        q = logits if self.q == 0 else features
        k = logits if self.k == 0 else features
        v = logits if self.v == 0 else features

        if "mean" in self.pre_processing:
            q = torch.mean(q, dim=1, keepdims=True)
        if "sum" in self.pre_processing:
            q = torch.sum(q, dim=1, keepdims=True)

        outputs = self.attn(q, k, v, need_weights=False)[0]

        if "mean" in self.pre_processing or "sum" in self.pre_processing:
            outputs = outputs.squeeze(1)

        if "mean" in self.post_processing:
            outputs = torch.mean(outputs, dim=1)
        if "sum" in self.post_processing:
            outputs = torch.sum(outputs, dim=1)

        if "linear" in self.post_processing:
            outputs = outputs.reshape(outputs.shape[0], -1)
            outputs = self.classifier(outputs)
        return outputs

    def return_network(self, i):
        if len(self.nets) == 0:
            self.resample()
            return self.net
        else:
            return self.nets[i]

    def resample(self):
        resample(self.net)



