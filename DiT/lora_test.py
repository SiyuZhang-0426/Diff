import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinearLoRALayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        merge=False,
        rank=8,
        lora_alpha=16,
        dropout=0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank
        self.linear = nn.Linear(in_features, out_features)
        if rank > 0:
            self.lora_a = nn.Parameter(
                torch.zeros(out_features, rank)
            )
            nn.init.kaiming_uniform_(self.lora_a, a=0.01)
            self.lora_b = nn.Parameter(
                torch.zeros(rank, in_features)
            )
            self.scale = lora_alpha / rank
            self.linear.weight.requires_grad = False
            self.linear.bias.requires_grad = False
        self.dropout = nn.Dropout(
            dropout
        ) if dropout > 0 else nn.Identity()
        if merge:
            self.merge_weight()
    
    def forward(self, x):
        if self.rank > 0 and not self.merge:
            output = self.linear(x) + self.scale * (x @ (self.lora_a @ self.lora_b).T)
        elif self.rank > 0 and self.merge:
            output = self.linear(x)
        else:
            output = self.linear(x)
        return self.dropout(output)
    
    def merge_weight(self):
        if self.merge and self.rank > 0:
            self.linear.weight.data += self.scale * (self.lora_a @ self.lora_b)
    
    def unmerge_weight(self):
        if self.rank > 0:
            self.linear.weight.data -= self.scale * (self.lora_a @ self.lora_b)


batch_size = 32
seq_len = 128
in_features = 768
out_features = 512
rank = 8
lora_alpha = 16
dropout = 0.1

x = torch.randn(batch_size, seq_len, in_features)

lora_layer = LinearLoRALayer(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
    lora_alpha=lora_alpha,
    dropout=dropout,
    merge=False,
)

output = lora_layer(x)
print(f"Output shape (no merge): {output.shape}")

lora_layer_merged = LinearLoRALayer(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
    lora_alpha=lora_alpha,
    dropout=dropout,
    merge=True,
)

output_merged = lora_layer_merged(x)
print(f"Output shape (merged): {output_merged.shape}")

lora_layer.merge_weight()
output_after_merge = lora_layer(x)
lora_layer.unmerge_weight()
output_after_unmerge = lora_layer(x)

print(
    "Max difference between merge/unmerge cycle: ",
    torch.max(torch.abs(output - output_after_unmerge)).item()
)
