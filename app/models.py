import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
try:
    from torch_scatter import scatter
except ImportError:
    from torch_geometric.utils import scatter


class GATEncoderWrapper(nn.Module):
    """GATv2Conv encoder with .encode() and .latent_dim for Stage 2 compatibility."""

    def __init__(self, input_dim=77, hidden_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.latent_dim = hidden_dim
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv3 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)

    def encode(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x  # [N, hidden_dim]

    def forward(self, x, edge_index):
        return self.encode(x, edge_index)

    @classmethod
    def from_checkpoint(cls, checkpoint, device='cpu'):
        config = checkpoint['model_config']
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.2),
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return model.to(torch.device(device))


class BERTEncoder:
    """BERT text encoder - extracts CLS token embeddings."""

    def __init__(self, model_name="bert-base-uncased", device=None):
        from transformers import AutoModel, AutoTokenizer
        self._device = device or torch.device('cpu')
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self._device)
        self.model.eval()
        self._hidden_dim = self.model.config.hidden_size
        self.freeze()
        print(f"  BERT loaded: hidden_dim={self._hidden_dim}")

    def encode(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors='pt'
        ).to(self._device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token [B, 768]

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def device(self):
        return self._device

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


class QFormerBridge(nn.Module):
    """Lightweight Q-Former for graph-to-fixed-dim bottleneck (BLIP-2/MolCA style)."""

    def __init__(self, num_queries=4, hidden_dim=256, gnn_dim=128,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.input_proj = nn.Linear(gnn_dim, hidden_dim)

        self.cross_attn_layers = nn.ModuleList()
        self.cross_attn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.cross_attn_norms.append(nn.LayerNorm(hidden_dim))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout),
            ))
            self.ffn_norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, node_embs, batch_idx):
        B = batch_idx.max().item() + 1
        node_proj = self.input_proj(node_embs)

        counts = torch.bincount(batch_idx, minlength=B)
        max_nodes = counts.max().item()

        padded = torch.zeros(B, max_nodes, self.hidden_dim, device=node_embs.device)
        key_padding_mask = torch.ones(B, max_nodes, dtype=torch.bool, device=node_embs.device)

        for i in range(B):
            mask_i = (batch_idx == i)
            n_i = mask_i.sum().item()
            padded[i, :n_i] = node_proj[mask_i]
            key_padding_mask[i, :n_i] = False

        queries = self.query_tokens.expand(B, -1, -1)

        for ca, ca_norm, ffn, ffn_norm in zip(
            self.cross_attn_layers, self.cross_attn_norms,
            self.ffn_layers, self.ffn_norms
        ):
            attn_out, _ = ca(query=queries, key=padded, value=padded,
                             key_padding_mask=key_padding_mask)
            queries = ca_norm(queries + attn_out)
            queries = ffn_norm(queries + ffn(queries))

        return queries.mean(dim=1)  # [B, hidden_dim]


class AuxiliaryHeads(nn.Module):
    """Auxiliary task heads (needed for checkpoint loading)."""

    def __init__(self, hidden_dim=256, num_attack_classes=7):
        super().__init__()
        self.attack_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim // 2, num_attack_classes)
        )
        self.node_predictor = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.edge_predictor = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.density_predictor = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.task_weights = {'attack': 0.4, 'property': 0.3, 'triplet': 0.3}


class CrossAttentionBridgeV2(nn.Module):
    """v2.2: QFormer + SigLIP + ConGraT soft targets."""

    def __init__(self, gnn_model, text_encoder, hidden_dim=256, dropout=0.1,
                 pooling='mean', use_auxiliary_tasks=True, num_attack_classes=7,
                 contrastive_weight=0.5, auxiliary_weight=0.5,
                 use_qformer=True, num_queries=4, num_qformer_layers=2,
                 soft_target_alpha=0.1):
        super().__init__()
        self.gnn = gnn_model
        self.text_encoder = text_encoder
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.use_auxiliary_tasks = use_auxiliary_tasks
        self.contrastive_weight = contrastive_weight
        self.auxiliary_weight = auxiliary_weight
        self.soft_target_alpha = soft_target_alpha

        self.gnn_dim = gnn_model.latent_dim
        self.text_dim = text_encoder.hidden_dim

        self.use_qformer = use_qformer
        if use_qformer:
            self.qformer = QFormerBridge(
                num_queries=num_queries, hidden_dim=hidden_dim,
                gnn_dim=self.gnn_dim, num_heads=4,
                num_layers=num_qformer_layers, dropout=dropout,
            )
            self.graph_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            )
        else:
            self.graph_proj = nn.Sequential(
                nn.Linear(self.gnn_dim, hidden_dim), nn.GELU(),
                nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            )

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
        )

        self.temperature = nn.Parameter(torch.tensor(0.07))

        if use_auxiliary_tasks:
            self.auxiliary = AuxiliaryHeads(hidden_dim, num_attack_classes)

    def pool_graph(self, node_embs, batch_idx):
        return scatter(node_embs, batch_idx, dim=0, reduce='mean')

    def encode_graph(self, x, edge_index, batch_idx):
        """Encode graph to normalized shared-space embedding."""
        node_embs = self.gnn.encode(x, edge_index)
        if self.use_qformer:
            graph_embs = self.qformer(node_embs, batch_idx)
        else:
            graph_embs = self.pool_graph(node_embs, batch_idx)
        graph_embs = self.graph_proj(graph_embs)
        return F.normalize(graph_embs, dim=-1)

    def encode_text(self, texts):
        """Encode text to normalized shared-space embedding."""
        text_embs = self.text_encoder.encode(texts)
        text_embs = self.text_proj(text_embs)
        return F.normalize(text_embs, dim=-1)
