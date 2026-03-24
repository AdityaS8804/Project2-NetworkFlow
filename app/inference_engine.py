import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from .config import CHECKPOINT_STAGE1, CHECKPOINT_STAGE2, NUM_CLASSES
from .models import GATEncoderWrapper, BERTEncoder, CrossAttentionBridgeV2

CLASSIFIER_PATH = os.path.join(os.path.dirname(CHECKPOINT_STAGE1), 'attack_classifier.pt')


class AttackClassifier(nn.Module):
    """Standalone attack classifier trained on normalized GNN embeddings."""

    def __init__(self, input_dim=128, num_classes=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class InferenceEngine:
    """Wraps Stage 1 (GNN) and Stage 2 (cross-attention bridge) models for inference."""

    def __init__(self, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.gnn_model = None
        self.bridge_model = None
        self.text_encoder = None
        self.attack_classifier = None
        self._emb_mean = None  # embedding normalization stats
        self._emb_std = None

    def load_models(self):
        """Load all models from checkpoints."""
        print(f"Loading models on device: {self.device}")

        # Stage 1: GNN encoder
        print("Loading Stage 1 GNN...")
        s1_ckpt = torch.load(CHECKPOINT_STAGE1, map_location='cpu', weights_only=False)
        self.gnn_model = GATEncoderWrapper.from_checkpoint(s1_ckpt, device=str(self.device))
        self.gnn_model.eval()
        for param in self.gnn_model.parameters():
            param.requires_grad = False
        print(f"  GNN loaded (epoch {s1_ckpt.get('epoch', '?')})")

        # Attack classifier head (trained on frozen GNN embeddings)
        if os.path.exists(CLASSIFIER_PATH):
            print("Loading attack classifier...")
            cls_ckpt = torch.load(CLASSIFIER_PATH, map_location='cpu', weights_only=False)
            self.attack_classifier = AttackClassifier(
                input_dim=cls_ckpt.get('input_dim', 128),
                num_classes=cls_ckpt.get('num_classes', NUM_CLASSES),
            ).to(self.device)
            self.attack_classifier.load_state_dict(cls_ckpt['model_state_dict'])
            self.attack_classifier.eval()
            for param in self.attack_classifier.parameters():
                param.requires_grad = False
            # Load embedding normalization stats
            if 'emb_mean' in cls_ckpt:
                self._emb_mean = cls_ckpt['emb_mean'].to(self.device)
                self._emb_std = cls_ckpt['emb_std'].to(self.device)
            print(f"  Classifier loaded (val_acc={cls_ckpt.get('val_acc', '?'):.1%}, test_acc={cls_ckpt.get('test_acc', '?'):.1%})")
        else:
            print(f"  WARNING: No attack classifier at {CLASSIFIER_PATH}")

        # BERT text encoder
        print("Loading BERT text encoder...")
        self.text_encoder = BERTEncoder(device=self.device)

        # Stage 2: Cross-attention bridge
        print("Loading Stage 2 bridge...")
        s2_ckpt = torch.load(CHECKPOINT_STAGE2, map_location='cpu', weights_only=False)
        cfg = s2_ckpt.get('config', {})

        self.bridge_model = CrossAttentionBridgeV2(
            gnn_model=self.gnn_model,
            text_encoder=self.text_encoder,
            hidden_dim=cfg.get('hidden_dim', 256),
            dropout=0.1,
            pooling=cfg.get('pooling', 'mean'),
            use_auxiliary_tasks=cfg.get('use_auxiliary_tasks', True),
            num_attack_classes=NUM_CLASSES,
            contrastive_weight=cfg.get('contrastive_weight', 0.5),
            auxiliary_weight=cfg.get('auxiliary_weight', 0.5),
            use_qformer=cfg.get('use_qformer', False),
            num_queries=4,
            num_qformer_layers=2,
            soft_target_alpha=cfg.get('soft_target_alpha', 0.1),
        ).to(self.device)

        self.bridge_model.load_state_dict(s2_ckpt['model_state_dict'], strict=False)
        self.bridge_model.eval()
        for param in self.bridge_model.parameters():
            param.requires_grad = False
        print(f"  Bridge loaded (epoch {s2_ckpt.get('epoch', '?')}, QFormer={cfg.get('use_qformer', False)})")

    @torch.no_grad()
    def get_graph_embedding(self, pyg_data):
        """Stage 1: graph -> 128-dim embedding via GNN + global_mean_pool."""
        batch = Batch.from_data_list([pyg_data]).to(self.device)
        node_emb = self.gnn_model.encode(batch.x, batch.edge_index)
        graph_emb = global_mean_pool(node_emb, batch.batch)
        return graph_emb.cpu().numpy()[0]

    @torch.no_grad()
    def get_attack_prediction(self, pyg_data):
        """Predict attack class using the standalone classifier on GNN embeddings.

        Returns (predicted_class_id, class_probabilities[7]).
        """
        batch = Batch.from_data_list([pyg_data]).to(self.device)
        node_emb = self.gnn_model.encode(batch.x, batch.edge_index)
        graph_emb = global_mean_pool(node_emb, batch.batch)

        if self.attack_classifier is not None:
            # Normalize embedding to match training distribution
            if self._emb_mean is not None:
                graph_emb = (graph_emb - self._emb_mean) / self._emb_std
            logits = self.attack_classifier(graph_emb)
        else:
            print("WARNING: attack_classifier is None — returning uniform logits (all Benign)")
            logits = torch.zeros(1, NUM_CLASSES, device=self.device)

        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(probs.argmax())
        return pred, probs

    @torch.no_grad()
    def get_shared_space_embedding(self, pyg_data):
        """Stage 2: graph -> 256-dim shared space embedding (L2-normalized)."""
        batch = Batch.from_data_list([pyg_data]).to(self.device)
        graph_emb = self.bridge_model.encode_graph(batch.x, batch.edge_index, batch.batch)
        return graph_emb.cpu().numpy()[0]

    @torch.no_grad()
    def get_text_embedding(self, text):
        """Encode text -> 256-dim shared space embedding (L2-normalized)."""
        text_emb = self.bridge_model.encode_text([text])
        return text_emb.cpu().numpy()[0]
