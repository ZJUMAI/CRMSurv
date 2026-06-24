import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor
from .util import NystromAttention,initialize_weights
from .util import SNN_Block, EMB_Block
from .util import MultiheadAttention
from dataclasses import dataclass
from .kmeans.kmeans_pytorch import kmeans

def conv1d_Block(dim1, dim2, dropout=0.15):
    return nn.Sequential(
        nn.Conv1d(dim1, dim2, 1),
        nn.InstanceNorm1d(dim2),
        nn.ReLU()
    )

def l2norm(t: Tensor, eps: float = 1e-8) -> Tensor:
    """L2-normalize the last dimension of a tensor."""
    return F.normalize(t, dim=-1, eps=eps)

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_G, self).__init__()
        # Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1).cuda()
        h = torch.cat((cls_tokens, features), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_P, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]

class Transformer(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer, self).__init__()
        # Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1).cuda()
        h = torch.cat((cls_tokens, features), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]

class PrototypeBank(nn.Module):
    """Maintain one prototype center per risk group with EMA updates."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        proto = torch.zeros(cfg.num_groups, cfg.dim)
        self.register_buffer("prototypes", l2norm(proto))
        self.register_buffer("counts", torch.zeros(cfg.num_groups))

    @torch.no_grad()
    def update(
        self,
        z: torch.Tensor,
        risk_group: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> None:
        """
        z: [B, N, D] patch/token embeddings, or [N, D].
        risk_group: [B] in [0, 1, 2, 3].
        c: [B], c=1 censored, c=0 uncensored.
        """
        if z.ndim == 2:
            z = z.unsqueeze(0)
        if z.ndim != 3:
            raise ValueError(f"z must be [B,N,D] or [N,D], got {z.shape}")

        pooled = l2norm(z.mean(dim=1))  
        g = risk_group.long()
        if c is None:
            c = torch.zeros_like(g, dtype=torch.float32, device=z.device)
        else:
            c = c.to(z.device).float()

        for i in range(pooled.shape[0]):
            gi = int(g[i].item())
            if c[i] > 0.5:
                target_groups = range(gi, self.cfg.num_groups)
                # scale = self.cfg.censored_update_scale
                scale = 1 / max(len(target_groups), 1)
            else:
                target_groups = [gi]
                scale = 1.0

            eff_m = 1 - (1 - self.cfg.momentum) * scale
            for gj in target_groups:
                old = self.prototypes[gj]
                new = eff_m * old + (1 - eff_m) * pooled[i]
                self.prototypes[gj] = l2norm(new.unsqueeze(0)).squeeze(0)
                self.counts[gj] += 1

    def logits_vs_prototypes(self, z: torch.Tensor) -> torch.Tensor:
        """z:[B,D] -> logits:[B, G]."""
        z = l2norm(z)
        return z @ self.prototypes.t() / self.cfg.temperature

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim=512, heads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

    def forward(self, query, key_value):
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        attn_output, _ = self.attn(q_norm, kv_norm, kv_norm)
        return query + attn_output 

class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)

class QFormerBlock(nn.Module):
    def __init__(self, modal='wsi', dim=512, heads=8):
        super().__init__()
        self.modal = modal
        self.self_attn = TransLayer(dim=dim)

        self.cross_attn1 = CrossAttentionLayer(dim=dim, heads=heads)
        self.cross_attn2 = CrossAttentionLayer(dim=dim, heads=heads)
        if modal=='wsi':
            self.cross_attn3 = CrossAttentionLayer(dim=dim, heads=heads)

        self.ffn = FeedForward(dim=dim)

    def forward(self, queries, latents, features, proto=None):

        if self.modal =='wsi':
            query1, query2, query3 = torch.chunk(queries, chunks=3, dim=1)
            query1_f = self.cross_attn1(query=query1, key_value=features)
            query2_p = self.cross_attn2(query=query2, key_value=proto)
            query3_l = self.cross_attn2(query=query3, key_value=latents)
            queries = self.self_attn(torch.cat([query1_f, query2_p, query3_l], dim=1))

        else:
            query1, query2 = torch.chunk(queries, chunks=2, dim=1)
            query1_f = self.cross_attn1(query=query1, key_value=features)
            query2_l = self.cross_attn2(query=query2, key_value=latents)
            queries = self.self_attn(torch.cat([query1_f, query2_l], dim=1))

        queries = self.ffn(queries)
        return queries


class ModalityQFormer(nn.Module):
    def __init__(self, num_queries, modal='wsi', dim=512, depth=3):
        super().__init__()

        self.modal = modal
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        self.layers = nn.ModuleList([QFormerBlock(modal=modal,dim=dim) for _ in range(depth)])

    def forward(self, latents, features, proto=None):
        batch_size = latents.shape[0]

        q = self.queries.expand(batch_size, -1, -1)
        if self.modal=='wsi':
            for layer in self.layers:
                q = layer(q, latents, features, proto)
        else:
            for layer in self.layers:
                q = layer(q, latents, features)

        return q 

class TokenCompression(nn.Module):
    """
    Reconstruction-aware Patch token compression module.
    """

    def __init__(
        self,
        dim: int,
        seed: int,
        use_l2_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.seed = seed
        self.dim = dim
        self.use_l2_norm = use_l2_norm
        self.eps = eps


    def forward(self, X, K, bases):

        squeeze_batch = False

        if X.dim() == 2:
            X = X.unsqueeze(0)  # [1, N, D]
            squeeze_batch = True

        assert X.dim() == 3, "X should be [N, D] or [B, N, D]"

        Bsz, N, D = X.shape
        assert D == self.dim

        if bases.dim() == 2:
            bases = bases.unsqueeze(0).expand(Bsz, -1, -1)  # [B, M, D]

        assert bases.dim() == 3
        assert bases.size(0) == Bsz
        assert bases.size(2) == D

        M = bases.size(1)

        # ------------------------------------------------
        # 1. L2 normalize
        # ------------------------------------------------
        if self.use_l2_norm:
            X_for_ls = F.normalize(X, p=2, dim=-1, eps=self.eps)
            bases_for_ls = F.normalize(bases, p=2, dim=-1, eps=self.eps)
        else:
            X_for_ls = X
            bases_for_ls = bases

        # ------------------------------------------------
        # 2. Least squares approximation:
        #    X_for_ls ≈ alpha @ bases_for_ls
        #
        #    For each sample:
        #    X: [N, D]
        #    bases: [M, D]
        #    alpha: [N, M]
        # ------------------------------------------------

        # A = bases^T: [B, D, M]
        A = bases_for_ls.transpose(1, 2)

        # pinv(A): [B, M, D]
        A_pinv = torch.linalg.pinv(A)

        # X^T: [B, D, N]
        X_T = X_for_ls.transpose(1, 2)

        # alpha_T: [B, M, N]
        alpha_T = torch.bmm(A_pinv, X_T)

        # alpha: [B, N, M]
        alpha = alpha_T.transpose(1, 2)

        # ------------------------------------------------
        # 3. Reconstruction
        # ------------------------------------------------
        X_recon = torch.bmm(alpha, bases_for_ls)  # [B, N, D]

        # ------------------------------------------------
        # 4. Reconstruction error
        # ------------------------------------------------
        error = torch.norm(X_for_ls - X_recon, p=2, dim=-1)  # [B, N]

        # ------------------------------------------------
        # 5. Keep Top-K patches with largest reconstruction error
        # ------------------------------------------------
        keep_idx = torch.topk(error, k=K, dim=1, largest=True).indices  # [B, K]
        keep_idx = keep_idx.sort(dim=1).values  # [B, K]

        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, D)
        final_X = torch.gather(X, dim=1, index=gather_idx)  # [B, K, D]

        aux = {
            "error": error,
            "keep_idx": keep_idx,
            "X_recon": X_recon,
            "alpha": alpha,
            "cluster": bases.squeeze(),
        }

        if squeeze_batch:
            final_X = final_X.unsqueeze(0)
            aux = {k: v.unsqueeze(0) for k, v in aux.items()}

        return final_X, aux

@dataclass
class ProtoCfg:
    num_groups: int = 4  # risk groups: 1/2/3/4
    dim: int = 256
    momentum: float = 0.5 
    temperature: float = 0.2
    censored_update_scale: float = 0.5 
    censored_loss_scale: float = 0.5  


class Network(nn.Module):
    def __init__(self,
                 omic_sizes=[100, 200, 300, 400, 500, 600],
                 input_dim=1024,
                 dim=256,
                 wsi_latent_dim=3584,
                 gene_latent_dim=3584,
                 wsi_queries=16,
                 gene_queries=16,
                 emb_dim=1536,
                 n_classes=4,
                 **kwargs): 
        super().__init__()

        if kwargs['use_cohort_loss']:
            cfg = ProtoCfg(num_groups=4, dim=256)
            self.wsi_bank = PrototypeBank(cfg)
            self.omic_bank = PrototypeBank(cfg)

        self.size_dict = {
            "pathomics": {"small": [input_dim, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [256, 256], "large": [1024, 1024, 1024, 256]},
            "omic_emb": {"small": [emb_dim, 256]}
        }
        self.use_txt = kwargs['use_txt']
        self.use_compression = kwargs['use_compression']

        self.num_cluster = kwargs['cluster']
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"]["small"]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)

        if self.use_txt:
            self.proj_wsi_txt = nn.Linear(wsi_latent_dim, dim)
            self.proj_gene_txt = nn.Linear(gene_latent_dim, dim)

        if self.use_compression:
            self.compression = TokenCompression(dim=input_dim,seed=kwargs['seed'])

        # Genomic Embedding Network
        hidden = self.size_dict["genomics"]["small"]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        self.wsi_qformer = ModalityQFormer(num_queries=wsi_queries, modal='wsi', dim=dim, depth=kwargs['depth'])
        self.gene_qformer = ModalityQFormer(num_queries=gene_queries, modal='gene',dim=dim, depth=kwargs['depth'])

        self.classifier = nn.Linear(dim, n_classes)

        self.f_cross1 = CrossAttentionLayer(dim=dim)
        self.f_cross2 = CrossAttentionLayer(dim=dim)

        self.evaluate = kwargs['evaluate']
        self.apply(initialize_weights)


    def forward(self, **kwargs):
        x_path = kwargs["x_path"]
        x_omic = [kwargs["x_omic%d" % i] for i in range(1, 7)]

        wsi_txt = kwargs["wsi_txt"]
        gene_txt = kwargs["gene_txt"]
        cluster = kwargs["cluster"]

        if self.use_txt:
            wsi_txt = self.proj_wsi_txt(wsi_txt).unsqueeze(0)  
            gene_txt = self.proj_gene_txt(gene_txt).unsqueeze(0)  

        # genomics embedding
        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0)  

        K = x_path.shape[0]

        if self.use_compression:
            pathomics_features, aux = self.compression(x_path, K= K//4, bases=cluster)

        pathomics_features = self.pathomics_fc(torch.cat([aux['cluster'], pathomics_features.squeeze(0)],dim=1))
        pathomics_features, proto = torch.split(pathomics_features,[self.num_cluster,K//4],dim=1)

        wsi_q = self.wsi_qformer(latents=wsi_latent, features=pathomics_features, proto=proto)
        gene_q = self.gene_qformer(latents=gene_latent, features=genomics_features)

        wsi2gene = self.f_cross1(query=wsi_q, key_value=gene_q)
        gene2wsi = self.f_cross2(query=gene_q, key_value=wsi_q)

        joint_q = torch.cat([wsi2gene, gene2wsi], dim=1)
        fusion = joint_q.mean(dim=1)  # [B, 256]

        logits = self.classifier(fusion)  # [1, n_classes]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        if not self.evaluate:
            return hazards, S, wsi_q.mean(dim=1), gene_q.mean(
                dim=1), fusion 
        else:
            return hazards, S, wsi_q.mean(dim=1), gene_q.mean(dim=1), fusion, aux