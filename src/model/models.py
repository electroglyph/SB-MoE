import torch
from torch import clamp as t_clamp
from torch import nn as nn
from torch import sum as t_sum
from torch import max as t_max
from torch import einsum
from torch.nn import functional as F

class Specializer(nn.Module):
    def __init__(self, hidden_size, device, latent_size=None, non_linearity='relu'):
        super(Specializer, self).__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        self.latent_size = latent_size if latent_size is not None else hidden_size // 2
        
        self.non_linearity = non_linearity.lower()
        if self.non_linearity == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = F.relu
            
        self.query_embedding_changer_1 = nn.Linear(self.hidden_size, self.latent_size).to(device)
        self.query_embedding_changer_4 = nn.Linear(self.latent_size, self.hidden_size).to(device)
        
    def forward(self, query_embs):
        query_embs_1 = self.activation(self.query_embedding_changer_1(query_embs))
        query_embs_2 = self.query_embedding_changer_4(query_embs_1)
        
        return query_embs_2


class MoEBiEncoder(nn.Module):
    def __init__(
        self,
        doc_model,
        tokenizer,
        num_classes,
        max_tokens=512,
        normalize=False,
        specialized_mode='sbmoe_top1',
        pooling_mode='mean',
        use_adapters=True,
        track_expert_usage=False,
        latent_size=None,
        non_linearity='relu',
        aux_loss_coeff=0.0,
        device='cpu',
    ):
        super(MoEBiEncoder, self).__init__()
        self.doc_model = doc_model.to(device)
        self.hidden_size = self.doc_model.config.hidden_size
        self.tokenizer = tokenizer
        self.device = device
        self.aux_loss_coeff = aux_loss_coeff
        self.normalize = normalize
        self.max_tokens = max_tokens
        self.use_adapters = use_adapters
        assert specialized_mode in ['sbmoe_top1', 'sbmoe_all', 'random'], 'Only random, sbmoe_top1 and sbmoe_all specialzed modes allowed'
        self.specialized_mode = specialized_mode
        assert pooling_mode in ['max', 'mean', 'cls', 'identity'], 'Only cls, identity, max and mean pooling allowed'
        if pooling_mode == 'mean':
            self.pooling = self.mean_pooling
        elif pooling_mode == 'max':
            self.pooling = self.max_pooling
        elif pooling_mode == 'cls':
            self.pooling = self.cls_pooling
        elif pooling_mode == 'identity':
            self.pooling = self.identity
        
        self.num_classes = num_classes
        self.expert_usage_counter = torch.zeros(self.num_classes).to(self.device)
        self.track_expert_usage = track_expert_usage
        self.init_cls()
        
        self.specializer = nn.ModuleList([Specializer(self.hidden_size, self.device, latent_size, non_linearity) for _ in range(self.num_classes)])    
        
        
    def query_encoder_no_moe(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])
        
    
    def init_cls(self):
        self.cls_1 = nn.Linear(self.hidden_size, self.hidden_size//2).to(self.device)
        # self.cls_2 = nn.Linear(self.hidden_size*2, self.hidden_size*4).to(self.device)
        self.cls_3 = nn.Linear(self.hidden_size//2, self.num_classes).to(self.device)
        self.noise_linear = nn.Linear(self.hidden_size, self.num_classes).to(self.device)
        
    
    def query_encoder(self, sentences):
        query_embedding = self.query_encoder_no_moe(sentences)
        aux_loss = None
        if self.use_adapters:
            query_class, probs = self._gate_forward(query_embedding)
            if self.training and self.aux_loss_coeff > 0:
                aux_loss = self.compute_load_balance_loss(probs, query_class)
            query_embedding = self.query_embedder(query_embedding, query_class)
        return query_embedding, aux_loss
        

    def doc_encoder_no_moe(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])

    def doc_encoder(self, sentences):
        doc_embedding = self.doc_encoder_no_moe(sentences)
        aux_loss = None
        if self.use_adapters:
            doc_class, probs = self._gate_forward(doc_embedding)
            if self.training and self.aux_loss_coeff > 0:
                aux_loss = self.compute_load_balance_loss(probs, doc_class)
            doc_embedding = self.doc_embedder(doc_embedding, doc_class)
        return doc_embedding, aux_loss
        

    def compute_load_balance_loss(self, probs, gating_weights):
        # probs: [B, N], gating_weights: [B, N]
        num_experts = self.num_classes
        f = (gating_weights > 0).float().mean(0)
        P = probs.mean(0)
        return num_experts * torch.sum(f * P)

    def _gate_forward(self, embedding):
        x1 = F.relu(self.cls_1(embedding))
        logits = self.cls_3(x1)
        probs = torch.softmax(logits, dim=-1)

        if self.training:
            noise_logits = self.noise_linear(embedding)
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            noisy_logits = logits + noise
            noisy_probs = torch.softmax(noisy_logits, dim=-1)
            
            topk_values, topk_indices = torch.topk(noisy_probs, 1, dim=1)
            mask = torch.zeros_like(noisy_probs).scatter_(1, topk_indices, 1)
            gating_weights = noisy_probs * mask
        else:
            if self.specialized_mode == 'sbmoe_top1':
                topk_values, topk_indices = torch.topk(probs, 1, dim=1)
                mask = torch.zeros_like(probs).scatter_(1, topk_indices, 1)
                gating_weights = probs * mask
            elif self.specialized_mode == 'sbmoe_all':
                gating_weights = probs
            else: # random
                gating_weights = torch.rand(embedding.shape[0], self.num_classes).to(self.device)
                topk_values, topk_indices = torch.topk(gating_weights, 1, dim=1)
                mask = torch.zeros_like(gating_weights).scatter_(1, topk_indices, 1)
                gating_weights = gating_weights * mask
            
        if self.track_expert_usage:
            top_expert = gating_weights.argmax(dim=1)
            for idx in top_expert:
                self.expert_usage_counter[idx] += 1
                
        return gating_weights, probs

    def forward(self, data):
        query_embedding, q_aux = self.query_encoder(data[0])
        pos_embedding, p_aux = self.doc_encoder(data[1])

        if self.training and self.aux_loss_coeff > 0 and q_aux is not None:
            return query_embedding, pos_embedding, (q_aux + p_aux) / 2
        return query_embedding, pos_embedding

    def val_forward(self, data):
        query_embedding, _ = self.query_encoder(data[0])
        pos_embedding, _ = self.doc_encoder(data[1])

        return query_embedding, pos_embedding


    def query_embedder(self, query_embedding, query_class):
        query_embs = [self.specializer[i](query_embedding) for i in range(self.num_classes)]
        query_embs = torch.stack(query_embs, dim=1)
        
        # Normalize base embedding before adding residual to match expert output scale
        base_emb = query_embedding
        if self.normalize:
            base_emb = F.normalize(base_emb, dim=-1)

        query_embs = F.normalize(einsum('bmd,bm->bd', query_embs, query_class), dim=-1, eps=1e-6) + base_emb

        if self.normalize:
            return F.normalize(query_embs, dim=-1)
        return query_embs
    
    def doc_embedder(self, doc_embedding, doc_class):
        return self.query_embedder(doc_embedding, doc_class)
        

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)


    @staticmethod
    def cls_pooling(model_output, attention_mask):
        last_hidden = model_output[0] # Use index for robustness across model types
        return last_hidden[:, 0]


    @staticmethod
    def identity(model_output, attention_mask):
        return model_output['pooler_output']
    

    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return t_max(token_embeddings, 1)[0]
