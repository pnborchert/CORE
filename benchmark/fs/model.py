from transformers import BertTokenizer, BertForMaskedLM, Trainer, BertPreTrainedModel, TrainingArguments, BertForSequenceClassification, BertModel
import transformers
from transformers.utils import logging
import os, torch
from torch.nn import functional as F
from torch import nn
from typing import Optional, Tuple
from transformers.utils import ModelOutput
from dataclasses import dataclass

@dataclass
class FewShotOutputs(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Tuple[torch.FloatTensor] = None

@dataclass
class HCRPOutputs(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Tuple[torch.FloatTensor] = None
    logits_bf_max: Optional[torch.FloatTensor] = None
    logits_proto: Optional[torch.FloatTensor] = None
    labels_proto: Optional[torch.FloatTensor] = None
    sim_scalar: Optional[torch.FloatTensor] = None

# ===========================================================================================

class HCRP(BertPreTrainedModel):

    def __init__(
        self,
        N,
        K,
        Q,
        na_rate,
        model_name,
        max_length,
        do_lower_case=True,
        ckpt=None,
        output_hidden_states=True,
        layer_norm=False,
        dropout=0.2,
        gamma=1,
        lamda=2.5,
        max_length_name=25,
        temp_proto=1,
        # dual_target=False
        ):

        encoder = BertModel.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states
            )

        super().__init__(encoder.config)

        # self.sep_marker = "*SEP*"
        self.entity_marker = {
            "*START_E1*": "[unused0]",
            "*END_E1*": "[unused1]",
            "*START_E2*": "[unused2]",
            "*END_E2*": "[unused3]",
        }

        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        
        self.max_length = max_length
        self.do_lower_case = do_lower_case
        # self.dual_target = dual_target
        self.encoder = encoder
        self.hidden = encoder.config.hidden_size

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=self.do_lower_case)
        self.tokenizer.truncation_side="right"
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": list(self.entity_marker.values())
            })

        self.entity_marker_ids = self.tokenizer.convert_tokens_to_ids(list(self.entity_marker.values()))

        # HCRP
        self.max_length_name = max_length_name
        self.gamma = gamma
        self.lamda = lamda
        self.temp_proto = temp_proto

        self.rel_glo_linear = nn.Linear(self.hidden, self.hidden * 2)

        if ckpt is not None:
            logger = logging.get_logger("transformers")
            logger.warning(f"{'*'*30}\nLoad model from checkpoint: {ckpt}\n{'*'*30}")
            if ckpt.split(".")[-1] not in ["bin", "pt", "pth"]:
                ckpt = os.path.join(ckpt, "pytorch_model.bin")

            # load from state dict
            self.load_state_dict(torch.load(ckpt))
    
    def __dist__(self, x, y, dim):
        return (x * y).sum(dim)
    
    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
        
    def forward(
        self,
        support_input_ids,
        support_attention_mask,
        support_marker_pos,
        query_input_ids,
        query_attention_mask,
        query_marker_pos,
        name_input_ids,
        name_attention_mask,
        is_eval,
        **kwargs,
        ):
        
        B = support_input_ids.size(0) # batch size
        TOTAL_Q = query_input_ids.size(1) # total query size

        # reshape inputs
        support_input_ids = support_input_ids.view(-1, support_input_ids.size(-1)) # (B * N * K, max_len)
        support_attention_mask = support_attention_mask.view(-1, support_attention_mask.size(-1)) # (B * N * K, max_len)
        query_input_ids = query_input_ids.view(-1, query_input_ids.size(-1)) # (B * N * K, max_len)
        query_attention_mask = query_attention_mask.view(-1, query_attention_mask.size(-1)) # (B * N * K, max_len)
        name_input_ids = name_input_ids.view(-1, name_input_ids.size(-1)) # (B * N * K, max_len_name)
        name_attention_mask = name_attention_mask.view(-1, name_attention_mask.size(-1)) # (B * N * K, max_len_name)

        # embeddings support
        support_loc = self.encoder(support_input_ids, attention_mask=support_attention_mask).hidden_states[-1] # (B * N * K, max_len, h)
        support_glo = self.get_embeddings_entity_start(support_loc, support_marker_pos.view(-1, support_marker_pos.size(-1))) # (B*N*K, 2*h)

        # embeddings query
        query_loc = self.encoder(query_input_ids, attention_mask=query_attention_mask).hidden_states[-1] # (B * N * Q, max_len, h)
        query_glo = self.get_embeddings_entity_start(query_loc, query_marker_pos.view(-1, query_marker_pos.size(-1))) # (B*N*Q, 2*h)

        # embeddings name
        name = self.encoder(name_input_ids, attention_mask=name_attention_mask)
        rel_text_loc = name.hidden_states[-1] # (B * N * Q, max_len_name, h)
        rel_text_glo = name.pooler_output # (B * N * Q, h)
        
        # global features
        support_glo = support_glo.view(-1, self.N, self.K, self.hidden * 2)  # (B, N, K, 2D)
        query_glo = query_glo.view(-1, TOTAL_Q, self.hidden * 2)  # (B, total_Q, 2D)
        rel_text_glo = self.rel_glo_linear(rel_text_glo.view(B, self.N, self.hidden))  # (B, N, 2D)

        # global prototypes
        proto_glo = torch.mean(support_glo, 2) + rel_text_glo  # Calculate prototype for each class (B, N, 2D)

        # local features
        rel_text_loc_s = rel_text_loc.unsqueeze(1).expand(-1, self.K, -1, -1).contiguous().view(B * self.N * self.K, -1, self.hidden)  # (B * N * K, L, D)
        rel_support = torch.bmm(support_loc, torch.transpose(rel_text_loc_s, 2, 1))  # (B * N * K, L, L)
        ins_att_score_s, _ = rel_support.max(-1)  # (B * N * K, L)

        ins_att_score_s = F.softmax(torch.tanh(ins_att_score_s), dim=1).unsqueeze(-1)  # (B * N * K, L, 1)
        support_loc = torch.sum(ins_att_score_s * support_loc, dim=1)  # (B * N * K, D)
        support_loc = support_loc.view(B, self.N, self.K, self.hidden)

        ins_att_score_r, _ = rel_support.max(1)  # (B * N * K, L)
        ins_att_score_r = F.softmax(torch.tanh(ins_att_score_r), dim=1).unsqueeze(-1)  # (B * N * K, L, 1)
        rel_text_loc = torch.sum(ins_att_score_r * rel_text_loc_s, dim=1).view(B, self.N, self.K, self.hidden)
        rel_text_loc = torch.mean(rel_text_loc, 2)  # (B, N, D)

        query_query = torch.bmm(query_loc, torch.transpose(query_loc, 2, 1))  # (B * total_Q, L, L)
        ins_att_score_q, _ = query_query.max(-1)  # (B * total_Q, L)
        ins_att_score_q = F.softmax(torch.tanh(ins_att_score_q), dim=1).unsqueeze(-1)  # (B * total_Q, L, 1)
        query_loc = torch.sum(ins_att_score_q * query_loc, dim=1)  # (B * total_Q, D)
        query_loc = query_loc.view(B, TOTAL_Q, self.hidden)  # (B, total_Q, D)

        # local prototypes
        proto_loc = torch.mean(support_loc, 2) + rel_text_loc  # (B, N, D)

        # hybrid prototype
        proto_hyb = torch.cat((proto_glo, proto_loc), dim=-1)  # (B, N, 3D)
        query_hyb = torch.cat((query_glo, query_loc), dim=-1)  # (B, total_Q, 3D)
        rel_text_hyb = torch.cat((rel_text_glo, rel_text_loc), dim=-1)  # (B, N, 3D)

        logits = self.__batch_dist__(proto_hyb, query_hyb)  # (B, total_Q, N)
        pred, _ = torch.max(logits.view(-1, self.N), 1) # (B, N * Q, N)

        logits_proto, labels_proto, sim_scalar = None, None, None

        if not is_eval:
            # relation-prototype contrastive learning
            # # relation as anchor
            rel_text_anchor = rel_text_hyb.view(B * self.N, -1).unsqueeze(1)  # (B * N, 1, 3D)

            # select positive prototypes
            proto_hyb = proto_hyb.view(B * self.N, -1)  # (B * N, 3D)
            pos_proto_hyb = proto_hyb.unsqueeze(1)  # (B * N, 1, 3D)

            # select negative prototypes
            neg_index = torch.zeros(B, self.N, self.N - 1)  # (B, N, N - 1)
            for b in range(B):
                for i in range(self.N):
                    index_ori = [i for i in range(b * self.N, (b + 1) * self.N)]
                    index_ori.pop(i)
                    neg_index[b, i] = torch.tensor(index_ori)

            neg_index = neg_index.long().view(-1).cuda()  # (B * N * (N - 1))
            neg_proto_hyb = torch.index_select(proto_hyb, dim=0, index=neg_index).view(B * self.N, self.N - 1, -1)

            # compute prototypical logits
            proto_selected = torch.cat((pos_proto_hyb, neg_proto_hyb), dim=1)  # (B * N, N, 3D)
            logits_proto = self.__batch_dist__(proto_selected, rel_text_anchor).squeeze(1)  # (B * N, N)
            logits_proto /= self.temp_proto  # scaling temperatures for the selected prototypes

            # targets
            labels_proto = torch.cat((torch.ones(B * self.N, 1), torch.zeros(B * self.N, self.N - 1)), dim=-1).cuda()  # (B * N, 2N)
            
            # task similarity scalar
            features_sim = torch.cat((proto_hyb.view(B, self.N, -1), rel_text_hyb), dim=-1)
            features_sim = self.l2norm(features_sim)
            sim_task = torch.bmm(features_sim, torch.transpose(features_sim, 2, 1))  # (B, N, N)
            sim_scalar = torch.norm(sim_task, dim=(1, 2))  # (B)
            sim_scalar = torch.softmax(sim_scalar, dim=-1)
            sim_scalar = sim_scalar.repeat(self.N*self.Q, 1).t().reshape(-1)  # (B*totalQ)

        out = HCRPOutputs(
            logits=pred,
            logits_bf_max=logits,
            logits_proto=logits_proto,
            labels_proto=labels_proto,
            sim_scalar=sim_scalar,
        )

        return out

    def get_embeddings_entity_start(self, embeddings, marker_pos):
        """
        embeddings (B*N*K, max_len, h)
        marker_pos (B*N*K, 2)
        """

        start_e1 = marker_pos[:,0] # (B*N*K)
        start_e2 = marker_pos[:,1] # (B*N*K)

        e_e1 = embeddings[torch.arange(embeddings.shape[0]),start_e1,:] # (B*N*K, h)
        e_e2 = embeddings[torch.arange(embeddings.shape[0]),start_e2,:] # (B*N*K, h)
        e_cat = torch.cat((e_e1, e_e2), dim=1) # (B*N*K, 2*h)

        return e_cat
    
    def l2norm(self, X):
        norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
        X = torch.div(X, norm)
        return X
    
    def hcrp_loss(self, logits, label, weight=None):
        """
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        """
        N = logits.size(-1)

        # focal weights
        logits_ = torch.softmax(self.l2norm(logits), dim=-1)
        logits_ = logits_.view(-1, N)
        logits, label = logits.view(-1, N), label.view(-1)
        probs = torch.stack([logits_[i, t] for i, t in enumerate(label)])
        focal_weight = torch.pow(1 - probs, self.gamma)

        # task adaptive weights
        if weight is not None:
            focal_weight = focal_weight * weight.view(-1)

        # add weights to cross entropy
        ce_loss = self.loss_fct(logits, label)  # (B*totalQ)
        tf_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            tf_loss = (tf_loss / focal_weight.sum()).sum()
        elif self.reduction == 'sum':
            tf_loss = tf_loss.sum()

        return tf_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # support_labels = inputs.pop("support_labels")
        query_labels = inputs.pop("query_labels")
        is_eval = model.training == False
        inputs["is_eval"] = is_eval

        out = model(**inputs)
        logits = out.logits_bf_max

        self.loss_fct = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reduction = 'mean'
        self.sigmoid = nn.Sigmoid()

        if is_eval:
            loss = self.hcrp_loss(logits=logits, label=query_labels)
        else:
            loss = self.hcrp_loss(logits=logits, label=query_labels) + self.lamda * self.bce_loss(out.logits_proto, out.labels_proto)
            
        out = HCRPOutputs(
            loss=loss,
            logits=logits,
        )

        return (loss, out) if return_outputs else loss

    def tokenize(self, episode):

        # replace entity_marker
        support_name, support_text = self.preprocess_episode(episode["data_support"])
        query_name, query_text = self.preprocess_episode(episode["data_query"])

        # encode inputs
        inputs_support = self.tokenizer(support_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs_query = self.tokenizer(query_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs_name = self.tokenizer(support_name, padding="max_length", truncation=True, max_length=self.max_length_name, return_tensors="pt")

        # labels to tensor
        support_labels = torch.as_tensor(episode["labels_support"], dtype=torch.long)
        query_labels = torch.as_tensor(episode["labels_query"], dtype=torch.long)
        
        # get marker positions
        support_marker_pos = self.get_marker_entity_start(inputs_support["input_ids"])
        query_marker_pos = self.get_marker_entity_start(inputs_query["input_ids"])

        # return dict
        enc = {
            "support_input_ids":inputs_support["input_ids"],
            "support_attention_mask":inputs_support["attention_mask"],
            # "support_labels":support_labels,
            "support_marker_pos":support_marker_pos,
            "query_input_ids":inputs_query["input_ids"],
            "query_attention_mask":inputs_query["attention_mask"],
            "query_labels":query_labels,
            "query_marker_pos":query_marker_pos,
            "name_input_ids":inputs_name["input_ids"],
            "name_attention_mask":inputs_name["attention_mask"],
        }

        return enc
    
    def preprocess_episode(self, episode):
        episode_name = []
        episode_text = []
        for i in episode:
            if i[0] not in episode_name:
                episode_name.append(i[0])
            episode_text.append(self.replace_entity_marker(i[1:]))
        return episode_name, episode_text

    def replace_entity_marker(self, x):
        tokens = []
        for token in x:
            if token in self.entity_marker:
                tokens.append(self.entity_marker[token])
            else:
                tokens.append(token)
        return " ".join(tokens)
    
    def get_marker_entity_start(self, input_ids):
        """
        BERT_EM ENTITY-START variant: Extract marker postions of e1_start and e2_start.
        """
        _, e1_start = torch.where(input_ids == self.entity_marker_ids[0])
        _, e2_start = torch.where(input_ids == self.entity_marker_ids[2])
        
        return torch.cat((e1_start.unsqueeze(1), e2_start.unsqueeze(1)), dim=1)

# ===========================================================================================

class BERTEMModel(BertPreTrainedModel):

    def __init__(
        self,
        N,
        K,
        Q,
        na_rate,
        model_name,
        max_length,
        do_lower_case=True,
        ckpt=None,
        output_hidden_states=True,
        layer_norm=False,
        dropout=0.2,
        ):

        encoder = BertForMaskedLM.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states
            )

        super().__init__(encoder.config)

        self.entity_marker = {
            "*START_E1*": "[unused0]",
            "*END_E1*": "[unused1]",
            "*START_E2*": "[unused2]",
            "*END_E2*": "[unused3]",
        }

        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        
        self.max_length = max_length
        self.do_lower_case = do_lower_case
        self.encoder = encoder
        self.hidden = encoder.config.hidden_size

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=self.do_lower_case)
        self.tokenizer.truncation_side="right"
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": list(self.entity_marker.values())
            })

        self.entity_marker_ids = self.tokenizer.convert_tokens_to_ids(list(self.entity_marker.values()))

        # BERT_EM
        if layer_norm is False:
            self.ff = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features=self.hidden*2, out_features=self.hidden),
            )
        else:
            self.ff = nn.Sequential(
                nn.LayerNorm(self.hidden*2),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=self.hidden*2, out_features=self.hidden),
            )

        if ckpt is not None:
            logger = logging.get_logger("transformers")
            logger.warning(f"{'*'*30}\nLoad model from checkpoint: {ckpt}\n{'*'*30}")
            if ckpt.split(".")[-1] not in ["bin", "pt", "pth"]:
                ckpt = os.path.join(ckpt, "pytorch_model.bin")

            # load from state dict
            self.load_state_dict(torch.load(ckpt))
        
    def forward(
        self,
        support_input_ids,
        support_attention_mask,
        support_marker_pos,
        query_input_ids,
        query_attention_mask,
        query_marker_pos,
        **kwargs,
        ):
        
        B = support_input_ids.size(0) # batch size


        # reshape inputs
        support_input_ids = support_input_ids.view(-1, support_input_ids.size(-1)) # (B * N * K, max_len)
        support_attention_mask = support_attention_mask.view(-1, support_attention_mask.size(-1)) # (B * N * K, max_len)
        query_input_ids = query_input_ids.view(-1, query_input_ids.size(-1)) # (B * N * K, max_len)
        query_attention_mask = query_attention_mask.view(-1, query_attention_mask.size(-1)) # (B * N * K, max_len)

        # embeddings support
        support = self.encoder(support_input_ids, attention_mask=support_attention_mask).hidden_states[-1] # (B * N * K, max_len, h)
        support = self.get_embeddings_entity_start(support, support_marker_pos.view(-1, support_marker_pos.size(-1))) # (B*N*K, 2*h)
        support = self.ff(support) # (B*N*K, h)

        # embeddings query
        query = self.encoder(query_input_ids, attention_mask=query_attention_mask).hidden_states[-1] # (B * N * Q, max_len, h)
        query = self.get_embeddings_entity_start(query, query_marker_pos.view(-1, query_marker_pos.size(-1))) # (B*N*Q, 2*h)
        query = self.ff(query) # (B*N*Q, h)

        # compute similarity
        support = support.view(B, self.N, self.K, self.hidden).unsqueeze(1) # (B, 1, N, K, h)
        query = query.view(B, -1, self.hidden).unsqueeze(2).unsqueeze(2) # (B, N * Q, 1, 1, h)

        logits = (support*query).sum(-1) # (B, N * Q, N, K)
        logits, _ = logits.max(-1) # (B, N * Q, N)
        logits = logits.view(-1, self.N) # (B * N * Q, N)

        out = FewShotOutputs(
            logits=logits,
        )

        return out

    def get_embeddings_entity_start(self, embeddings, marker_pos):
        """
        embeddings (B*N*K, max_len, h)
        marker_pos (B*N*K, 2)
        """

        start_e1 = marker_pos[:,0] # (B*N*K)
        start_e2 = marker_pos[:,1] # (B*N*K)

        e_e1 = embeddings[torch.arange(embeddings.shape[0]),start_e1,:] # (B*N*K, h)
        e_e2 = embeddings[torch.arange(embeddings.shape[0]),start_e2,:] # (B*N*K, h)
        e_cat = torch.cat((e_e1, e_e2), dim=1) # (B*N*K, 2*h)

        return e_cat

    def compute_loss(self, model, inputs, return_outputs=False):
        query_labels = inputs.pop("query_labels")

        out = model(**inputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(out.logits, query_labels.view(-1))

        out = FewShotOutputs(
            loss=loss,
            logits=out.logits.unsqueeze(0),
        )

        return (loss, out) if return_outputs else loss

    def tokenize(self, episode):

        # replace entity_marker
        support_text = [self.replace_entity_marker(i) for i in episode["data_support"]]
        query_text = [self.replace_entity_marker(i) for i in episode["data_query"]]

        # encode inputs
        inputs_support = self.tokenizer(support_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs_query = self.tokenizer(query_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        # labels to tensor
        support_labels = torch.as_tensor(episode["labels_support"], dtype=torch.long)
        query_labels = torch.as_tensor(episode["labels_query"], dtype=torch.long)
        
        # get marker positions
        support_marker_pos = self.get_marker_entity_start(inputs_support["input_ids"])
        query_marker_pos = self.get_marker_entity_start(inputs_query["input_ids"])

        # return dict
        enc = {
            "support_input_ids":inputs_support["input_ids"],
            "support_attention_mask":inputs_support["attention_mask"],
            "support_marker_pos":support_marker_pos,
            "query_input_ids":inputs_query["input_ids"],
            "query_attention_mask":inputs_query["attention_mask"],
            "query_labels":query_labels,
            "query_marker_pos":query_marker_pos,
        }

        return enc
    
    def replace_entity_marker(self, x):
        tokens = []
        for token in x:
            if token in self.entity_marker:
                tokens.append(self.entity_marker[token])
            else:
                tokens.append(token)
        return " ".join(tokens)
    
    def get_marker_entity_start(self, input_ids):
        """
        BERT_EM ENTITY-START variant: Extract marker postions of e1_start and e2_start.
        """
        _, e1_start = torch.where(input_ids == self.entity_marker_ids[0])
        _, e2_start = torch.where(input_ids == self.entity_marker_ids[2])
        return torch.cat((e1_start.unsqueeze(1), e2_start.unsqueeze(1)), dim=1)

# ===========================================================================================

class ProtoModel(BertPreTrainedModel):

    def __init__(
        self,
        N,
        K,
        Q,
        na_rate,
        model_name,
        max_length,
        do_lower_case=True,
        ckpt=None,
        output_hidden_states=False,
        layer_norm=False,
        dropout=0.2,
        ):

        encoder = BertModel.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states
            )

        super().__init__(encoder.config)

        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        
        self.max_length = max_length
        self.do_lower_case = do_lower_case
        self.encoder = encoder
        self.hidden = encoder.config.hidden_size

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=self.do_lower_case)
        self.tokenizer.truncation_side="right"

        if ckpt is not None:
            logger = logging.get_logger("transformers")
            logger.warning(f"{'*'*30}\nLoad model from checkpoint: {ckpt}\n{'*'*30}")
            if ckpt.split(".")[-1] not in ["bin", "pt", "pth"]:
                ckpt = os.path.join(ckpt, "pytorch_model.bin")

            # load from state dict
            self.load_state_dict(torch.load(ckpt))
        
    def forward(
        self,
        support_input_ids,
        support_attention_mask,
        query_input_ids,
        query_attention_mask,
        **kwargs,
        ):
        
        B = support_input_ids.size(0) # batch size

        # reshape inputs
        support_input_ids = support_input_ids.view(-1, support_input_ids.size(-1)) # (B * N * K, max_len)
        support_attention_mask = support_attention_mask.view(-1, support_attention_mask.size(-1)) # (B * N * K, max_len)
        query_input_ids = query_input_ids.view(-1, query_input_ids.size(-1)) # (B * N * K, max_len)
        query_attention_mask = query_attention_mask.view(-1, query_attention_mask.size(-1)) # (B * N * K, max_len)

        # embeddings
        support = self.encoder(support_input_ids, attention_mask=support_attention_mask).pooler_output # (B * N * K, h)
        query = self.encoder(query_input_ids, attention_mask=query_attention_mask).pooler_output # (B * N * Q, h)

        # compute similarity
        support = support.view(B, self.N, self.K, self.hidden).mean(2) # (B, N, h)
        support = support.unsqueeze(1) # (B, 1, N, h)
        query = query.view(B, -1, self.hidden).unsqueeze(2) # (B, N * Q, 1, h)
        logits = (support*query).sum(-1) # (B, N * Q, N)
        logits = logits.view(-1, self.N) # (B * N * Q, N)

        out = FewShotOutputs(
            logits=logits,
        )

        return out

    def compute_loss(self, model, inputs, return_outputs=False):
        query_labels = inputs.pop("query_labels")

        out = model(**inputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(out.logits, query_labels.view(-1))

        out = FewShotOutputs(
            loss=loss,
            logits=out.logits.unsqueeze(0),
        )

        return (loss, out) if return_outputs else loss

    def tokenize(self, episode):

        # encode inputs
        inputs_support = self.tokenizer(episode["data_support"], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs_query = self.tokenizer(episode["data_query"], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        # labels to tensor
        query_labels = torch.as_tensor(episode["labels_query"], dtype=torch.long)

        # return dict
        enc = {
            "support_input_ids":inputs_support["input_ids"],
            "support_attention_mask":inputs_support["attention_mask"],
            "query_input_ids":inputs_query["input_ids"],
            "query_attention_mask":inputs_query["attention_mask"],
            "query_labels":query_labels,
        }

        return enc

# ===========================================================================================

class BERTPAIRModel(BertPreTrainedModel):

    def __init__(
        self,
        N,
        K,
        Q,
        na_rate,
        model_name,
        max_length,
        do_lower_case=True,
        ckpt=None,
        output_hidden_states=True,
        layer_norm=False,
        dropout=0.2,
        ):

        encoder = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            )

        super().__init__(encoder.config)

        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.total_Q = ((N-1)*Q)+(Q*na_rate)
        
        self.max_length = max_length
        self.do_lower_case = do_lower_case
        self.encoder = encoder
        self.hidden = encoder.config.hidden_size

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=self.do_lower_case)
        self.tokenizer.truncation_side="right"
        
        # Pair relation prediction head
        # self.rel_cls = nn.Sequential(
        #     nn.Linear(self.hidden, self.hidden),
        #     nn.Tanh(),
        #     nn.Linear(self.hidden, 2),
        # )

        if ckpt is not None:
            logger = logging.get_logger("transformers")
            logger.warning(f"{'*'*30}\nLoad model from checkpoint: {ckpt}\n{'*'*30}")
            if ckpt.split(".")[-1] not in ["bin", "pt", "pth"]:
                ckpt = os.path.join(ckpt, "pytorch_model.bin")

            # load from state dict
            self.load_state_dict(torch.load(ckpt))

    def forward(
        self,
        batch_input_ids,
        batch_attention_mask,
        **kwargs,
        ):
        
        B = batch_input_ids.size(0) # batch size

        # reshape inputs
        batch_input_ids = batch_input_ids.view(-1, batch_input_ids.size(-1)) # (B * Q * N * K, max_len)
        batch_attention_mask = batch_attention_mask.view(-1, batch_attention_mask.size(-1)) # (B * N * K, max_len)

        # logits
        logits = self.encoder(batch_input_ids, attention_mask=batch_attention_mask)[0] # (B * N * K, h)
        logits = logits[:,1].view(B, self.total_Q, self.N, self.K) # (B, Q, N, K)
        logits = logits.mean(3) # (B, Q, N)
        logits = logits.view(-1, self.N) # (B * Q, N)
        
        out = FewShotOutputs(
            logits=logits,
        )

        return out
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        out = model(**inputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(out.logits, labels.view(-1))

        out = FewShotOutputs(
            loss=loss,
            logits=out.logits.unsqueeze(0),
        )

        return (loss, out) if return_outputs else loss

    def tokenize(self, episode):

        text_query = []
        text_support = []

        for q in episode["data_query"]:
            for s in episode["data_support"]:
                text_query.append(q)
                text_support.append(s)
        
        # encode inputs
        inputs = self.tokenizer(text_support, text_query, padding="max_length", truncation="longest_first", max_length=self.max_length, return_tensors="pt")

        # labels to tensor
        labels = torch.as_tensor(episode["labels_query"], dtype=torch.long)
        
        # return dict
        enc = {
            "batch_input_ids":inputs["input_ids"],
            "batch_attention_mask":inputs["attention_mask"],
            "labels":labels,
        }

        return enc

# ===========================================================================================

class BERTPromptModel(BertPreTrainedModel):
    def __init__(
        self,
        N,
        K,
        Q,
        na_rate,
        model_name,
        max_length,
        template,
        do_lower_case=True,
        ckpt=None,
        output_hidden_states=False,
        layer_norm=False,
        dropout=0.2,
        # dual_target=False
        ):

        encoder = BertForMaskedLM.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states
            )

        super().__init__(encoder.config)

        self.N = N
        self.K = K
        self.Q = Q
        self.template = template
        self.na_rate = na_rate
        self.total_Q = ((N-1)*Q)+(Q*na_rate)
        
        self.max_length = max_length
        self.do_lower_case = do_lower_case
        self.encoder = encoder
        self.hidden = encoder.config.hidden_size

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=self.do_lower_case)
        self.tokenizer.truncation_side="left"

        if ckpt is not None:
            logger = logging.get_logger("transformers")
            logger.warning(f"{'*'*30}\nLoad model from checkpoint: {ckpt}\n{'*'*30}")
            if ckpt.split(".")[-1] not in ["bin", "pt", "pth"]:
                ckpt = os.path.join(ckpt, "pytorch_model.bin")

            # load from state dict
            self.load_state_dict(torch.load(ckpt))

    def forward(
        self,
        batch_input_ids,
        batch_attention_mask,
        mask_pos,
        label_scope,
        **kwargs,
        ):

        # logits
        B = batch_input_ids.size(0) # batch size

        # reshape inputs
        batch_input_ids = batch_input_ids.view(-1, batch_input_ids.size(-1)) # (B * N, max_len)
        batch_attention_mask = batch_attention_mask.view(-1, batch_attention_mask.size(-1)) # (B * N, max_len)
        mask_pos = mask_pos.view(-1, mask_pos.size(-1)) # (B * N, max_len)
        label_scope = label_scope.repeat(int(batch_input_ids.size(0)/label_scope.size(0)), 1) # (B * N, masks)

        logits = self.encoder(batch_input_ids, attention_mask=batch_attention_mask).logits # (B, max_len, vocab)

        B_logits = logits.size(0)
        i = torch.arange(B_logits).reshape(B_logits, 1, 1)
        j = mask_pos.reshape(B_logits, mask_pos.shape[1], 1)
        # k = label_scope.reshape(B_logits, 1, label_scope.shape[1])
        k = torch.unique(label_scope)
        logits = logits[i,j,k].squeeze(1) # (B, masks, N)

        out = FewShotOutputs(
            logits=logits,
        )

        return out

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        out = model(**inputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(out.logits, labels.view(-1))

        out = FewShotOutputs(
            loss=loss,
            logits=out.logits.unsqueeze(0),
        )

        return (loss, out) if return_outputs else loss

    def tokenize(self, episode):

        # prompt
        prompts = [self.getprompt(i) for i in episode["batch_data"]]
        
        # encode inputs
        inputs = self.tokenizer(prompts, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")        
        mask_pos = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1].unsqueeze(-1)

        # labels to tensor
        label_ids = torch.as_tensor(self.tokenizer.convert_tokens_to_ids(episode["batch_labels"]))
        label_scope = torch.unique(torch.as_tensor(self.tokenizer.convert_tokens_to_ids(episode["label_scope"])))

        # # compute label scope for each batch
        # label_scope = torch.unique(label_ids)
        # label_scope = torch.nn.functional.pad(
        #     label_scope,
        #     pad=(0, label_ids.size(-1) - label_scope.size(0)),
        #     mode="constant",
        #     value=-100
        #     ) # pad with -100

        labels = torch.tensor([torch.where(label_scope==i) for i in label_ids]).flatten().unsqueeze(0)
        
        # return dict
        enc = {
            "batch_input_ids":inputs["input_ids"],
            "batch_attention_mask":inputs["attention_mask"],
            "mask_pos":mask_pos,
            "labels":labels,
            "label_scope":label_scope,
        }

        return enc

    def getprompt(self, x, verbose=False):
        template = self.template

        # potential double white spaces
        fill_template = {
            "*sent_0*": x["text"],
            "*e1*": " " + x["e1"],
            "*e2*": " " + x["e2"],
            "*mask*": f" {self.tokenizer.mask_token}",
            "_": " ",
            "*cls*": "",
            "*sep+*": "",
        }

        # fill template
        for k,v in fill_template.items(): 
            template = template.replace(k,v)
        
        if verbose:
            print(template.encode("utf-8"))
        
        return template