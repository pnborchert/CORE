import torch, json, os
import numpy as np
from collections import defaultdict

def load(data_dir, task_name, file_name, model_name, mapping=None):

    def load_mapping(data_dir,task_name,name):
        return json.load(open(os.path.join(data_dir,task_name,name)))

    mapping = None
    # mapping = load_mapping(data_dir, task_name, mapping)

    if model_name in ["bertem", "bertpair"]:

        def _preprocess(x,mapping):
            marker = {
                "e1":{
                    "start": "*START_E1*",
                    "end": "*END_E1*",
                },
                "e2":{
                    "start": "*START_E2*",
                    "end": "*END_E2*",
                },
            }
            x_out = defaultdict(list)
            for rel, v in x["data"].items():
                for i in v:
                    marked_tokens = i["tokens"]
                    
                    # start of e2 > end of e1
                    if i['pos_e2'][0] > i['pos_e1'][-1]:
                        marked_tokens.insert(i['pos_e2'][-1], marker["e2"]["end"])
                        marked_tokens.insert(i['pos_e2'][0], marker["e2"]["start"])
                        marked_tokens.insert(i['pos_e1'][-1], marker["e1"]["end"])
                        marked_tokens.insert(i['pos_e1'][0], marker["e1"]["start"])
                    else:
                        marked_tokens.insert(i['pos_e1'][-1], marker["e1"]["end"])
                        marked_tokens.insert(i['pos_e1'][0], marker["e1"]["start"])
                        marked_tokens.insert(i['pos_e2'][-1], marker["e2"]["end"])
                        marked_tokens.insert(i['pos_e2'][0], marker["e2"]["start"])

                    x_out[rel].append(marked_tokens)
                
            if mapping is not None:
                x_out = {mapping[k]:v for k,v in x_out.items() if k in mapping}
            return x_out
        
    elif model_name in ["hcrp"]:

        def _preprocess(x,mapping):
            marker = {
                "e1":{
                    "start": "*START_E1*",
                    "end": "*END_E1*",
                },
                "e2":{
                    "start": "*START_E2*",
                    "end": "*END_E2*",
                },
            }

            # load relation descriptions
            rel_map = load_mapping(data_dir, task_name, "mapping.json")
            rel_desc = load_mapping(data_dir, task_name, "relation_description.json")
            rel_desc = {rel_map[k]:v for k,v in rel_desc.items() if k in rel_map}

            x_out = defaultdict(list)
            for rel, v in x["data"].items():
                for i in v:
                    marked_tokens = i["tokens"]
                    
                    # start of e2 > end of e1
                    # assert i['pos_e2'][0] > i['pos_e1'][-1], print(rel, i)

                    if i['pos_e2'][0] > i['pos_e1'][-1]:
                        marked_tokens.insert(i['pos_e2'][-1], marker["e2"]["end"])
                        marked_tokens.insert(i['pos_e2'][0], marker["e2"]["start"])
                        marked_tokens.insert(i['pos_e1'][-1], marker["e1"]["end"])
                        marked_tokens.insert(i['pos_e1'][0], marker["e1"]["start"])
                    else:
                        marked_tokens.insert(i['pos_e1'][-1], marker["e1"]["end"])
                        marked_tokens.insert(i['pos_e1'][0], marker["e1"]["start"])
                        marked_tokens.insert(i['pos_e2'][-1], marker["e2"]["end"])
                        marked_tokens.insert(i['pos_e2'][0], marker["e2"]["start"])
                    
                    # add relation description
                    name = rel.replace("_", " ") + ": " + rel_desc[rel]
                    marked_tokens.insert(0, name)

                    x_out[rel].append(marked_tokens)
                
            if mapping is not None:
                x_out = {mapping[k]:v for k,v in x_out.items() if k in mapping}
            return x_out
    
    elif model_name in ["proto"]:

        def _preprocess(x,mapping):
            
            x_out = defaultdict(list)
            for rel, v in x["data"].items():
                for i in v:
                    text = " ".join(i["tokens"])
                    x_out[rel].append(text)
                
            if mapping is not None:
                x_out = {mapping[k]:v for k,v in x_out.items() if k in mapping}
            return x_out

    elif model_name == "bertprompt":

        def _preprocess(x,mapping):
            
            x_out = defaultdict(list)
            for rel, v in x["data"].items():
                for i in v:
                    tokens = i.pop("tokens")
                    i["text"] = " ".join(tokens)
                    x_out[rel].append(i)
                
            if mapping is not None:
                x_out = {mapping[k]:v for k,v in x_out.items() if k in mapping}
            return x_out
        
    else:
        NotImplementedError
        
    return _preprocess(json.load(open(os.path.join(data_dir,task_name, file_name))), mapping)

class FSDataset(torch.utils.data.Dataset):
    def __init__(self, model, data, N, K, Q, na_rate, na_relations=["unrelated"], support_data=None, shuffle=True, eval_max_steps=10000, sampling="uniform"):
        self.model = model
        self.data = data
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.support_data = support_data
        self.sampling = sampling
        self.na_relations = na_relations

        self.relation_mapping = dict(zip(list(self.data.keys()), range(len(self.data.keys())))) # only relevant for shuffle = False
        self.relations = [i for i in self.data.keys() if i not in self.na_relations]
        # self.relations = list(self.data.keys())
        self.shuffle = shuffle
        self.eval_max_steps = eval_max_steps

        if self.na_rate > 0:
            self.Nrel = self.N
            self.N += 1
        else:
            self.Nrel = self.N
        
        if sampling == "random":
            # create flat list of dicts
            self.data_flat = []
            self.lookup_data_flat = {}

            for k, v in self.data.items():
                if (k in self.na_relations) and (self.na_rate == 0):
                    # do not add NA relation
                    continue
                self.lookup_data_flat[k] = {}

                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        data_dict = item.copy()
                        data_dict["relation"] = k
                        self.data_flat.append(data_dict)
                    else:
                        data_dict = {"text": item, "relation": k}
                        self.data_flat.append(data_dict)
                    self.lookup_data_flat[k][i] = len(self.data_flat)-1
                    

    def get_episode(self):
        
        if self.sampling == "uniform":
            return self.get_episode_uniform()
        elif self.sampling == "random":
            return self.get_episode_random()

    def get_episode_uniform(self):
        """
        Sample random episode uniformly from N relations: 
        support (N * K), query (N * Q)

        If na_rate > 0: 
        support (N+1 * K), query (N * Q + Q * na_rate) 
        """

        data_support = []
        data_query = []
        label_support = []
        label_query = []

        if self.shuffle:
            rel = np.random.choice(self.relations, self.Nrel, replace=False)
        else:
            rel = self.relations.copy()
        
        for index, r in enumerate(rel):
            index_support = np.random.choice(range(len(self.data[r])), self.K, replace=False)
            index_query = np.random.choice([i for i in range(len(self.data[r])) if i not in index_support], self.Q, replace=False)

            # support
            data_support += [self.data[r][i] for i in index_support]
            label_support += [index] * self.K

            # query
            data_query += [self.data[r][i] for i in index_query]
            label_query += [index] * self.Q

        # NA
        if self.na_rate > 0:

            # support (NA = K)
            indices_support = defaultdict(list)
            for _ in range(self.K):
                na_rel = np.random.choice([i for i in self.data.keys() if i not in rel], 1).item()
                # na_rel = np.random.choice(self.na_relations, 1)
                index_support = np.random.choice([i for i in range(len(self.data[na_rel])) if i not in indices_support[na_rel]], 1).item()
                indices_support[na_rel].append(index_support)
                data_support.append(self.data[na_rel][index_support])
            label_support += [self.Nrel] * self.K

            # query (NA = Q*na_rate)
            indices_query = defaultdict(list)
            for _ in range(self.Q*self.na_rate):
                na_rel = np.random.choice([i for i in self.data.keys() if i not in rel], 1).item()
                # na_rel = np.random.choice(self.na_relations, 1)
                index_query = np.random.choice([i for i in range(len(self.data[na_rel])) if i not in indices_query[na_rel]], 1).item()
                indices_query[na_rel].append(index_query)
                data_query.append(self.data[na_rel][index_query])
            label_query += [self.Nrel] * (self.Q*self.na_rate)

        episode = {
            "data_support": data_support,
            "labels_support": label_support,
            "data_query": data_query,
            "labels_query": label_query,
        }

        return episode

    def get_episode_random(self):
        """
        Sample random queries from all instances: 
        support (N * K), query (N * Q)

        If na_rate > 0: 
        support (N+1 * K), query (N * Q + Q * na_rate) 
        """

        data_support = []
        data_query = []
        label_support = []
        label_query = []

        if self.shuffle:
            rel = np.random.choice(self.relations, self.Nrel, replace=False)
        else:
            rel = self.relations.copy()
        

        support_indices_flat = []

        # support
        for index, r in enumerate(rel):
            index_support = np.random.choice(range(len(self.data[r])), self.K, replace=False)
            support_indices_flat += [self.lookup_data_flat[r][i] for i in index_support]

            # support
            data_support += [self.data[r][i] for i in index_support]
            label_support += [index] * self.K

        # NA
        if self.na_rate > 0:

            # support (NA = K)
            indices_support = defaultdict(list)
            for i in range(self.K):
                na_rel = np.random.choice([i for i in self.data.keys() if i not in rel], 1).item()
                # na_rel = np.random.choice(self.na_relations, 1)
                index_support = np.random.choice([i for i in range(len(self.data[na_rel])) if i not in indices_support[na_rel]], 1).item()
                indices_support[na_rel].append(index_support)
                support_indices_flat.append(self.lookup_data_flat[na_rel][index_support])
                data_support.append(self.data[na_rel][index_support])
            label_support += [self.Nrel] * self.K
        
        # query (NA = Q*na_rate)
        indices = np.random.choice([i for i in range(len(self.data_flat)) if i not in support_indices_flat], self.Nrel*self.Q+self.Q*self.na_rate, replace=False)
        for index in indices:
            data_query.append(self.data_flat[index]["text"])
            q_label = list(rel).index(self.data_flat[index]["relation"]) if self.data_flat[index]["relation"] in rel else self.Nrel
            label_query.append(q_label)

        episode = {
            "data_support": data_support,
            "labels_support": label_support,
            "data_query": data_query,
            "labels_query": label_query,
        }

        return episode
    
    def tokenize(self, episode):
        return self.model.tokenize(episode)

    def __getitem__(self, index):
        return self.tokenize(self.get_episode())
    
    def __len__(self):
        return self.eval_max_steps

class BERTEMDataset(FSDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ProtoDataset(FSDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BERTPAIRDataset(FSDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HCRPDataset(FSDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BERTPromptDataset(FSDataset):
    def __init__(self, model, data, N, K, Q, na_rate, na_relations=["unrelated"], shuffle=False, eval_max_steps=10000, sampling="uniform", desc=None, fs_seed=None):
        self.model = model
        self.data = data
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.na_relations = na_relations
        self.sampling = sampling
        self.desc = desc
        self.fs_seed = fs_seed

        self.relation_mapping = dict(zip(list(self.data.keys()), range(len(self.data.keys())))) # only relevant for shuffle = False
        self.relations = [i for i in self.data.keys() if i not in self.na_relations]
        # self.relations = list(self.data.keys())
        self.shuffle = shuffle
        self.eval_max_steps = eval_max_steps


        if (self.na_rate > 0):
            self.Nrel = self.N
            self.N += 1
            self.label_scope = self.relations + self.na_relations
        else:
            self.Nrel = self.N
            self.label_scope = self.relations
        
        if sampling == "random":
            # create flat list of dicts
            self.data_flat = []
            self.lookup_data_flat = {}

            for k, v in self.data.items():
                if (k in self.na_relations) and (self.na_rate == 0):
                    # do not add NA relation
                    continue
                self.lookup_data_flat[k] = {}

                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        data_dict = item.copy()
                        data_dict["relation"] = k
                        self.data_flat.append(data_dict)
                    else:
                        data_dict = {"text": item, "relation": k}
                        self.data_flat.append(data_dict)
                    self.lookup_data_flat[k][i] = len(self.data_flat)-1
        
        # set seed episode
        if self.fs_seed is not None:
            np.random.seed(self.fs_seed)
            self.seed_episode = self.tokenize(self.get_episode())

    def __getitem__(self, index):
        if self.fs_seed is not None:
            return self.seed_episode
        else:
            return self.tokenize(self.get_episode())

    def get_episode(self):
        
        if self.sampling == "uniform":
            return self.get_episode_uniform()
        elif self.sampling == "random":
            return self.get_episode_random()     
    
    def get_episode_uniform(self):
        """
        Sample random episode uniformly from N relations: 
        support (N * K), query (N * Q)

        If na_rate > 0: 
        support (N+1 * K), query (N * Q + Q * na_rate) 
        """

        batch_data = []
        batch_label = []

        if self.shuffle:
            rel = np.random.choice(self.relations, self.Nrel, replace=False)
        else:
            rel = self.relations.copy()
        
        if self.desc == "train":

            for index, r in enumerate(rel):
                indices = np.random.choice(range(len(self.data[r])), self.K, replace=False)
                batch_data += [self.data[r][i] for i in indices]
                batch_label += [r] * self.K

            # NA
            if self.na_rate > 0:

                # support (NA = K)
                indices_support = defaultdict(list)
                for i in range(self.K):
                    na_rel = np.random.choice([i for i in self.data.keys() if i not in rel], 1).item()
                    # na_rel = np.random.choice(self.na_relations, 1)
                    index_support = np.random.choice([i for i in range(len(self.data[na_rel])) if i not in indices_support[na_rel]], 1).item()
                    indices_support[na_rel].append(index_support)
                    batch_data.append(self.data[na_rel][index_support])
                batch_label += [self.na_relations[0]] * self.K


        elif self.desc == "test":

            for index, r in enumerate(rel):
                indices = np.random.choice(range(len(self.data[r])), self.Q, replace=False)
                batch_data += [self.data[r][i] for i in indices]
                batch_label += [r] * self.Q

            # NA
            if self.na_rate > 0:
                # query (NA = Q*na_rate)
                indices_query = defaultdict(list)
                for i in range(self.Q*self.na_rate):
                    na_rel = np.random.choice([i for i in self.data.keys() if i not in rel], 1).item()
                    # na_rel = np.random.choice(self.na_relations, 1)
                    index_query = np.random.choice([i for i in range(len(self.data[na_rel])) if i not in indices_query[na_rel]], 1).item()
                    indices_query[na_rel].append(index_query)
                    batch_data.append(self.data[na_rel][index_query])
                batch_label += [self.na_relations[0]] * (self.Q * self.na_rate)

        episode = {
            "batch_data": batch_data,
            "batch_labels": batch_label,
            "label_scope": self.label_scope,
        }

        return episode
    
    def get_episode_random(self):
        """
        Sample random queries from all instances: 
        support (N * K), query (N * Q)

        If na_rate > 0: 
        support (N+1 * K), query (N * Q + Q * na_rate) 
        """
        
        batch_data = []
        batch_label = []

        if self.shuffle:
            rel = np.random.choice(self.relations, self.Nrel, replace=False)
        else:
            rel = self.relations.copy()
        
        if self.desc == "train":

            for index, r in enumerate(rel):
                indices = np.random.choice(range(len(self.data[r])), self.K, replace=False)
                batch_data += [self.data[r][i] for i in indices]
                batch_label += [r] * self.K

            # NA
            if self.na_rate > 0:
                # support (NA = K)
                indices_support = defaultdict(list)
                for i in range(self.K):
                    na_rel = np.random.choice([i for i in self.data.keys() if i not in rel], 1).item()
                    # na_rel = np.random.choice(self.na_relations, 1)
                    index_support = np.random.choice([i for i in range(len(self.data[na_rel])) if i not in indices_support[na_rel]], 1).item()
                    indices_support[na_rel].append(index_support)
                    batch_data.append(self.data[na_rel][index_support])
                batch_label += [self.na_relations[0]] * self.K

        if self.desc == "test":
                
            # query (NA = Q*na_rate)
            indices = np.random.choice(range(len(self.data_flat)), (self.Nrel*self.Q)+(self.Q*self.na_rate), replace=False)
            for index in indices:
                batch_data.append(self.data_flat[index])
                batch_label.append(self.data_flat[index]["relation"])

        episode = {
            "batch_data": batch_data,
            "batch_labels": batch_label,
            "label_scope": self.label_scope,
        }

        return episode
    