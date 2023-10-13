import argparse, json ,os
from collections import defaultdict
import numpy as np

# parse input arguments
parser=argparse.ArgumentParser()

parser.add_argument('--data_dir', help='', type=str, default="./data")
parser.add_argument('--train', help='', type=str, default="train.json")
parser.add_argument('--test', help='', type=str, default="test.json")
parser.add_argument('--mapping', help='', type=str, default="mapping.json")
parser.add_argument('--na', help='', type=str, default="unrelated")
parser.add_argument('--config_folder', help='', type=str, default="episodes")

parser.add_argument('--task_name', help='', type=str, required=True)
parser.add_argument('--N', help='', type=int, required=True)
parser.add_argument('--include_na', dest='include_na', help='', action="store_true")

parser.add_argument('--seeds', help='', type=int, nargs='+', required=True)

args=vars(parser.parse_args())

def load_dataset(file_name, args):

    # load mapping
    def load_mapping(data_dir,task_name,name):
        return json.load(open(os.path.join(data_dir,task_name,name)))

    mapping = load_mapping(args["data_dir"], args["task_name"], args["mapping"])

    # load dataset
    if args["task_name"] == "core":
        def _preprocess(x,mapping):
            x_out = defaultdict(list)
            for i in x:
                # swap e1, e2 for inverted relations
                e1 = "e1"
                e2 = "e2"
                if i["invert_relation"] == 1:
                    e1 = "e2"
                    e2 = "e1"
                x_out[i["relation"]].append({
                    "tokens":i["context"],
                    e1:" ".join(i["context"][i["e1_start"]:i["e1_end"]+1]),
                    e2:" ".join(i["context"][i["e2_start"]:i["e2_end"]+1]),
                    f"pos_{e1}":[j for j in range(i["e1_start"],i["e1_end"]+1)],
                    f"pos_{e2}":[j for j in range(i["e2_start"],i["e2_end"]+1)],
                })
            if mapping is not None:
                x_out = {mapping[k]:v for k,v in x_out.items() if k in mapping}
            return x_out

        return _preprocess(json.load(open(os.path.join(args["data_dir"],args["task_name"], file_name))), mapping)

    elif args["task_name"] == "fewrel":
        def _preprocess(x,mapping):
            x_out = defaultdict(list)
            for k, v in x.items():
                for i in v:
                    x_out[k].append({
                        "tokens":i["tokens"],
                        "e1":" ".join([i["tokens"][j] for j in i['h'][-1][0]]),
                        "e2":" ".join([i["tokens"][j] for j in i['t'][-1][0]]),
                        "pos_e1":i['h'][-1][0],
                        "pos_e2":i['t'][-1][0],
                    })
            if mapping is not None:
                x_out = {mapping[k]:v for k,v in x_out.items() if k in mapping}
            return x_out

        return _preprocess(json.load(open(os.path.join(args["data_dir"],args["task_name"], file_name))), mapping)
    
    elif args["task_name"] == "tacred":
        def _preprocess(x,mapping):
            x_out = defaultdict(list)
            for i in x:
                x_out[i["relation"]].append({
                    "tokens":i["token"],
                    "e1":" ".join(i["token"][i["subj_start"]:i["subj_end"]+1]),
                    "e2":" ".join(i["token"][i["obj_start"]:i["obj_end"]+1]),
                    "pos_e1":[i for i in range(i["subj_start"],i["subj_end"]+1)],
                    "pos_e2":[i for i in range(i["obj_start"],i["obj_end"]+1)],
                })
            if mapping is not None:
                x_map = defaultdict(list)
                for k,v in x_out.items():
                    if k in mapping:
                        x_map[mapping[k]] += v
                x_out = x_map
                # x_out = {mapping[k]:v for k,v in x_out.items() if k in mapping}
            return x_out

        return _preprocess(json.load(open(os.path.join(args["data_dir"],args["task_name"], file_name))), mapping)
    
    else:
        raise NotImplementedError


def main(args):
    train = load_dataset(args["train"], args)
    test = load_dataset(args["test"], args)

    assert len([i for i in train.keys() if i not in test.keys()]) == 0
    all_relations = list(train.keys())
    na_relation = args["na"]
    nonna_relations = [i for i in all_relations if i != na_relation]

    def _get_base_config():
        configuration = {
            "meta":{"task_name":args["task_name"]},
            "data":defaultdict(list),
        }
        return configuration
    
    print("="*30)
    print(f"Train file: {os.path.join(args['data_dir'], args['task_name'], args['train'])}")
    print(f"Test file: {os.path.join(args['data_dir'], args['task_name'], args['test'])}")
    print(f"Creating {len(args['seeds'])} episodes")
    print(f"Seeds: {args['seeds']}")
    print(f"N: {args['N']}")
    print(f"Include NA: {args['include_na']}")
    print(f"NA relation: {args['na']}")
    print(f"Output folder: {os.path.join(args['data_dir'], args['task_name'], args['config_folder'])}")
    print("="*30)

    for seed in args["seeds"]:
        rng = np.random.RandomState(seed)
        rel_train = rng.choice(nonna_relations, args["N"], replace=False)
        rel_test = rng.choice([i for i in nonna_relations if i not in rel_train], args["N"], replace=False)
        rel_na = [i for i in all_relations if i not in list(rel_train) + list(rel_test)]

        #train data
        configuration_train = _get_base_config()
        configuration_train["meta"]["split"] = "train"
        configuration_train["meta"]["file_name"] = args["train"]
        configuration_train["meta"]["seed"] = seed
        for rel in rel_train:
            configuration_train["data"][rel] = train[rel]
        
        # train data with test relations (few-shot data)
        configuration_train_fs = _get_base_config()
        configuration_train_fs["meta"]["split"] = "train_fs"
        configuration_train_fs["meta"]["file_name"] = args["train"]
        configuration_train_fs["meta"]["seed"] = seed
        for rel in rel_test:
            configuration_train_fs["data"][rel] = train[rel]

        # test data
        configuration_test = _get_base_config()
        configuration_test["meta"]["split"] = "test"
        configuration_test["meta"]["file_name"] = args["test"]
        configuration_test["meta"]["seed"] = seed
        for rel in rel_test:
            configuration_test["data"][rel] = test[rel]
        
        if args["include_na"] is True:

            # na relation
            if args["na"] != "":
                configuration_train["meta"]["na"] = na_relation
                configuration_train["data"][na_relation] += train[na_relation]

                configuration_train_fs["meta"]["na"] = na_relation
                configuration_train_fs["data"][na_relation] += train[na_relation]

                configuration_test["meta"]["na"] = na_relation
                configuration_test["data"][na_relation] += test[na_relation]
            else:
                na_relation = "unrelated"

            # add unused relations
            for rel in rel_na:
                configuration_train["data"][na_relation] += train[rel]
                configuration_train_fs["data"][na_relation] += train[rel]
                configuration_test["data"][na_relation] += test[rel]


        #save configuration -> data_dir/task_name/config_folder/file_name
        path = os.path.join(args["data_dir"], args["task_name"], args["config_folder"])
        os.makedirs(path, exist_ok=True)
        #train
        json.dump(configuration_train,open(os.path.join(path,f"{args['N']}-{seed}-train.json"), "w"))
        #train fs
        json.dump(configuration_train_fs,open(os.path.join(path,f"{args['N']}-{seed}-train_fs.json"), "w"))
        #test
        json.dump(configuration_test,open(os.path.join(path,f"{args['N']}-{seed}-test.json"), "w"))
        
if __name__ == '__main__':
    main(args)