import argparse, json ,os
from collections import defaultdict
import numpy as np

# parse input arguments
parser=argparse.ArgumentParser()

parser.add_argument('--data_dir', help='', type=str, default="./data")
parser.add_argument('--train', help='', type=str, default="train_wiki.json")
parser.add_argument('--test', help='', type=str, default="val_wiki.json")
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
    if args["task_name"] == "fewrel":
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


def main(args):
    train = load_dataset(args["train"], args)
    test = load_dataset(args["test"], args)

    assert all([i not in train.keys() for i in test.keys()])
    all_train_relations = list(train.keys())
    all_test_relations = list(test.keys())

    print(all_train_relations)
    print(all_test_relations)

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
        rel_train = rng.choice(all_train_relations, args["N"], replace=False)
        rel_train_na = [i for i in all_train_relations if i not in list(rel_train)]

        rel_test = rng.choice(all_test_relations, args["N"], replace=False)
        rel_test_na = [i for i in all_test_relations if i not in list(rel_test)]
        
        indices_fs = {}
        for rel in rel_test:
            indices_fs[rel] = rng.choice(range(len(test[rel])), 100, replace=False)

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
            configuration_train_fs["data"][rel] = [row for i, row in enumerate(test[rel]) if i in indices_fs[rel]]

        # test data
        configuration_test = _get_base_config()
        configuration_test["meta"]["split"] = "test"
        configuration_test["meta"]["file_name"] = args["test"]
        configuration_test["meta"]["seed"] = seed
        for rel in rel_test:
            configuration_test["data"][rel] = [row for i, row in enumerate(test[rel]) if i not in indices_fs[rel]]
        
        if args["include_na"] is True:

            # add unused relations
            for rel in rel_train_na:
                configuration_train["data"][args['na']] += train[rel]
                configuration_train_fs["data"][args['na']] += train[rel]
            
            for rel in rel_test_na:
                configuration_test["data"][args['na']] += test[rel]


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