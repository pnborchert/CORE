import argparse, os, warnings, wandb, json, torch
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
from transformers.utils import logging
from sklearn.metrics import *
import numpy as np
from fs import model, dataset
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--task_name', help='', type=str, default="core")
parser.add_argument('--data_dir', help='', type=str, default="./data")
parser.add_argument('--mapping', help='', type=str, default="mapping.json")

parser.add_argument('--model_name', help='', type=str, required=True)
parser.add_argument('--seed', help='', type=int, required=True)
parser.add_argument('--N', help='', type=int, required=True)
parser.add_argument('--K', help='', type=int, required=True)
parser.add_argument('--Q', help='', type=int, required=True)

parser.add_argument('--encoder', help='', type=str, default="bert-base-uncased")
parser.add_argument('--sampling', help='', type=str, default="random")
parser.add_argument('--max_len', help='', type=int, default=256)
parser.add_argument('--batch_size', help='', type=int, default=1)
parser.add_argument('--na_rate', help='', type=int, default=2)
parser.add_argument('--gradient_accumulation_steps', help='', type=int, default=64)
parser.add_argument('--max_steps', help='', type=int, default=500)
parser.add_argument('--eval_steps', help='', type=int, default=100)
parser.add_argument('--eval_max_steps', help='', type=int, default=100)
parser.add_argument('--test_max_steps', help='', type=int, default=3500)
parser.add_argument('--ckpt', help='', type=str, default=None)
parser.add_argument('--do_lower_case', dest="do_lower_case", action='store_true', help='')
parser.add_argument('--fp16', dest="fp16", action='store_true', help='')
parser.add_argument('--lr', help='', type=float, default=2e-5)
parser.add_argument('--output_dir', help='', type=str, default="./runs")
parser.add_argument('--wandb_project', help='', type=str, default="CORE")
parser.add_argument('--wandb_dir', help='', type=str, default="./wandb")

parser.add_argument('--fs_seed', help='', type=int, default=None)
parser.add_argument('--eval_only', dest="eval_only", action='store_true', help='')

args = vars(parser.parse_args())

def resolve_args(args):

    args["output_dir"] = os.path.join(args["output_dir"], args["task_name"], args["model_name"], f"NA{args['na_rate']}_{args['sampling']}")

    args["train_suffix"] = os.path.join("episodes", f"{args['N']}-{args['seed']}-train.json")
    args["train_fs_suffix"] = os.path.join("episodes", f"{args['N']}-{args['seed']}-train_fs.json")
    args["test_suffix"] = os.path.join("episodes", f"{args['N']}-{args['seed']}-test.json")

    args["wandb_suffix"] = f"{args['model_name']}_NA{args['na_rate']}_{args['sampling']}"
    args["run_name"] = f"{args['model_name']}_{args['N']}_{args['K']}_{args['seed']}"

    args["Nrel"] = args["N"]

    if args["na_rate"] > 0:
        args["N"] += 1

    if args["model_name"] == "bertem":
        args["model_cls"] = model.BERTEMModel
        args["dataset_cls"] = dataset.BERTEMDataset
        args["label_names"] = ["query_labels"]

    elif args["model_name"] == "proto":
        args["model_cls"] = model.ProtoModel
        args["dataset_cls"] = dataset.ProtoDataset
        args["label_names"] = ["query_labels"]

    elif args["model_name"] == "bertpair":
        args["model_cls"] = model.BERTPAIRModel
        args["dataset_cls"] = dataset.BERTPAIRDataset
        args["label_names"] = ["labels"]
        args["fp16"] = True
    
    elif args["model_name"] == "hcrp":
        args["model_cls"] = model.HCRP
        args["dataset_cls"] = dataset.HCRPDataset
        args["label_names"] = ["query_labels"]

    elif args["model_name"] == "bertprompt":
        args["model_cls"] = model.BERTPromptModel
        args["dataset_cls"] = dataset.BERTPromptDataset
        args["label_names"] = ["labels", "label_scope"]
        if args["fs_seed"] is not None:
            args["wandb_suffix"] += "_FS"
            args["run_name"] += f"_FS{args['fs_seed']}"
        else:
            args["wandb_suffix"] += "_ZS"
            args["run_name"] += "_ZS"

    else:
        raise NotImplementedError


    return args

def get_metrics(args):
    if args["model_name"] in ["bertem", "proto", "bertpair", "hcrp"]:
        return compute_metrics_1, acc_at_relation_1
    elif args["model_name"] in ["bertprompt"]:
        return compute_metrics_2, acc_at_relation_2
    else:
        NotImplementedError


# evaluation metrics
def compute_metrics_1(eval_preds):
    logits, labels = eval_preds
    labels = labels.reshape(-1)

    pred = np.argmax(logits, axis=-1).reshape(-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc         = accuracy_score(labels, pred)
        f1          = f1_score(labels, pred, average="micro")

    return {
        "Accuracy": acc,
        'F1': f1,
    }

# evaluation metrics
def compute_metrics_2(eval_preds):
    logits, (labels, _) = eval_preds
    labels = labels.reshape(-1)

    pred = np.argmax(logits, axis=-1).reshape(-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc         = accuracy_score(labels, pred)
        f1          = f1_score(labels, pred, average="micro")

    return {
        "Accuracy": acc,
        'F1': f1,
    }

class FewShotTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model.compute_loss(
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
            )

def acc_at_relation_1(trainer, ds, args):
    mapping = ds.relation_mapping

    out = trainer.predict(ds)
    preds = out.predictions
    preds = preds.reshape(-1, preds.shape[-1])
    labels = out.label_ids.reshape(-1)

    res = dict(out.metrics)
    for word, target in zip(mapping, np.unique(labels)):
        mask = np.where(np.hstack(out.label_ids) == target)
        pred_word = np.vstack(np.argmax(preds[mask,:], axis=-1)[0])
        labels_word = labels[mask]
        acc = accuracy_score(labels_word, pred_word)
        res[f"Acc@{word}"] = acc
    
    return res

def acc_at_relation_2(trainer, ds, args):
    out = trainer.predict(ds)

    labels, label_scope = out.label_ids
    predictions = out.predictions

    labels = labels.reshape(-1, labels.shape[-1])
    predictions = predictions.reshape(-1, labels.shape[-1], predictions.shape[-1])

    res = dict(out.metrics)
    res_at_rel = defaultdict(list)

    label_words = np.array([trainer.model.tokenizer.convert_ids_to_tokens(i) for i in label_scope])
    for i in range(label_words.shape[0]):
        lw = label_words[i,:]
        l = labels[i,:]
        p = predictions[i,:]
        # for word, target in zip(lw, np.unique(np.hstack(l))):
        for target, word in enumerate(lw):
            mask = np.where(np.hstack(l) == target)[0]
            if len(mask) == 0: continue # row does not contain label
            preds = np.vstack(np.argmax(p[mask,:], axis=-1))
            l_mask = l[mask]
            acc = accuracy_score(l_mask, preds)
            res_at_rel[f"Acc@{word}"].append((acc, len(l_mask)))
    
    for k,v in res_at_rel.items():
        res[k] = np.sum([i[0]*i[1] for i in v]) / np.sum([i[1] for i in v])
    
    return res

def evaluate(eval_fct, trainer, ds, args):
    out = eval_fct(trainer,ds,args)
    wandb.log({"test":wandb.Table(data=list(out.items()),columns = ["metric", "value"])})
    print(out)

def get_dataloader(args):

    if args["model_name"] in ["bertem", "proto", "bertpair", "hcrp"]:

        train = dataset.load(
            data_dir=args["data_dir"],
            task_name=args["task_name"],
            file_name=args["train_suffix"],
            model_name=args["model_name"],
            mapping=args["mapping"],
        )

        test = dataset.load(
            data_dir=args["data_dir"],
            task_name=args["task_name"],
            file_name=args["test_suffix"],
            model_name=args["model_name"],
            mapping=args["mapping"],
        )

        model = args["model_cls"](
            model_name=args["encoder"],
            max_length=args["max_len"],
            do_lower_case=args["do_lower_case"],
            ckpt=args["ckpt"],
            N=args["N"],
            K=args["K"],
            Q=args["Q"],
            na_rate=args["na_rate"],
        )

        ds_train = args["dataset_cls"](
            model = model,
            data = train,
            N = args["Nrel"],
            K = args["K"],
            Q = args["Q"],
            shuffle = True,
            na_rate = args["na_rate"],
            sampling = args["sampling"],
        )

        ds_valid = args["dataset_cls"](
            model = model,
            data = train,
            N = args["Nrel"],
            K = args["K"],
            Q = args["Q"],
            shuffle = False,
            eval_max_steps = args["eval_max_steps"],
            na_rate = args["na_rate"],
            sampling = args["sampling"],
        )

        ds_test = args["dataset_cls"](
            model = model,
            data = test,
            N = args["Nrel"],
            K = args["K"],
            Q = args["Q"],
            shuffle = False,
            eval_max_steps = args["test_max_steps"],
            na_rate = args["na_rate"],
            sampling = args["sampling"],
        )

    elif args["model_name"] in ["bertprompt"]:

        def load_template(args):
            with open(os.path.join(args["data_dir"], args["task_name"], f"template.txt"), encoding="utf-8") as fp:
                lines = fp.readlines()
            lines = [l.replace("\n", "") for l in lines]
            return lines
        
        templates = load_template(args)
        template = templates[0]

        if args["fs_seed"] is not None:

            logger = logging.get_logger("transformers")
            logger.warning(f"{'*'*30}\n{args['N']}-Way {args['K']}-Shot Setting\n{'*'*30}")

            train = dataset.load(
                data_dir=args["data_dir"],
                task_name=args["task_name"],
                file_name=args["train_fs_suffix"],
                model_name=args["model_name"],
                mapping=args["mapping"],
            )

            test = dataset.load(
                data_dir=args["data_dir"],
                task_name=args["task_name"],
                file_name=args["test_suffix"],
                model_name=args["model_name"],
                mapping=args["mapping"],
            )

            model = args["model_cls"](
                model_name=args["encoder"],
                max_length=args["max_len"],
                do_lower_case=args["do_lower_case"],
                ckpt=args["ckpt"],
                N=args["N"],
                K=args["K"],
                Q=args["Q"],
                na_rate=args["na_rate"],
                template=template,
            )

            ds_train = args["dataset_cls"](
                model = model,
                data = train,
                N = args["Nrel"],
                K = args["K"],
                Q = args["Q"],
                shuffle = True,
                na_rate = args["na_rate"],
                sampling = args["sampling"],
                desc="train",
                fs_seed=args["fs_seed"] # seed for train_fs episode
            )

            ds_valid = args["dataset_cls"](
                model = model,
                data = train,
                N = args["Nrel"],
                K = args["K"],
                Q = args["Q"],
                shuffle = True,
                eval_max_steps = args["eval_max_steps"],
                na_rate = args["na_rate"],
                sampling = args["sampling"],
                desc="train"
            )

            ds_test = args["dataset_cls"](
                model = model,
                data = test,
                N = args["Nrel"],
                K = args["K"],
                Q = args["Q"],
                shuffle = False,
                eval_max_steps = args["test_max_steps"],
                na_rate = args["na_rate"],
                sampling = args["sampling"],
                desc="test"
            )

        else:

            logger = logging.get_logger("transformers")
            logger.warning(f"{'*'*30}\nZero-Shot Setting\n{'*'*30}")


            train = dataset.load(
                data_dir=args["data_dir"],
                task_name=args["task_name"],
                file_name=args["train_suffix"],
                model_name=args["model_name"],
                mapping=args["mapping"],
            )

            test = dataset.load(
                data_dir=args["data_dir"],
                task_name=args["task_name"],
                file_name=args["test_suffix"],
                model_name=args["model_name"],
                mapping=args["mapping"],
            )

            model = args["model_cls"](
                model_name=args["encoder"],
                max_length=args["max_len"],
                do_lower_case=args["do_lower_case"],
                ckpt=args["ckpt"],
                N=args["N"],
                K=args["K"],
                Q=args["Q"],
                na_rate=args["na_rate"],
                template=template,
            )

            ds_train = args["dataset_cls"](
                model = model,
                data = train,
                N = args["Nrel"],
                K = args["K"],
                Q = args["Q"],
                shuffle = True,
                na_rate = args["na_rate"],
                sampling = args["sampling"],
                desc="train",
                fs_seed=args["fs_seed"]
            )

            ds_valid = args["dataset_cls"](
                model = model,
                data = train,
                N = args["Nrel"],
                K = args["K"],
                Q = args["Q"],
                shuffle = True,
                eval_max_steps = args["eval_max_steps"],
                na_rate = args["na_rate"],
                sampling = args["sampling"],
                desc="train"
            )

            ds_test = args["dataset_cls"](
                model = model,
                data = test,
                N = args["Nrel"],
                K = args["K"],
                Q = args["Q"],
                shuffle = False,
                eval_max_steps = args["test_max_steps"],
                na_rate = args["na_rate"],
                sampling = args["sampling"],
                desc="test"
            )

    else:
        raise NotImplementedError
    
    return ds_train, ds_valid, ds_test, model

        

def main(args):

    ds_train, ds_valid, ds_test, model = get_dataloader(args=args)
    compute_metrics, acc_at_relation = get_metrics(args=args)

    wandb.init(
        project=f'{args["wandb_project"]}_{args["wandb_suffix"]}',
        name=args["run_name"],
        dir=args["wandb_dir"],
        reinit=True,
    )

    # setup trainer
    training_args = TrainingArguments(
        output_dir=args["output_dir"],
        do_train=True,
        do_eval=True,
        do_predict=True,
        remove_unused_columns=False,
        evaluation_strategy="steps", 
        eval_steps = args["eval_steps"] ,
        per_device_train_batch_size=args["batch_size"],
        per_device_eval_batch_size=args["batch_size"],
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
        learning_rate=args["lr"],
        max_steps = args["max_steps"] ,
        lr_scheduler_type="linear",
        save_strategy="no",
        save_total_limit=1,
        label_names=args["label_names"],
        report_to="wandb",
        run_name=args["run_name"],
        fp16=args["fp16"],
    )

    trainer = FewShotTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        compute_metrics=compute_metrics,
    )

    if not args["eval_only"]:
        trainer.train()

        # save model
        trainer.model.save_pretrained(os.path.join(trainer.args.output_dir, args["run_name"]))

    # evaluate: few-shot
    evaluate(
        eval_fct=acc_at_relation,
        trainer=trainer,
        ds=ds_test,
        args=args
        )

    wandb.finish()

if __name__ == "__main__":
    args = resolve_args(args)
    main(args)