import os
import json
import random
import numpy as np
from CascadeXML import CascadeXML
import torch
import argparse
import traceback
from dataset import InferenceDataset
from data_utils import get_tokenizer
from dist_eval_sampler import DistributedEvalSampler

NUM_LABELS = {'Amazon-670K': 670091, 'Amazon-3M': 2812281, 'Wiki-500K' : 501070, 'AmazonCat-13K': 13330, 'Wiki10-31K': 30938, 'Eurlex': 3993, 'AT670': 670091, 'WT500': 501070, 'WSAT350': 352072}
def init_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model(model, name):
    checkpoint = torch.load(name, map_location=torch.device('cpu'))
    if 'state_dict' in checkpoint:
        # checkpoint has entire training state
        state_dict = checkpoint['state_dict']
        if state_dict and all(key.startswith('module.') for key in state_dict):
            state_dict = {key.removeprefix('module.'): value for key, value in state_dict.items()}
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(traceback.format_exc())
            raise e
    else:
        # checkpoint only has model
        try:
            model.load_state_dict(checkpoint, strict=True)
        except RuntimeError as e:
            print(traceback.format_exc())
            raise e
    return model


@torch.no_grad()
def main(params):
    init_seed(params.seed)
    device = torch.device('cuda:0')

    if params.num_labels is None:
        try:
            params.num_labels = NUM_LABELS[params.dataset]
        except KeyError as exc:
            raise ValueError(
                "--num-labels is required for datasets not listed in NUM_LABELS"
            ) from exc
    params.max_patience = 0
    params.return_shortlist = False
    params.rw_loss = False
    params.sparse = False

    # import pdb; pdb.set_trace()
    if not os.path.exists(params.model_name):
        raise ValueError("Model path doesn't exist")

    params.data_path = os.path.join('./data/', params.dataset)
    inference_groups = InferenceDataset(params)
    params.embed_drops = [0.0] * (len(inference_groups.groups) + 1)
    # import pdb; pdb.set_trace()

    model = CascadeXML(params, inference_groups, device).to(device)
    model.eval()
    print("Loading model from ", params.model_name)
    model = load_model(model, params.model_name)

    tokenizer = get_tokenizer(params.bert)

    if not os.path.exists(params.input):
        print('Input is not a valid path, assuming string input')
        text = params.input
        text = text.replace('\n', ' ')
    else:
        with open(params.input, 'r') as f:
            text = f.read()
        text = text.replace('\n', ' ')
    
    if params.label_map and os.path.exists(params.label_map):
        with open(params.label_map, 'r') as label_map_file:
            label_map = json.load(label_map_file)
    text_tokens = torch.tensor(tokenizer.encode(text)[:params.max_len])
    attn_mask = torch.ones_like(text_tokens)

    all_probs, all_candidates, all_probs_weighted = model(text_tokens.unsqueeze(0).to(device), attn_mask.unsqueeze(0).to(device))
    if params.eval_scheme == 'level':
        all_preds = [torch.topk(probs, min(params.output_k, probs.shape[1]))[1].cpu() for probs in all_probs]
    else:
        all_preds = [torch.topk(probs, min(params.output_k, probs.shape[1]))[1].cpu() for probs in all_probs_weighted]

    all_preds = [candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds].cpu()
                for candidates, preds in zip(all_candidates, all_preds)]
    
    # Meta labels discarde for inference
    actual_labels = all_preds[-1][0]

    if params.label_map and os.path.exists(params.label_map):
        # turn label idx to text -> missing label remapping + label text querying
        print([label_map[int(i)]['title'] for i in actual_labels])
    else:
        print(actual_labels)


        







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=False, default=29)
    parser.add_argument('--mn', dest='model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=False, default='Wiki-500K')
    parser.add_argument('--num-labels', type=int)
    parser.add_argument('--bert', type=str, required=False, default='bert-base')
    parser.add_argument('--max_len', type=int, required=False, default=128)
    parser.add_argument('--topk', required=False, type=int, default=[128, 256, 512], nargs='+')
    parser.add_argument('--eval_scheme', type=str, choices=['weighted', 'level'], default='level')
    parser.add_argument('--output-k', type=int, default=10)
    #Parabel Cluster params
    parser.add_argument('--cluster_name', default='Eclusters_1865.pkl')
    parser.add_argument('--tree_depth', type=int, nargs='+', default=[10, 13, 16])
    parser.add_argument('--cluster_method', default='AugParabel')
    parser.add_argument('--verbose_lbs', type=int, default=0)

    parser.add_argument('--input', required=True, type=str, 
    help='input to run model on. If text, will run on text, if csv, will run on first column entries')
    parser.add_argument('--label_map', required=False, help='label id to value map as json')

    params = parser.parse_args()
    # import pdb; pdb.set_trace()
    main(params)
