import sys, os
import argparse
import torch
import time
import numpy as np
from utils import *

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sklearn.externals.joblib import Memory
# from persistent_lru_cache import persistent_lru_cache

import data


cachedir = os.getenv('CACHE_DIR', 'tmp')
print(f'Cache directory is: {cachedir}')
memory = Memory(cachedir=cachedir, verbose=0)

app = Flask(__name__)
cors = CORS(app, resources={r"/demo/": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')
# Model parameters.
parser.add_argument('--data', type=str, default='chardict.pt',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./charlm.pt',
                    help='model checkpoint to use')
parser.add_argument('--maxlen', type=int, default=128,
                    help='max length of generated sequence')
parser.add_argument('--beamsize', type=int, default=16,
                    help='location of the data corpus')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')

args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    if args.cuda:
        model = torch.load(f).to(device)
    else:
        model = torch.load(f, map_location='cpu')
model.eval()

corpus = data.Corpus(args.data)
ntokens = 5000


# @persistent_lru_cache(filename='fib_imp.db', maxsize=None)
@memory.cache
def getSuggestion(raw_text):
    possible_prompts = [['UNK'] + list(raw_text)]
    encoded_prompts = [corpus.encode_wordlist(pp) for pp in possible_prompts]
    input_indices = torch.stack(encoded_prompts).t().to(device)
    suggestions = beam_forward(input_indices, model, args).squeeze().t()
    suggested_list = []
    for i in range(5):
        suggested_list.append(decode_suggestion(possible_prompts[0][1:], suggestions[:,i], corpus))
    return suggested_list


@app.route("/demo", methods=["POST", "GET"])
@cross_origin(origin='*',headers=['Content-Type'])
def demo():
    raw_text = request.get_json()['prompt']
    raw_text = raw_text.lower()
    print("raw_text", raw_text)
    start_time = time.time()
    print("start_time", start_time)
    suggested_list = getSuggestion(raw_text)
    suggested_list = rerank4diversity(suggested_list)
    print('suggested_list', suggested_list)
    print("time:", time.time() - start_time)
    return jsonify(suggested_list)

if __name__ == "__main__":
	app.run(host= '0.0.0.0', port='4001',debug=False, use_reloader=False)
