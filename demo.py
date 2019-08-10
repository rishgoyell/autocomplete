import argparse
import torch
import data

from utils import *

parser = argparse.ArgumentParser(description='Autocomplete Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='data/chardict.pt',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='models/subject_title.pt',
                    help='model checkpoint to use')
parser.add_argument('--maxlen', type=int, default=128,
                    help='max length of generated sequence')
parser.add_argument('--beamsize', type=int, default=16,
                    help='beam length while decoding')
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
    model = torch.load(f, map_location='cpu').to(device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = 5000


def interactive():
    try:
        while True:
            raw_text = input("Enter Partial Title:\n")
            raw_text = raw_text.lower()
            possible_prompts = [['UNK'] + list(raw_text)]
            encoded_prompts = [corpus.encode_wordlist(pp) for pp in possible_prompts]
            input_indices = torch.stack(encoded_prompts).to(device).t()
            suggestions = beam_forward(input_indices, model, args).squeeze().t()
            suggestions = [decode_suggestion(possible_prompts[0][1:], suggestions[:,i], corpus) for i in range(8)]
            suggestions = rerank4diversity(suggestions)
            for i in range(5):
                print(suggestions[i])
            print("\n")
    except KeyboardInterrupt:
        print("Exiting ........")


if __name__ == '__main__':
    interactive()
