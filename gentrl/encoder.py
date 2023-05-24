import torch
from torch import nn
from gentrl.tokenizer import encode
from transformers import AutoTokenizer, AutoModelWithLMHead

d_model = 767  # embedding size - 1
bs = 50  # batch_size 50
pad_size = 101

class RNNEncoder(nn.Module):
    def __init__(self, vocab, hidden_size=256, num_layers=2, latent_size=50,
                 bidirectional=False):
        super(RNNEncoder, self).__init__()

        self.vocab = vocab
        self.embs = nn.Embedding(len(vocab), hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, 2 * latent_size))

    def encode(self, sm_list):
        """
        Maps smiles onto a latent space
        """

        tokens, lens = encode(sm_list, self.vocab)
        to_feed = tokens.transpose(1, 0).to(self.embs.weight.device)

        outputs = self.rnn(self.embs(to_feed))[0]
        outputs = outputs[lens, torch.arange(len(lens))]

        return self.final_mlp(outputs)


class ChemBERTaEncoder(nn.Module):
    def __init__(self, vocab, latent_size=50):
        super(ChemBERTaEncoder, self).__init__()

        self.vocab = vocab
        self.embs = nn.Embedding(len(vocab), d_model)
        self.chemberta = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

        self.final_mlp = nn.Sequential(
            nn.Linear(pad_size * d_model, d_model), nn.LeakyReLU(),
            nn.Linear(d_model, 2 * latent_size))

    def encode(self, sm_list):
        """
        Maps smiles onto a latent space
        """

        tokens, lens = encode(sm_list, self.vocab)  # batch_size * sec_len
        to_feed = tokens.to(self.embs.weight.device)
        enc_outputs = self.chemberta.roberta(to_feed).last_hidden_state  # [batch_size, src_len, d_model + 1]
        enc_outputs = self.chemberta.lm_head(enc_outputs)  # [batch_size, src_len, d_model]

        return self.final_mlp(enc_outputs.reshape(bs, pad_size * d_model))
