import copy
import torch

from torch.utils.data import Dataset

import pandas as pd
import numpy as np


class MolecularDataset(Dataset):
    def __init__(self, sources=[], props=['logIC50', 'BFL', 'pipeline'],
                 with_missings=False):
        self.num_sources = len(sources)

        self.source_smiles = []
        self.source_props = []
        self.source_missings = []
        self.source_probs = []

        self.with_missings = with_missings

        self.len = 0
        for source_descr in sources:
            cur_df = pd.read_csv(source_descr['path'])
            cur_smiles = list(cur_df[source_descr['smiles']].values)

            cur_props = torch.zeros(len(cur_smiles), len(props)).float()
            cur_missings = torch.zeros(len(cur_smiles), len(props)).long()

            for i, prop in enumerate(props):
                if prop in source_descr:
                    if isinstance(source_descr[prop], str):
                        cur_props[:, i] = torch.from_numpy(
                            cur_df[source_descr[prop]].values)
                    else:
                        cur_props[:, i] = torch.from_numpy(
                            cur_df[source_descr['smiles']].map(
                                source_descr[prop]).values)
                else:
                    cur_missings[:, i] = 1

            self.source_smiles.append(cur_smiles)
            self.source_props.append(cur_props)
            self.source_missings.append(cur_missings)
            self.source_probs.append(source_descr['prob'])

            self.len = max(self.len,
                           int(len(cur_smiles) / source_descr['prob']))

        self.source_probs = np.array(self.source_probs).astype(np.float)

        self.source_probs /= self.source_probs.sum()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        trial = np.random.random()

        s = 0
        for i in range(self.num_sources):
            if (trial >= s) and (trial <= s + self.source_probs[i]):
                bin_len = len(self.source_smiles[i])
                sm = self.source_smiles[i][idx % bin_len]

                props = self.source_props[i][idx % bin_len]
                miss = self.source_missings[i][idx % bin_len]

                if self.with_missings:
                    return sm, torch.concat([props, miss])
                else:
                    return sm, props

            s += self.source_probs[i]


class MlmDataset(Dataset):
    """
    注意：由于没有使用data_collator，batch放在dataset里边做，
    因而在dataloader出来的结果会多套一层batch维度，传入模型时注意squeeze掉
    """

    def __init__(self, input_texts, tokenizer, config):
        self.input_texts = input_texts
        self.tokenizer = tokenizer
        self.config = config
        self.ori_inputs = copy.deepcopy(input_texts)

    def __len__(self):
        return len(self.input_texts) // self.config.batch_size

    def __getitem__(self, idx):
        batch_text = self.input_texts[: self.config.batch_size]
        features = self.tokenizer(batch_text, max_length=512, truncation=True, padding=True, return_tensors='pt')
        inputs, labels = self.mask_tokens(features['input_ids'])
        batch = {"inputs": inputs, "labels": labels}
        self.input_texts = self.input_texts[self.config.batch_size:]
        if not len(self):
            self.input_texts = self.ori_inputs

        return batch

    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.config.mlm_probability)
        if self.config.special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = self.config.special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, self.config.prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        current_prob = self.config.prob_replace_rand / (1 - self.config.prob_replace_mask)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
