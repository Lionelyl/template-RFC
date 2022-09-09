import os

import torch.nn as nn
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm, trange
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
# from sklearn.preprocessing import normalize
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForMaskedLM

import data_utils_NEW as data_utils
import features
from eval_utils import evaluate, apply_heuristics


START_TAG = 7
STOP_TAG = 8

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    # 函数等价于
    # return torch.log(torch.sum(torch.exp(vec)))
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BERT_BERT_MLP(nn.Module):

    def __init__(self, bert_model_name, chunk_hidden_dim, max_chunk_len, max_seq_len, feat_sz,
                 batch_size, output_dim, use_features=False, bert_freeze=0, templates=None, device='cpu'):
        super(BERT_BERT_MLP, self).__init__()
        self.chunk_hidden_dim = chunk_hidden_dim
        self.max_chunk_len = max_chunk_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.bert_model_name = bert_model_name
        self.device = device

        bert_config = AutoConfig.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)

        print("initialing")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.chunk_bert = AutoModelForMaskedLM.from_pretrained(bert_model_name)
        # self.fc = nn.Linear(768, vocab_size)
        self.crossEntropyLoss = nn.CrossEntropyLoss()

        self.projection = nn.Linear(768+feat_sz, 768)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(output_dim + 2, output_dim + 2))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[START_TAG, :] = -10000
        self.transitions.data[:, STOP_TAG] = -10000

        self.templates = templates if templates else []

        if bert_freeze > 0:
            # We freeze here the embeddings of the model
            for param in self.bert_model.embeddings.parameters():
                param.requires_grad = False
            for layer in self.bert_model.encoder.layer[:bert_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.use_features = use_features

    def _get_bert_features(self, x, x_feats, x_len, x_chunk_len, device):
        self.bert_batch = 8
        # print("_get_bert_features()")
        #print("x", x.shape)

        input_ids = x[:,0,:,:]
        token_ids = x[:,1,:,:]
        attn_mask = x[:,2,:,:]

        max_seq_len = max(x_len)
        #print("x_len", x_len.shape, max_seq_len)
        #print("x_chunk_len", x_chunk_len.shape, x_chunk_len)

        tensor_seq = torch.zeros((len(x_len), max_seq_len, 768)).float().to(device)
        # tensor_seq = torch.zeros((len(x_len), max_seq_len, 768)).float()

        idx = 0
        for inp, tok, att, seq_length, chunk_lengths in zip(input_ids, token_ids, attn_mask, x_len, x_chunk_len):
            curr_max = max(chunk_lengths)

            inp = inp[:seq_length, :curr_max]
            tok = tok[:seq_length, :curr_max]
            att = att[:seq_length, :curr_max]
            #print("inp", inp.shape)

            # Run bert over this
            outputs = self.bert_model(inp, attention_mask=att, token_type_ids=tok,
                                      position_ids=None, head_mask=None)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            tensor_seq[idx, :seq_length] = pooled_output
            del outputs

            #print("output", pooled_output.shape)
        #print("tensor_seq.shape", tensor_seq.shape)

        ## concate features
        if self.use_features:
            x_feats = x_feats[:, :max_seq_len, :]
            tensor_seq = torch.cat((tensor_seq, x_feats), 2)

        ## projection
        tensor_seq = self.projection(tensor_seq)

        return tensor_seq

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        temp = torch.tensor([START_TAG], dtype=torch.long).to(self.device)

        tags = torch.cat([temp, tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[STOP_TAG, tags[-1]]   # score表示正确路径的得分
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.output_dim + 2), -10000.)
        init_vvars[0][START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        # forward_var = init_vvars.cuda()
        forward_var = init_vvars.to(self.device)
        #print("feats.size()", feats.size())

        for feat_idx, feat in enumerate(feats):
            #print(feat_idx)
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.output_dim + 2):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[STOP_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == START_TAG  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.output_dim + 2), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # forward_var = init_alphas.cuda()
        forward_var = init_alphas.to(self.device)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.output_dim + 2):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.output_dim + 2)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score +  emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[STOP_TAG]
        alpha = log_sum_exp(terminal_var) # 所有路径的总分 log(exp(s1) + exp(s2) + ... + exp(sn))，其中si表示路径i
        return alpha

    def predict_sequence(self, x, x_feats, x_len, x_chunk_len, device, index2id):
        # Get the emission scores from the model
        #print("x.shape", x.shape)
        feats = self._get_bert_features(x, x_feats, x_len, x_chunk_len, device)
        output = self.chunk_bert(inputs_embeds=feats)
        labels_ids = [x[0] for x in sorted(index2id.items(), key=lambda x: x[1])]

        outputs = []
        for x_len_i, sequence in zip(x_len, output.logits[:,:,labels_ids]):
            #print("sequence.shape", sequence[:x_len_i].shape)
            # Find the best path, given the features.
            # set emission score of START_TAG and END_TAG to zero
            feats = torch.cat((sequence[:x_len_i], torch.zeros(sequence.shape[0],2).to(device)), -1)
            score, tag_seq = self._viterbi_decode(feats)
            outputs.append(tag_seq)

        return outputs

    def forword_entlm(self, x, x_feats, x_len, x_chunk_len, y, device, index2id):

        loss_accum = torch.autograd.Variable(torch.FloatTensor([0])).to(device)
        n = 0
        feats = self._get_bert_features(x, x_feats, x_len, x_chunk_len, device)
        output = self.chunk_bert(inputs_embeds=feats)
        labels_ids = [x[0] for x in sorted(index2id.items(), key=lambda x: x[1])]

        for x_len_i, sent_feats, tags in zip(x_len, output.logits[:,:,labels_ids], y):
            #print(sent_feats[:x_len_i].size(), tags[:x_len_i].size())
            # set emission score of START_TAG and END_TAG to zero
            feat = torch.cat((sent_feats[:x_len_i],torch.zeros(sent_feats.shape[0],2).to(device)), -1)
            forward_score = self._forward_alg(feat)
            gold_score = self._score_sentence(feat, tags[:x_len_i])
            loss_accum += forward_score - gold_score
            n += 1
        return loss_accum / n , output.logits

    def forward(self, x, x_feats, x_len, x_chunk_len,):  # dont confuse this with _forward_alg above
        output = self._get_bert_features(x, x_feats, x_len, x_chunk_len, self.device)
        return output

def evaluate_entlm(model, test_dataloader, device, tag2index, index2id):
    label_indexs = [v for k, v in tag2index.items()]
    model.eval()
    total_loss_dev = 0
    preds = []; labels = []
    for x, x_feats, x_len, x_chunk_len, y in test_dataloader:
        x = x.to(device)
        x_feats = x_feats.to(device)
        x_len = x_len.to(device)
        x_chunk_len = x_chunk_len.to(device)
        y = y.to(device)

        model.zero_grad()
        loss, logits = model.forword_entlm(x, x_feats, x_len, x_chunk_len, y, device, index2id)

        tag = y[0][:x_len[0]]

        total_loss_dev += loss.item()
        label_logits = logits.squeeze(0).cpu()
        label_logits = label_logits[:, label_indexs]
        pred = label_logits.argmax(-1)
        # pred = [tag2index['O'] if ll[pred[i]]<0.5 else label_indexs[pred[i].item()] for i, ll in enumerate(label_logits)]
        pred = [label_indexs[p.item()] for p in pred]


        # preds += pred
        # labels += tag.cpu().data.numpy().tolist()
        preds.append(pred)
        labels.append(tag.cpu().data.numpy().tolist())

    return total_loss_dev / len(test_dataloader), labels, preds

def evaluate_bert(model, test_dataloader, device, index2id):
    model.eval()
    total_loss_dev = 0
    preds = []; labels = []
    for x, x_feats, x_len, x_chunk_len, y in test_dataloader:
        x = x.to(device)
        x_feats = x_feats.to(device)
        x_len = x_len.to(device)
        x_chunk_len = x_chunk_len.to(device)
        y = y.to(device)

        model.zero_grad()
        batch_p = model.predict_sequence(x, x_feats, x_len, x_chunk_len, device, index2id)
        batch_preds = []
        for p in batch_p:
            batch_preds += p

        batch_y = y.view(-1)
        # Focus on non-pad elemens
        idx = batch_y >= 0
        batch_y = batch_y[idx]
        label = batch_y.to('cpu').numpy()
        #print(len(batch_preds), label.shape)
        #exit()

        # Accumulate predictions
        preds += list(batch_preds)
        labels += list(label)

        # Get loss
        loss,_ = model.forword_entlm(x, x_feats, x_len, x_chunk_len, y, device, index2id)
        #print(loss.item())
        total_loss_dev += loss.item()
    return total_loss_dev / len(test_dataloader), labels, preds

def evaluate_sequences(model, test_dataloader, device, index2id):
    model.eval()
    preds = []; labels = []
    for x, x_feats, x_len, x_chunk_len, y in test_dataloader:
        x = x.to(device)
        x_feats = x_feats.to(device)
        x_len = x_len.to(device)
        x_chunk_len = x_chunk_len.to(device)
        y = y.to(device)

        model.zero_grad()
        batch_p = model.predict_sequence(x, x_feats, x_len, x_chunk_len, device, index2id)
        batch_preds = []
        for p in batch_p:
            batch_preds += p

        batch_y = y.view(-1)
        # Focus on non-pad elemens
        idx = batch_y >= 0
        batch_y = batch_y[idx]
        label = batch_y.to('cpu').numpy()

        preds.append(list(batch_preds))
        labels.append(list(label))
    return labels, preds
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs_emissions', action='store_true')
    parser.add_argument('--use_transition_priors', action='store_true')
    parser.add_argument('--protocol', type=str,  help='protocol', required=True)
    parser.add_argument('--printout', default=False, action='store_true')
    parser.add_argument('--features', default=False, action='store_true')
    parser.add_argument('--token_level', default=False, action='store_true', help='perform prediction at token level')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--word_embed_path', type=str)
    parser.add_argument('--word_embed_size', type=int, default=100)
    parser.add_argument('--token_hidden_dim', type=int, default=50)
    parser.add_argument('--chunk_hidden_dim', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--write_results', default=False, action='store_true')
    parser.add_argument('--heuristics', default=False, action='store_true')
    parser.add_argument('--bert_model', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--multi_template', default=False, action='store_true')
    parser.add_argument('--template_num', type=int, default=1)
    parser.add_argument('--template_id', type=int, default=0)
    parser.add_argument('--label_map_path', type=str)
    parser.add_argument('--handmaded_label_map_path', type=str)
    parser.add_argument('--warmup', default=False, action='store_true')
    parser.add_argument('--redundancy', default=False, action='store_true')

    # I am not sure about what this is anymore
    parser.add_argument('--partition_sentence', default=False, action='store_true')

    args = parser.parse_args()

    protocols = ["TCP", "SCTP", "PPTP", "LTP", "DCCP", "BGPv4"]
    if args.protocol not in protocols:
        print("Specify a valid protocol")
        exit(-1)

    device = 'cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu'
    args.savedir_fold = os.path.join(args.savedir, "output/checkpoint_EntLM_{}_{}_{}_{}.pt".format("handmade+data_search" if args.handmaded_label_map_path else "data_search",
                                                                                                   "warmup" if args.warmup else "",
                                                                                                   "redundancy" if args.redundancy else "",
                                                                                                   args.protocol))

    # for debug
    # args.savedir_fold = os.path.join(args.savedir, "output/checkpoint_EntLM_{}.pt".format(args.protocol))

    word2id = {}; tag2id = {}; pos2id = {}; id2cap = {}; stem2id = {}; id2word = {}
    transition_counts = {}
    # Get variable and state definitions
    def_vars = set(); def_states = set(); def_events = set(); def_events_constrained = set()
    data_utils.get_definitions(def_vars, def_states, def_events_constrained, def_events)

    together_path_list = [p for p in protocols if p != args.protocol]
    args.train = ["rfcs-bio/{}_phrases_train.txt".format(p) for p in together_path_list]
    args.test = ["rfcs-bio/{}_phrases.txt".format(args.protocol)]

    X_train_data_orig, y_train, level_h, level_d = data_utils.get_data(args.train, word2id, tag2id, pos2id, id2cap, stem2id, id2word, transition_counts, partition_sentence=args.partition_sentence)
    X_test_data_orig, y_test, level_h, level_d = data_utils.get_data(args.test, word2id, tag2id, pos2id, id2cap, stem2id, id2word, partition_sentence=args.partition_sentence)


    def_var_ids = [word2id[x.lower()] for x in def_vars if x.lower() in word2id]
    def_state_ids = [word2id[x.lower()] for x in def_states if x.lower() in word2id]
    def_event_ids = [word2id[x.lower()] for x in def_events if x.lower() in word2id]

    max_chunks, max_tokens = data_utils.max_lengths(X_train_data_orig, y_train)
    max_chunks_test, max_tokens_test = data_utils.max_lengths(X_test_data_orig, y_test)

    max_chunks = max(max_chunks, max_chunks_test)
    max_tokens = max(max_tokens, max_tokens_test)

    print(max_chunks, max_tokens)
    #exit()

    id2tag = {v: k for k, v in tag2id.items()}

    vocab_size = len(stem2id)
    pos_size = len(pos2id)
    X_train_feats = features.transform_features(X_train_data_orig, vocab_size, pos_size, def_var_ids, def_state_ids, def_event_ids, id2cap, id2word, word2id, True)
    X_test_feats = features.transform_features(X_test_data_orig, vocab_size, pos_size, def_var_ids, def_state_ids, def_event_ids, id2cap, id2word, word2id, True)
    feat_sz = X_train_feats[0].shape[1]

    # templates
    # templates_list = ["[UNK] is a [MASK]",
    #                   "The type of [UNK] is [MASK]",
    #                   "[UNK] belongs to [MASK] category",
    #                   "[UNK] should be tagged as [MASK]"
    #                   ]
    # if args.multi_template:
    #     templates = templates_list[:args.template_num]
    # else:
    #     templates = [templates_list[args.template_id]]
    # for t in templates:
    #     print(t)
    # Create model
    model = BERT_BERT_MLP(args.bert_model,
                          args.chunk_hidden_dim,
                          max_chunks, max_tokens, feat_sz, args.batch_size, output_dim=len(tag2id),
                          use_features=args.features, bert_freeze=10, device=device)
    model.to(device)

    ## label to template words
    # {'B-TRIGGER': 0, 'B-ACTION': 1, 'O': 2, 'B-TRANSITION': 3, 'B-TIMER': 4, 'B-ERROR': 5, 'B-VARIABLE': 6}
    labelwords = ['trigger', 'action', 'other', 'transition', 'timer', 'error', 'variable']
    labelwords = [ 'B-'+lw.upper() for lw in labelwords]

    tag2index = {}

    tokenizer = model.tokenizer
    data_utils.add_label_token_bert(model, tokenizer, args.protocol, args.label_map_path, args.handmaded_label_map_path, tag2index, args.redundancy)
    print(tag2index)
    index2id = { v: tag2id[k] for k, v in tag2index.items()}
    print(index2id)

    X_train_data, y_train, x_len, x_chunk_len = \
        data_utils.bert_sequences(X_train_data_orig, y_train, max_chunks, max_tokens, id2word, tokenizer, tag2index, id2tag, is_y_with_tag_index=False)
    X_test_data, y_test, x_len_test, x_chunk_len_test = \
        data_utils.bert_sequences(X_test_data_orig, y_test, max_chunks, max_tokens, id2word, tokenizer, tag2index, id2tag, is_y_with_tag_index=False)

    X_train_feats = data_utils.pad_features(X_train_feats, max_chunks)
    X_test_feats = data_utils.pad_features(X_test_feats, max_chunks)

    # Subsample a development set (10% of the data)
    n_dev = int(X_train_data.shape[0] * 0.1)
    dev_excerpt = random.sample(range(0, X_train_data.shape[0]), n_dev)
    train_excerpt = [i for i in range(0, X_train_data.shape[0]) if i not in dev_excerpt]

    X_dev_data = X_train_data[dev_excerpt]
    y_dev = y_train[dev_excerpt]
    x_len_dev = x_len[dev_excerpt]
    X_dev_feats = X_train_feats[dev_excerpt]
    x_chunk_len_dev = x_chunk_len[dev_excerpt]

    X_train_data = X_train_data[train_excerpt]
    y_train = y_train[train_excerpt]
    x_len = x_len[train_excerpt]
    X_train_feats = X_train_feats[train_excerpt]
    x_chunk_len = x_chunk_len[train_excerpt]

    print(X_train_data.shape, X_train_feats.shape, y_train.shape, x_len.shape, x_chunk_len.shape)
    print(X_dev_data.shape, X_dev_feats.shape, y_dev.shape, x_len_dev.shape, x_chunk_len_dev.shape)

    #print(x_chunk_len)
    #exit()

    print(y_train.shape)

    y_train_ints = list(map(int, y_train.flatten()))
    y_train_ints = [y for y in y_train_ints if y >= 0]

    classes = list(set(y_train_ints))
    print(classes, tag2id)

    train_dataset = data_utils.ChunkDataset(X_train_data, X_train_feats, x_len, x_chunk_len, y_train)
    dev_dataset = data_utils.ChunkDataset(X_dev_data, X_dev_feats, x_len_dev, x_chunk_len_dev, y_dev)
    test_dataset = data_utils.ChunkDataset(X_test_data, X_test_feats, x_len_test, x_chunk_len_test, y_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    if args.do_train:
        n_epoch = 100
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2e-5,
            div_factor=2,
            final_div_factor=10,
            epochs=n_epoch,
            steps_per_epoch=1,
            pct_start=0.1)

        # Training loop
        patience = 0; best_f1 = 0; epoch = 0; best_loss = 10000
        training_statics = ""


        print("TRAINING!!!!")
        while epoch < n_epoch:
            pbar = tqdm(total=len(train_dataloader))
            model.train()
            total_loss = 0

            print(f'lr: {optimizer.param_groups[0]["lr"]}')

            for x, x_feats, x_len, x_chunk_len, y in train_dataloader:
                x = x.to(device)
                x_feats = x_feats.to(device)
                x_len = x_len.to(device)
                x_chunk_len = x_chunk_len.to(device)
                y = y.to(device)
                # print(f'chunk_size : {x_len}')

                model.zero_grad()

                loss, _ = model.forword_entlm(x, x_feats, x_len, x_chunk_len, y, device, index2id)
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                pbar.update(1)

            if args.warmup and scheduler:
                scheduler.step()

            dev_loss, dev_labels, dev_preds = evaluate_bert(model, dev_dataloader, device, index2id)
            test_loss, test_labels, test_preds = evaluate_bert(model, test_dataloader, device, index2id)
            macro_f1 = f1_score(dev_labels, dev_preds, average='macro')
            test_macro_f1 = f1_score(test_labels, test_preds, average='macro')
            if macro_f1 > best_f1:
                #if dev_loss < best_loss:
                # Save model
                #print("Saving model...")
                torch.save(model.state_dict(), args.savedir_fold)
                best_f1 = macro_f1
                best_loss = dev_loss
                patience = 0
            else:
                patience += 1

            training_str = "\nepoch {} loss {} dev_loss {} dev_macro_f1 {} test_macro_f1 {}".format(
                epoch,
                total_loss / len(train_dataloader),
                dev_loss,
                macro_f1,
                test_macro_f1)
            print(training_str)
            training_statics += training_str

            epoch += 1
            if patience >= args.patience:
                break

    if args.do_eval:
        # Load model
        model.load_state_dict(torch.load(args.savedir_fold, map_location=lambda storage, loc: storage))

        # _, y_test, y_pred = evaluate_entlm(model, test_dataloader, device, tag2index, index2id)
        y_test, y_pred = evaluate_sequences(model, test_dataloader, device, index2id)

        # y_test_trans = data_utils.translate_entlm(y_test, tag2index)
        # y_pred_trans = data_utils.translate_entlm(y_pred, tag2index)
        y_test_trans = data_utils.translate(y_test, id2tag)
        y_pred_trans = data_utils.translate(y_pred, id2tag)

        # Do it in a way that preserves the original chunk-level segmentation
        _, y_test_trans_alt, _, _ = data_utils.alternative_expand(X_test_data_orig, y_test_trans, level_h, level_d, id2word, debug=False)
        X_test_data_alt, y_pred_trans_alt, level_h_alt, level_d_alt = data_utils.alternative_expand(X_test_data_orig, y_pred_trans, level_h, level_d, id2word, debug=True)

        # Do it in a way that flattens the chunk-level segmentation for evaluation
        X_test_data_old = X_test_data_orig[:]
        _, y_test_trans_eval = data_utils.expand(X_test_data_orig, y_test_trans, id2word, debug=False)
        X_test_data_eval, y_pred_trans_eval = data_utils.expand(X_test_data_orig, y_pred_trans, id2word, debug=True)


        evaluate(y_test_trans_eval, y_pred_trans_eval)

        def_states_protocol = {}; def_events_protocol = {}; def_events_constrained_protocol = {}; def_variables_protocol = {}
        data_utils.get_protocol_definitions(args.protocol, def_states_protocol, def_events_constrained_protocol, def_events_protocol, def_variables_protocol)

        y_pred_trans_alt = \
            apply_heuristics(X_test_data_alt, y_test_trans_alt, y_pred_trans_alt,
                             level_h_alt, level_d_alt,
                             id2word, def_states_protocol, def_events_protocol, def_variables_protocol,
                             transitions=args.heuristics, outside=args.heuristics, actions=args.heuristics,
                             consecutive_trans=True)

        X_test_data_orig, y_pred_trans, level_h_trans, level_d_trans = \
            data_utils.alternative_join(
                X_test_data_alt, y_pred_trans_alt,
                level_h_alt, level_d_alt,
                id2word, debug=True)

        if args.heuristics:
            _, y_test_trans_eval = data_utils.expand(X_test_data_old, y_test_trans, id2word, debug=False)
            evaluate(y_test_trans_eval, y_pred_trans)


        if args.write_results:
            output_xml = os.path.join(args.outdir, "{}.xml".format(args.protocol))
            results = data_utils.write_results(X_test_data_orig, y_test_trans, y_pred_trans, level_h_trans, level_d_trans,
                                               id2word, def_states_protocol, def_events_protocol, def_events_constrained_protocol,
                                               args.protocol, cuda_device=args.cuda_device)
            with open(output_xml, "w") as fp:
                fp.write(results)

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(4321)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(4321)
    random.seed(4321)

    main()
