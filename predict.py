import json
import os
import torch
import transformers
import random

import numpy as np

from transformers import BertTokenizer, BertConfig, BertModel

from sklearn import metrics

from tqdm import tqdm

from catboost import CatBoostClassifier, Pool

tagset = ['O', 'Scene', 'Nonscene']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 'T-Systems-onsite/german-roberta-sentence-transformer-v2'

class SentSegBiRnnCrf(torch.nn.Module):
    def __init__(self, bert_model, sent_bert_dims, layers, tagset, num_rnn_layers=1, sliding_window_size=9, weights=[1.0, 1.0, 1.0]):
        super(SentSegBiRnnCrf, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.context_encoder = BertModel.from_pretrained(bert_model, output_hidden_states=True)
        self.sentence_encoder = BertModel.from_pretrained(bert_model, output_hidden_states=True)
        
        self.tagset = tagset
        self.dims = sent_bert_dims
        self.num_layers = layers
        self.window_size = sliding_window_size

        self.linear = torch.nn.Linear(sent_bert_dims * (sliding_window_size + 1) * 2, 1)

        self.loss = torch.nn.HingeEmbeddingLoss()

        self.sliding_window_size = sliding_window_size

    def encode_set_test(self, path):
        ret_list = []

        for file in tqdm(os.listdir(path)):
            if file.endswith('.json'):
                json_parsed = json.load(open(os.path.join(path, file), 'r', encoding='utf8'))
                
                sentence_tags = [torch.tensor(0) for sentence in json_parsed['sentences']]
                sentence_starts = {}

                last_scene_end = -1
                for i, sentence in enumerate(json_parsed['sentences']):
                    sentence_starts[sentence['begin']] = i

                for scene in json_parsed['scenes']:
                    sentence_tags[sentence_starts[scene['begin']]] = torch.tensor(1) if scene['type'] == 'Scene' else torch.tensor(2)
                                
                sentences = ['[CLS] ' + json_parsed['text'][sent_bound['begin']:sent_bound['end']] + ' [SEP]' for sent_bound in json_parsed['sentences']]

                sentences = ['[PAD]' for l in range(self.sliding_window_size // 2)] + sentences + ['[PAD]' for l in range(self.sliding_window_size // 2)]

                sentences = [self.tokenizer(sentence, return_tensors='pt') for sentence in sentences]

                ret_list.append((sentences, sentence_tags))

        slide_ret_list = []

        count_1 = 0
        count_2 = 0
        for pos in ret_list:
            for i in range(len(pos[1])):
                if pos[1][i] == 1:
                    count_1 += 1
                if pos[1][i] == 2:
                    count_2 += 1
                slide_ret_list.append((pos[0][i:i + self.sliding_window_size], pos[1][i]))
        print(count_1)
        print(count_2)
        return ret_list, slide_ret_list

    def encode_set_train(self, path, max_samples):
        ret_list = []

        for file in tqdm(os.listdir(path)):
            if file.endswith('.json'):
                json_parsed = json.load(open(os.path.join(path, file), 'r', encoding='utf8'))
                
                sentence_tags = [torch.tensor(0) for sentence in json_parsed['sentences']]
                sentence_starts = {}

                last_scene_end = -1
                for i, sentence in enumerate(json_parsed['sentences']):
                    sentence_starts[sentence['begin']] = i

                for scene in json_parsed['scenes']:
                    sentence_tags[sentence_starts[scene['begin']]] = torch.tensor(1) if scene['type'] == 'Scene' else torch.tensor(2)
                                
                sentences = ['[CLS] ' + json_parsed['text'][sent_bound['begin']:sent_bound['end']] + ' [SEP]' for sent_bound in json_parsed['sentences']]

                sentences = ['[PAD]' for l in range(self.sliding_window_size // 2)] + sentences + ['[PAD]' for l in range(self.sliding_window_size // 2)]

                sentences = [self.tokenizer(sentence, return_tensors='pt') for sentence in sentences]

                ret_list.append((sentences, sentence_tags))

        slide_ret_list = []

        count_1 = 0
        count_2 = 0
        for pos in ret_list:
            for i in range(len(pos[1])):
                if pos[1][i] == 1:
                    count_1 += 1
                if pos[1][i] == 2:
                    count_2 += 1
                slide_ret_list.append((pos[0][i:i + self.sliding_window_size], pos[1][i]))

        kernel_list = []
        no_equal_count = 0
        equal_count = 0

        equal_scenes = 0
        equal_nonschenes = 0
        unequal_scenes = 0
        unequal_nonscenes = 0

        slide_ret_list_2 = slide_ret_list.copy()
        random.shuffle(slide_ret_list)

        steps = 0
        for elem in slide_ret_list:
            random.shuffle(slide_ret_list_2)
            for other_elem in slide_ret_list_2:
                if elem[1] == 0 and other_elem[1] == 0:
                    continue

                if elem[1] == other_elem[1] and equal_count > no_equal_count:
                    continue
                elif elem[1] != other_elem[1] and equal_count < no_equal_count:
                    continue

                if (elem[1] == 1 or other_elem[1] == 1) and elem[1] != other_elem[1]:
                    unequal_scenes += 1

                if (elem[1] == 2 or other_elem[1] == 2) and elem[1] != other_elem[1]:
                    unequal_nonscenes += 1

                kernel_list.append((elem[0], other_elem[0], torch.tensor(1) if elem[1] == other_elem[1] else torch.tensor(-1)))

                if elem[1] == 1 and elem[1] == other_elem[1]:
                    equal_scenes += 1

                if elem[1] == 2 and elem[1] == other_elem[1]:
                    equal_nonschenes += 1

                if (elem[1] == 1 or other_elem[1] == 1) and elem[1] != other_elem[1]:
                    unequal_scenes += 1

                if (elem[1] == 2 or other_elem[1] == 2) and elem[1] != other_elem[1]:
                    unequal_nonscenes += 1

                if elem[1] == other_elem[1]:
                    equal_count += 1
                elif elem[1] != other_elem[1]:
                    no_equal_count += 1

                steps += 1
                
                if steps % 100 == 0:
                    print(steps)

                if steps == max_samples:
                    break

            if steps == max_samples:
                break

        print(no_equal_count)
        print(equal_count)
        print(equal_scenes)
        print(equal_nonschenes)
        print(unequal_scenes)
        print(unequal_nonscenes)
        return ret_list, kernel_list

    def forward(self, input_sequence, batch_size=1, training_mode=False):
        embs = torch.zeros((batch_size, len(input_sequence[0][0]), self.dims)).to(device)
        core = torch.zeros((batch_size, self.dims)).to(device)

        embs_2 = torch.zeros((batch_size, len(input_sequence[0][1] if training_mode else input_sequence[0]), self.dims)).to(device)
        core_2 = torch.zeros((batch_size, self.dims)).to(device)

        for i in range(batch_size):
            for j in range(len(input_sequence[i][0])):
                inputs = input_sequence[i][0][j].to(device)
                inputs_2 = input_sequence[i][1][j].to(device)

                outputs = self.context_encoder(**inputs)
                outputs_2 = self.context_encoder(**inputs_2)

                embs[i][j] = outputs.pooler_output[0]
                if j == len(input_sequence[i][0]) // 2 + 1:
                    outputs_core = self.sentence_encoder(**inputs)
                    outputs_core_2 = self.sentence_encoder(**inputs_2)
                    core[i] = outputs_core.pooler_output[0]
                    core_2[i] = outputs_core_2.pooler_output[0]

        kernel_out_1 = torch.cat((torch.flatten(embs, start_dim=1), core), dim=1)
        kernel_out_2 = torch.cat((torch.flatten(embs_2, start_dim=1), core_2), dim=1)

        concat_out = torch.cat((kernel_out_1, kernel_out_2, torch.abs(torch.subtract(kernel_out_1, kernel_out_2))), dim=1)

        if training_mode:
            return self.linear(concat_out)

        return kernel_out_1

    def encode(self, input_sequence):
        embs = torch.zeros((1, len(input_sequence[0]), self.dims)).to(device)
        core = torch.zeros((1, self.dims)).to(device)

        for j in range(len(input_sequence[0])):
            inputs = input_sequence[0][j].to(device)

            outputs = self.context_encoder(**inputs)

            embs[0][j] = outputs.pooler_output[0]
            if j == len(input_sequence[0]) // 2 + 1:
                outputs_core = self.sentence_encoder(**inputs)
                core[0] = outputs_core.pooler_output[0]

        kernel_out_1 = torch.cat((torch.flatten(embs, start_dim=1), core), dim=1)

        return kernel_out_1

    def fit(self, optimizer, training_set, epochs=50, evaluation_set=None, batch_size=1):
        for ep in range(epochs):
            r = 0.0
            num_t = 0.0

            random.shuffle(training_set)

            batches = [training_set[i:i + batch_size] for i in range(len(training_set) // batch_size)]
            for batch in tqdm(batches):
                out = self.forward([batch[i] for i in range(len(batch))], batch_size=batch_size, training_mode=True)
                loss_out = self.loss(out, torch.tensor([batch[i][2] for i in range(len(batch))]).to(device))

                optimizer.zero_grad()
                loss_out.backward()
                optimizer.step()

                r += loss_out.item()
                num_t += 1.0

                if num_t % 20 == 0:
                    print(str(r / num_t))
            
            torch.save(optimizer.state_dict(), str(ep) + '_newarchhh_optim_final.pt')
            torch.save(self.state_dict(), str(ep) + '_newarchhh_final.pt')

    def predict_per_sentence(self, in_set):
        sent_labels = [0 for x in in_set]
        print(len(sent_labels))
        sent_labels[0] = 2

        for i, window in tqdm(enumerate(in_set)):
            prediction = self.forward(window[0])[1][0]

            for j, pred in enumerate(prediction):
                if i + j == len(sent_labels):
                    break
                if pred > 0:
                    sent_labels[i + j] = pred

        return sent_labels

def to_set(input_set, model):
    out = np.zeros((len(input_set), 768 * 10))
    out_labels = np.zeros(len(input_set))
    for i, elem in enumerate(tqdm(input_set)):
        out[i] = model.encode(elem).cpu().detach().numpy()
        out_labels[i] = elem[1].cpu().detach().numpy()
    return out, out_labels

def final_pred(model_nn, model_gb, borders_path, path, outpath, sliding_window_size):

    def get_bound(begin, end, typ):
        return {
            'begin': begin,
            'end': end,
            'type': typ
        }

    for file in tqdm(os.listdir(path)):
        print('Predicting for ' + file)
        if file.endswith('.json'):
            source_file = open(os.path.join(path, file), 'r', encoding='utf8')
            json_parsed = json.load(source_file)
            source_file.close()

            begins = [sent_bound['begin'] for sent_bound in json_parsed['sentences']]
            ends = [sent_bound['end'] for sent_bound in json_parsed['sentences']]

            sentences_init = ['[CLS] ' + json_parsed['text'][sent_bound['begin']:sent_bound['end']] + ' [SEP]' for sent_bound in json_parsed['sentences']]

            sentences = ['[PAD]' for l in range(sliding_window_size // 2)] + sentences_init + ['[PAD]' for l in range(sliding_window_size // 2)]

            sentences = [model_nn.tokenizer(sentence, return_tensors='pt') for sentence in sentences]

            slide_ret_list = []

            for i in range(len(sentences_init)):
                slide_ret_list.append([sentences[i:i + sliding_window_size]])

            vectors = np.zeros((len(slide_ret_list), 7680))

            for i in tqdm(range(len(slide_ret_list))):
                vectors[i] = model_nn.encode(slide_ret_list[i]).cpu().detach().numpy()

            pred_pool = Pool(data=vectors)
            pred_pool.quantize(input_borders=borders_path)

            predicted_labels = model_gb.predict(pred_pool)
            
            scenes = []
            for i in range(len(predicted_labels)):
                if predicted_labels[i] > 0:
                    if len(scenes) > 0:
                        scenes[len(scenes) - 1]['end'] = begins[i]

                    scenes.append(get_bound(begins[i], -1, 'Scene' if predicted_labels[i] == 1 else 'Nonscene'))
                
                if i == len(predicted_labels) - 1:
                    scenes[len(scenes) - 1]['end'] = ends[i]
            
            json_parsed['scenes'] = scenes

            target_file = open(os.path.join(outpath, file), 'w', encoding='utf8')
            json.dump(json_parsed, target_file)
            target_file.close()

if __name__ == '__main__':
    model_nn = SentSegBiRnnCrf('pretrained-model', 768, 12, tagset)
    
    model_nn.load_state_dict(torch.load('model_nn.pt', map_location=device))
    model_nn = model_nn.to(device)
    model_nn = model_nn.eval()
    weights = [1 - 54582 / 55813, 1 - 1189 / 55813, 1 - 65 / 55813]
    model_cb = CatBoostClassifier(class_weights=weights)
    model_cb.load_model('model_gb.cbm', format='cbm')

    final_pred(model_nn, model_cb, 'borders.dat', 'data/test', 'predictions', 9)




