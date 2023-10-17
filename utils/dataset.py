import argparse
import pickle
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import fasttext

# from gensim.models import FastText
# from gensim.models import KeyedVectors

from preprocess import preprocess_en
from preprocess import preprocess_fr


def load_dataset(dataset_folder, set_name='train'):

    ## NOTE: 150/256 Sentence length seems like a good cut-off
    ## You can also try masking the rest of <PAD> tokens
    
    with open(f'{dataset_folder}/{set_name}.en', 'r', encoding='utf-8') as file:
        en = file.readlines()
    
    # en_len = [len(line) for line in en]
    # print(f'{set_name}_en:\n{pd.DataFrame(en_len).describe()}\n')
    # print()

    with open(f'{dataset_folder}/{set_name}.fr', 'r', encoding='utf-8') as file:
        fr = file.readlines()

    # fr_len = [len(line) for line in fr]
    # print(f'{set_name}_fr:\n{pd.DataFrame(fr_len).describe()}\n')
    # print()
    
    return en, fr


def training_vocabulary(train):
    # Using Train Vocabulary

    train_vocab = set()
    train_count = defaultdict(int)
    for sen in train:
        for token in sen:
            train_vocab.add(token)
            train_count[token] += 1
    
    return train_vocab, train_count


def add_unk_1(train, train_vocab, train_count, unk_token='<UNK>', threshold=1):
    for token, count in train_count.items():
        if count <= threshold:
            train_vocab.remove(token)
    
    # train = [[token if (token in train_vocab) else unk_token for token in x] for x in train]

    for sen in train:
        for i in range(len(sen)):
            if sen[i] not in train_vocab:
                sen[i] = unk_token

    # return train, train_vocab


def add_unk_2(data, vocab_map, unk_token='<UNK>'):
    # data = [[vocab_map[token] if (token in vocab_map) else vocab_map[unk_token] for token in x] for x in data]
    
    for sen in data:
        for i in range(len(sen)):
            if sen[i] in vocab_map:
                sen[i] = vocab_map[sen[i]]
            else:
                sen[i] = vocab_map[unk_token]

    # return data


def encode_en(vocab_map):
    model = fasttext.load_model('./weights/cc.en.300.bin')
    # vector_map = {i:model.get_word_vector(token) for token, i in vocab_map.items()}
    vector_map = {int(i):model.get_word_vector(token).tolist() for token, i in vocab_map.items()}
    weights = [vector for i, vector in sorted(vector_map.items())]
    return vector_map, weights

def encode_fr(vocab_map):
    model = fasttext.load_model('./weights/cc.fr.300.bin')
    vector_map = {int(i):model.get_word_vector(token).tolist() for token, i in vocab_map.items()}
    weights = [vector for i, vector in sorted(vector_map.items())]
    return vector_map, weights


def main():

    parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets.")

    parser.add_argument('--dataset', type=str, default='./dataset', help='Path to the Dataset Folder')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for Reproducibility.')
    parser.add_argument('--threshold', type=int, default=1, help='Remove Stuff from train_vocabulary')
    parser.add_argument('--start_token', type=str, default='<START>', help='Start of Sentence Token')
    parser.add_argument('--end_token', type=str, default='<END>', help='End of Sentence Token')
    parser.add_argument('--pad_token', type=str, default='<PAD>', help='Padding Token')
    parser.add_argument('--unknown_token', type=str, default='<UNK>', help='Token to represent out-of-vocab tokens')

    args = parser.parse_args()

    dataset_folder = args.dataset
    seed = args.seed
    threshold = args.threshold

    start_token = args.start_token
    end_token = args.end_token
    pad_token = args.pad_token
    unk_token = args.unknown_token
    
    # Load and process the dataset here
    train_en, train_fr = load_dataset(dataset_folder, 'train')
    val_en, val_fr = load_dataset(dataset_folder, 'dev')
    test_en, test_fr = load_dataset(dataset_folder, 'test')

    preprocess_en(train_en, start_token=start_token, end_token=end_token)
    preprocess_en(val_en, start_token=start_token, end_token=end_token)
    preprocess_en(test_en, start_token=start_token, end_token=end_token)
    
    preprocess_fr(train_fr, start_token=start_token, end_token=end_token)
    preprocess_fr(val_fr, start_token=start_token, end_token=end_token)
    preprocess_fr(test_fr, start_token=start_token, end_token=end_token)

    # -------------

    train_vocab_en, train_count_en = training_vocabulary(train_en)
    add_unk_1(train_en, train_vocab_en, train_count_en, threshold=threshold, unk_token=unk_token)

    train_vocab_en.add(pad_token)
    train_vocab_en.add(unk_token)
    
    train_map_en = {str(token):int(i) for i, token in enumerate(sorted(train_vocab_en))}
    # word_map = {i:token for i, token in train_map.items()}
    
    add_unk_2(train_en, train_map_en, unk_token=unk_token)
    add_unk_2(test_en, train_map_en, unk_token=unk_token)
    add_unk_2(val_en, train_map_en, unk_token=unk_token)

    vocab_vectors_en, vector_weights_en = encode_en(train_map_en)
    
    # ---------------
    
    train_vocab_fr, train_count_fr = training_vocabulary(train_fr)
    add_unk_1(train_fr, train_vocab_fr, train_count_fr, threshold=threshold, unk_token=unk_token)

    train_vocab_fr.add(pad_token)
    train_vocab_fr.add(unk_token)
    
    train_map_fr = {str(token):int(i) for i, token in enumerate(sorted(train_vocab_fr))}
    # word_map = {i:token for i, token in train_map.items()}
    
    add_unk_2(train_fr, train_map_fr, unk_token=unk_token)
    add_unk_2(test_fr, train_map_fr, unk_token=unk_token)
    add_unk_2(val_fr, train_map_fr, unk_token=unk_token)

    vocab_vectors_fr, vector_weights_fr = encode_fr(train_map_fr)
    
    # ---------------

    # Saving these
    # np.save('./dataset/train.npy', train)
    # np.save('./dataset/test.npy', test)
    # np.save('./dataset/val.npy', val)
    # np.save('./dataset/vocab_vectors.npy', vocab_vectors)

    with open('./dataset/train.pkl', 'wb') as file:
        pickle.dump([train_en, train_fr], file)

    with open('./dataset/test.pkl', 'wb') as file:
        pickle.dump([test_en, test_fr], file)

    with open('./dataset/val.pkl', 'wb') as file:
        pickle.dump([val_en, val_fr], file)

    with open('./dataset/train_word_embeddings.pkl', 'wb') as file:
        pickle.dump([vector_weights_en, vector_weights_fr], file)

    with open('./dataset/vocab_map.json', 'w') as file:
        json.dump({0: train_map_en, 1: train_map_fr}, file, sort_keys=True, indent=4)

    with open('./dataset/vocab_vectors.json', 'w') as file:
        json.dump({0: vocab_vectors_en, 1: vocab_vectors_fr}, file, sort_keys=True, indent=4)
    
if __name__ == "__main__":
    main()
