import re
import numpy as np

import nltk
# nltk.download('punkt')

# from nltk.tokenize import sent_tokenize
# from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize


def word_tokenization(sen, start_token='<START>', end_token='<END>'):
    
    # pattern1 = r'[,!?;:]'
    pattern1 = r'[,]'                       # Replace some special characters with ''
    pattern2 = r'https?://\S+|www\.\S+'     # Replace URLs with <URL>
    pattern3 = r'\S+@\S+'                   # Replace emails with <EMAIL>
    pattern4 = r'\d+(\.\d+)?'               # Replace numbers with <NUM>
    # pattern5 = r'[^\w\s<>]'               # Replace punctuations with space

    for i in range(len(sen)):

        # sen[i] = sen[i].split('-', 1)
        sen[i] = re.sub(pattern1, '', sen[i])
        sen[i] = re.sub(pattern2, 'URL', sen[i])
        sen[i] = re.sub(pattern3, 'EMAIL', sen[i])
        sen[i] = re.sub(pattern4, 'NUM', sen[i])        
        # sen[i] = re.sub(pattern5, r' ', sen[i])

        # sen[i] = word_tokenize(sen[i])
        sen[i] = wordpunct_tokenize(sen[i])

        # sen[i] = [token.strip().lower() for token in sen[i] if (token.strip() and (token.isalnum() or (['-', '.'] in token)))]
        sen[i] = [token.strip().lower() for token in sen[i] if (token.strip() and token.isalnum())]

        sen[i] = [start_token, *sen[i], end_token]
    
    return


def preprocess_en(dataset, start_token='<START>', end_token='<END>'):
    word_tokenization(dataset, start_token=start_token, end_token=end_token)
    # return dataset

def preprocess_fr(dataset, start_token='<START>', end_token='<END>'):
    word_tokenization(dataset, start_token=start_token, end_token=end_token)
    # return dataset

def main():
    input_file = 'temp_en.txt'
    
    with open(f'./{input_file}', 'r', encoding='utf-8') as file:
        input = file.readlines()

    preprocess_en(input)
    print(f'en:')
    for sen in input:
        print(sen)
    print()

    input_file = 'temp_fr.txt'
    
    with open(f'./{input_file}', 'r', encoding='utf-8') as file:
        input = file.readlines()

    preprocess_fr(input)
    print(f'fr:')
    for sen in input:
        print(sen)
    print()


if __name__ == '__main__':
    main()
