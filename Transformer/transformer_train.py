import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import wandb
# wandb.login(key='913841cb22c908099db4951c258f4242c1d1b7aa')

import os
os.environ['WANDB_API_KEY'] = '913841cb22c908099db4951c258f4242c1d1b7aa'
os.environ['WANDB_SILENT'] = 'true'

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler

from torch.nn.utils.rnn import pad_sequence

from transformer import Transformer

## To avoid Cuda out of Memory Error (if doesn't work, try reducing batch size)
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## Fixing Seed (https://arxiv.org/abs/2109.08203)
# torch.manual_seed(3407)


def load_dataset(folder='./dataset'):

    with open(f'{folder}/train.pkl', 'rb') as file:
        train = np.load(file, allow_pickle=True) 

    with open(f'{folder}/test.pkl', 'rb') as file:
        test = np.load(file, allow_pickle=True) 

    with open(f'{folder}/val.pkl', 'rb') as file:
        valid = np.load(file, allow_pickle=True) 

    with open(f'{folder}/train_word_embeddings.pkl', 'rb') as file:
        word_embeddings = np.load(file, allow_pickle=True)

    with open(f'{folder}/vocab_map.json', 'r') as file:
        word_map = json.load(file)

    return train, test, valid, word_embeddings, word_map.values() 


# NOTE: Padding to make the sequence length same
def create_sequence(X_arr, y_arr, word_map_en, word_map_fr, pad_flag=False, end_token='<END>', pad_token='<PAD>', cutoff_len=None):
    
    for i in range(len(X_arr)):
        if cutoff_len:
            X_arr[i] = torch.tensor(X_arr[i][:cutoff_len], dtype=torch.long)
            y_arr[i] = torch.tensor(y_arr[i][:cutoff_len], dtype=torch.long)
        else:
            X_arr[i] = torch.tensor(X_arr[i], dtype=torch.long)
            y_arr[i] = torch.tensor(y_arr[i], dtype=torch.long)

    if pad_flag:
        X_arr = pad_sequence(X_arr, batch_first=True, padding_value=word_map_en[pad_token])
        y_arr = pad_sequence(y_arr, batch_first=True, padding_value=word_map_fr[pad_token])

    return X_arr, y_arr


def validate(model, valid_dl, loss_func):
    
    val_loss = 0
    val_acc = 0

    model.eval()

    # with torch.inference_mode(): # Had trouble with tensor clone (NaN outputs)
    with torch.no_grad():
        for idx, (X, y) in enumerate(valid_dl):

            X, y = X.to(device), y.to(device)

            outputs = model(X, y[..., :-1])

            ## For the cross-entropy function to work we flatten them
            # y = y.view(-1)
            y_re = y[..., 1:].contiguous().view(-1)
            outputs_re = outputs.contiguous().view(-1, outputs.size(-1))
            
            loss = loss_func(outputs_re, y_re)

            val_loss += loss.item()
            # val_loss += loss.item() * y_re.size(0)

            _, y_pred = torch.max(outputs_re, 1)
            acc = (y_pred == y_re).sum().item()

            val_acc += acc / y_re.size(0)
            # val_acc += acc

        val_loss /= len(valid_dl)
        # val_loss /= len(valid_dl.dataset)
        val_acc /= len(valid_dl)
        # val_acc /= len(valid_dl.dataset)

    return val_loss, val_acc


from nltk.translate.bleu_score import sentence_bleu

def log_metrics(model, X_data, y_data, index_map_x, index_map_y, file_path):

    with open(file_path, 'w') as f:

        avg_loss = 0
        avg_bleu = 0

        model.eval()

        # with torch.inference_mode(): # Had trouble with tensor clone (NaN outputs)
        with torch.no_grad():
            for idx in range(len(X_data)):

                X, y = X_data[idx].unsqueeze(0).to(device), y_data[idx].unsqueeze(0).to(device)

                outputs = model(X, y[..., :-1])

                ## For the cross-entropy function to work we flatten them
                # y = y.view(-1)
                y_re = y[..., 1:].contiguous().view(-1)
                outputs_re = outputs.contiguous().view(-1, outputs.size(-1))

                loss = F.cross_entropy(outputs_re, y_re).item()
                # loss = F.cross_entropy(outputs_re, y_re).item() * y_re.size(0)

                _, y_pred = torch.max(outputs_re, 1)

                perplexity = (2 ** loss)
                avg_loss += loss

                src_text = X.squeeze().detach().cpu().numpy()
                src_text = ' '.join([index_map_x[token] for token in src_text if 'pad' not in index_map_x[token].lower()])
                
                trg_text = y.squeeze().detach().cpu().numpy()
                trg_text = ' '.join([index_map_y[token] for token in trg_text if 'pad' not in index_map_y[token].lower()])

                pred_text = y_pred.detach().cpu().numpy()
                pred_text = ' '.join([index_map_y[token] for token in pred_text if 'pad' not in index_map_y[token].lower()])

                bleu_score = sentence_bleu(trg_text, pred_text)
                avg_bleu += bleu_score

                f.write(f'Sentence: \"{src_text}\",\t Translation: \"{pred_text}\",\t Blue_Score: {bleu_score:.2f}\t, Perplexity: {perplexity:.2f}\n')


            avg_loss /= len(X_data)
            avg_perplexity = 2 ** avg_loss
            avg_bleu /= len(X_data)

            f.write(f'\nAvg. Bleu Score: {avg_bleu:.2f}\n')
            f.write(f'\nAvg. Perplexity Score: {avg_perplexity:.2f}\n')
            f.write(f'\n')

        f.close()

    return


def main(args):

    (X_train, y_train), (X_test, y_test), (X_valid, y_valid), (word_embeddings_en, word_embeddings_fr), (word_map_en, word_map_fr) = load_dataset(folder=args.input_dir)

    index_map_en = {int(i):str(w) for w, i in sorted(word_map_en.items())} 
    word_embeddings_en = torch.tensor(word_embeddings_en, dtype=torch.float, device=device)

    index_map_fr = {int(i):str(w) for w, i in sorted(word_map_fr.items())} 
    word_embeddings_fr = torch.tensor(word_embeddings_fr, dtype=torch.float, device=device)

    CONTEXT_SIZE = args.context_size
    
    # NOTE: Creating and Padding Tensors 
    X_train, y_train = create_sequence(X_train, y_train, word_map_en, word_map_fr, pad_flag=True, cutoff_len=CONTEXT_SIZE)
    X_test, y_test = create_sequence(X_test, y_test, word_map_en, word_map_fr, pad_flag=True, cutoff_len=CONTEXT_SIZE)
    X_valid, y_valid = create_sequence(X_valid, y_valid, word_map_en, word_map_fr, pad_flag=True, cutoff_len=CONTEXT_SIZE)
    
    ## Tensor Dataset
    train_tensor = TensorDataset(X_train, y_train) 
    test_tensor = TensorDataset(X_test, y_test) 
    valid_tensor = TensorDataset(X_valid, y_valid) 
    
    ## Data Loaders
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    
    ## Add num_workers accordingly ?
    if torch.cuda.device_count() > 0 and NUM_WORKERS == 0:
        NUM_WORKERS = int(torch.cuda.device_count()) * 4

    ## Loading Data
    train_dl = DataLoader(train_tensor, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_tensor, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    valid_dl = DataLoader(valid_tensor, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    ## Initializing Model
    VOCAB_SIZE_1 = len(word_embeddings_en)
    VOCAB_SIZE_2 = len(word_embeddings_fr)
    
    if args.d_model:
        EMBED_SIZE_1 = args.d_model 
        EMBED_SIZE_2 = args.d_model
    else:
        EMBED_SIZE_1 = len(word_embeddings_en[0])
        EMBED_SIZE_2 = len(word_embeddings_fr[0])

    NUM_HEADS = args.num_heads
    NUM_LAYERS = args.num_layers
    HIDDEN_SIZE = args.hidden_size
    DROPOUT = args.dropout
    EPS = args.eps

    # checkpoint = torch.load(f'./weights/transformer-best.pt')

    model = Transformer(d_model=EMBED_SIZE_1, n_head=NUM_HEADS, n_layers=NUM_LAYERS, ffn_hidden=HIDDEN_SIZE, context_size=CONTEXT_SIZE, src_vocab_size=VOCAB_SIZE_1, trg_vocab_size=VOCAB_SIZE_2, src_pad=word_map_en['<PAD>'], trg_pad=word_map_fr['<PAD>'], dropout=DROPOUT, eps=EPS)

    # model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.device_count() > 1:
        # dim = 0 [40, xxx] -> [10, ...], [10, ...], [10, ...], [10, ...] on 4 GPUs
        print("Using", torch.cuda.device_count(), "GPUs.", flush=True)

        model = nn.DataParallel(model)
        
        # NOTE: This might be required for RNN based models (when explicity initializing hidden state)
        # model = nn.DataParallel(model, dim=1)

    # Important
    model = model.to(device)

    ## Hyperparameters
    EPOCHS = args.epochs 
    LR = args.learning_rate
    # GAMMA = args.gamma
    # STEP_SIZE = args.step_size
    FACTOR = args.factor
    PATIENCE = args.patience
    
    # NOTE: Ignore pad index
    criterion = nn.CrossEntropyLoss(ignore_index=word_map_en['<PAD>'])

    optimizer = optim.Adam(model.parameters(), lr=LR)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=FACTOR, patience=PATIENCE)
    
    LOG_STEP = args.log_step

    wandb.init(
        project="ANLP-Assignment-3",
        name="Transformer", 
        config={
            "architecture": "Transformer",
            "dataset": "IWSLT-2016",
            "batch-size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            # "gamma": GAMMA,
            # "step_size": STEP_SIZE
            "factor": FACTOR,
            "patience": PATIENCE,
            "log_step": LOG_STEP
        }
    )

    curr_step = 0
    best_valid_loss = float('inf')
    best_valid_acc = 0
    best_epoch = 0

    # for epoch in tqdm(range(EPOCHS)):
    for epoch in range(EPOCHS):

        model.train()
        
        train_loss = 0
        train_acc = 0
        
        for idx, (X, y) in enumerate(train_dl):

            ## NOTE: Important
            optimizer.zero_grad()

            X, y = X.to(device), y.to(device)

            outputs = model(X, y[..., :-1])

            ## For the cross-entropy function to work we flatten them
            # y = y.view(-1)
            y_re = y[..., 1:].contiguous().view(-1)
            outputs_re = outputs.contiguous().view(-1, outputs.size(-1))
            
            loss = criterion(outputs_re, y_re)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            
            train_loss += loss.item()
            # train_loss += loss.item() * y_re.size(0)

            _, y_pred = torch.max(outputs_re, 1)
            acc = (y_pred == y_re).sum().item()

            train_acc += acc / y_re.size(0)
            # train_acc += acc

            if ((curr_step + 1) % LOG_STEP == 0):
                step_metrics = {
                    # "step/lr": scheduler.get_lr(), # ReduceLROnPlateau doesn't have get_lr() for some fucking reason
                    # "step/lr": scheduler.get_last_lr(), # more reliable than get_lr() ?
                    "step/lr": scheduler.optimizer.param_groups[0]['lr'], 

                    "step/loss": loss.item(),
                    "step/acc": acc / y_re.size(0),

                    "step/perplexity": 2 ** loss.item(),

                    "step/steps": curr_step,
                    "step/epochs": epoch+1,
                }

                wandb.log(step_metrics)

            curr_step += 1

        
        train_loss /= len(train_dl)
        # train_loss /= len(train_dl.dataset)

        train_acc /= len(train_dl)
        # train_acc /= len(train_dl.dataset)

        val_loss, val_acc = validate(model, valid_dl, criterion)

        ## NOTE: Important
        scheduler.step(val_loss)


        ## Saving Best model
        if best_valid_loss >= val_loss:
            best_valid_loss = val_loss
            best_valid_acc = val_acc
            best_epoch = epoch+1
            
            torch.save({
                'epoch': epoch+1,
                # 'model_state_dict': model.state_dict(),
                'model_state_dict': model.module.state_dict() if model.module else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,

            }, './weights/transformer-best.pt')


        print(f"\nepoch: {epoch:03d} | Train Loss: {train_loss:.3f}, Train Accuracy: {train_acc:.2f}, Valid Loss: {val_loss:3f}, Valid Accuracy: {val_acc:.2f}", flush=True)

        train_metrics = {
            # "epoch/lr": scheduler.get_lr(),
            # "epoch/lr": scheduler.get_last_lr(),
            "epoch/lr": scheduler.optimizer.param_groups[0]['lr'],
            
            "epoch/avg_train_loss": train_loss,
            "epoch/avg_train_acc": train_acc,

            "epoch/train_perplexity": 2 ** train_loss,

            "epoch/epochs": epoch,
        }

        val_metrics = {
            "epoch/avg_val_loss": val_loss,
            "epoch/avg_val_acc": val_acc,
            "epoch/val_perplexity": 2 ** val_loss,
        }
        
        wandb.log({
            **train_metrics,
            **val_metrics
        })

    
    print()
    print(f"Best Validation --> Epoch: {best_epoch}, Loss: {best_valid_loss:.3f}, Accuracy: {best_valid_acc:.2f}, Perplexity {2 ** best_valid_loss:.2f}\n", flush=True)
    print()
    
    
    test_loss, test_acc = validate(model, test_dl, criterion)

    wandb.summary['test_loss'] = test_loss
    wandb.summary['test_accuracy'] = test_acc
    wandb.summary['test_perplexity'] = 2 ** test_loss
    wandb.finish()
    
    # print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.2f}\n", flush=True)
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.2f}, Test Perplexity: {2 ** test_loss:.2f}\n", flush=True)
    print()

    
    ## Saving Last model
    torch.save({
        'epoch': EPOCHS,
        # 'model_state_dict': model.state_dict(),
        'model_state_dict': model.module.state_dict() if model.module else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss,

    }, './weights/transformer-last.pt')
    

    # ----------------------------------------------------------

    log_metrics(model, X_train, y_train, index_map_en, index_map_fr, file_path='./results/transformer-train.txt')
    log_metrics(model, X_test, y_test, index_map_en, index_map_fr, file_path='./results/transformer-test.txt')

    return


if __name__ == '__main__':

    # print(f'package: {__package__}')
    # print()


    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./dataset',
                        help='input directory dataset.')
    
    parser.add_argument('--d_model', type=int, default=512,
                        help='No. of features in encoder/decoder input')

    parser.add_argument('--num_heads', type=int, default=8,
                        help='No. of transformer heads')

    parser.add_argument('--hidden_size', type=int, default=2048,
                        help='The number of features in the hidden state.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of encoder/decoder layers.')
    
    parser.add_argument('--context_size', type=int, default=256,
                        help='Max input / output length.')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability to be used in the Transformer architecture.')
    
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='Epsilone value for Layer Normalization.')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch Size')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of Workers between which batch size is divided parallely ?')

    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of Training Epochs')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning Rate')

    # parser.add_argument('--gamma', type=float, default=0.9,
    #                     help='Gamma Value')
    #
    # parser.add_argument('--step_size', type=int, default=3,
    #                     help='Step Size')

    parser.add_argument('--factor', type=float, default=0.1,
                        help='Gamma Value')

    parser.add_argument('--patience', type=int, default=3,
                        help='Step Size')

    parser.add_argument('--log_step', type=int, default=10,
                        help='Logging at every batch step')

    args = parser.parse_args()


    main(args)

