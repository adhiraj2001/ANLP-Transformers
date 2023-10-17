# ANLP_Assignment - 3

## Instructions

### Setting up Environment

- Install the relevent python packages: `pip install -r requirements.txt`
<!-- - Download `Spacy` pre-trained pipeline for tokenization: `python3 -m spacy install en_core_web_md` -->
- Download weights for extracting word-vectors from Fasttext: `https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz`

### Preprocess

- To Preprocess our Dataset: 
```bash
python3 -u ./utils/dataset.py \
	--dataset ./dataset \
	--seed 0 \
	--threshold 1 \

```

### Train

- Training Downstream:
```bash
    python3 -u ./Transformer/transformer_train.py \
	--input_dir ./dataset \
	--d_model 256 \
	--num_heads 4 \
	--hidden_size 1024 \
	--num_layers 2 \
	--context_size 128 \
	--dropout 0.1 \
	--batch_size 128 \
	--num_workers 0 \
	--epochs 10 \
	--learning_rate 0.01 \
	--factor 0.1 \
	--patience 3 \
	--log_step 10 \
```

## Analyze

- Pre-processed dataset is in `./dataset` folder
- Pre-trained weights are in `./weights` folder
- Report in the assignment directory `./Report.pdf`
