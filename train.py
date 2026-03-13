import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from model import build_transformer
from dataset import BilingualDataset, causal_mask, postprocess_wordpiece
from config import Config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, WordPiece, BPE
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, loss_fn, val_ds, tokenizer_src, tokenizer_tgt, max_len, device, epoch, writer, num_examples=10):
    model.eval()

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80


    with torch.no_grad():
        val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
        batch_iterator = tqdm(val_dl, desc=f"Validating Epoch {epoch:02d}")
        avg_loss = 0
        n_items = 0

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            avg_loss += loss.item()
            n_items += 1
            batch_iterator.set_postfix({"avg_loss": f"{avg_loss / n_items:6.3f}"})

        writer.add_scalar('validation loss', avg_loss / n_items, epoch)
        writer.flush()

        count = 0
        source_texts = []
        expected = []
        predicted = []
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

        for batch in val_dl:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            model_out_text = postprocess_wordpiece(model_out_text)

            source_texts.append(source_text)
            expected.append([target_text])
            predicted.append(model_out_text)

            if count <= num_examples:
                # Print the source, target and model output
                print('-'*console_width)
                print(f"{f'SOURCE: ':>12}{source_text}")
                print(f"{f'TARGET: ':>12}{target_text}")
                print(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count > 200:
                break

        print('-'*console_width)

    if writer:
        if False:
            # Evaluate the character error rate
            # Compute the char error rate
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scalar('validation cer', cer, epoch)
            writer.flush()

            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scalar('validation wer', wer, epoch)
            writer.flush()

        #print(expected[:10])
        #print(predicted[:10])
        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected) * 100
        writer.add_scalar('validation BLEU', bleu, epoch)
        writer.flush()
        print(f"BLEU score: {bleu:.4f}")

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config.tokenizer_file.format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], vocab_size=25000, min_frequency=3)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    tokenizer.decoder = tokenizers.decoders.WordPiece()
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(config.datasource, name=config.datasource_name, split='train', data_files=config.data_files, token = config.hf_token)

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config.lang_tgt)

    try:
        val_ds_raw = load_dataset(config.datasource, name=config.datasource_name, split='validation', data_files=config.data_files, token = config.hf_token)
    except:
        val_ds_raw = None

    if val_ds_raw is None:
        print("No validation dataset found, splitting the training dataset into train and validation sets...")
        train_ds_size = int(0.98 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size], generator=torch.Generator().manual_seed(320))
    else:
        train_ds_raw = ds_raw

    if False:
        # Find the maximum length of each sentence in the source and target sentence
        max_len_src = 0
        max_len_tgt = 0

        n_long_sentences = 0
        n_sentences = 0
        for item in ds_raw:
            src_ids = tokenizer_src.encode(item['translation'][config.lang_src]).ids
            tgt_ids = tokenizer_tgt.encode(item['translation'][config.lang_tgt]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))
            n_sentences += 1
            if len(src_ids) > config.seq_len - 2 or len(tgt_ids) > config.seq_len - 1:
                n_long_sentences += 1

        print(f'Max length of source sentence: {max_len_src}')
        print(f'Max length of target sentence: {max_len_tgt}')
        print(f'Number of long sentences:      {n_long_sentences} of {n_sentences}, {n_long_sentences / n_sentences:.2%}')

    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config.lang_src, config.lang_tgt, config.seq_len)
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config.lang_src, config.lang_tgt, config.seq_len)

    return train_ds, val_ds, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config.seq_len, config.seq_len, d_model=config.d_model)
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(config.model_folder).mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Model has {params:,} parameters')

    # Tensorboard
    writer = SummaryWriter(config.experiment_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config.preload
    model_filename = config.latest_weights_file_path() if preload == 'latest' else config.get_weights_file_path(preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        print(f"Model preloaded successfully, starting from epoch {initial_epoch} and global step {global_step}")
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    #run_validation(model, loss_fn, val_ds, tokenizer_src, tokenizer_tgt, config.seq_len, device, initial_epoch, writer)

    for epoch in range(initial_epoch, initial_epoch + config.num_epochs):
        train_sampler = RandomSampler(
            train_ds,
            replacement = False,
            num_samples = 250000
            #num_samples = int(len(train_ds) * 0.25)
        )

        train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler)

        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        loss_smooth_alpha = 0.015
        smoothed_loss = 0

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            if global_step == 0:
                smoothed_loss = loss.item()
            smoothed_loss = loss_smooth_alpha * loss.item() + (1 - loss_smooth_alpha) * smoothed_loss
            batch_iterator.set_postfix({"loss": f"{loss:6.3f} | {smoothed_loss:6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, loss_fn, val_ds, tokenizer_src, tokenizer_tgt, config.seq_len, device, epoch, writer)

        # Save the model at the end of every epoch
        batch_iterator.write("Saving model...")

        model_filename = config.get_weights_file_path(f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = Config()
    train_model(config)
