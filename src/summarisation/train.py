from accelerate import Accelerator

from datasets import load_dataset

from rouge_score import rouge_scorer

import torch
from torch import nn, optim, utils

from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_scheduler
from transformers import DataCollatorForSeq2Seq

import nltk

import numpy as np
import pandas as pd

import argparse
import os
import random
import time

def get_model(name):
    if name == 't5':
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
    elif name == 'bart':
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    elif name == 'pegasus':
        model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    else:
        raise ValueError(f'Unknown model {name}. Must be t5, bart, or pegasus.')

    return (model, tokenizer)

def preprocess_text(examples, tokenizer):
    model_inputs = tokenizer(examples['content'], max_length=args.article_max_len, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['title'], max_length=args.headline_max_len, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def run(args):
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Get model and tokenizer
    print('Loading model...')
    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = get_model(args.model)
    model = model.to(device)
    elapsed = time.time() - start
    print('Done in {:.2f}s.'.format(elapsed))

    # Setup datasets
    print('Setting up datasets...')
    start = time.time()
    data_files = {split: os.path.join(os.getcwd(), 'data', '{}.csv'.format(split)) for split in ['train', 'val', 'test']}
    print('\tLoading files...', end=' ')
    newsheadlines_dataset = load_dataset('csv', data_files=data_files)
    print('Done.')
    print('\tPreprocessing texts...', end=' ')
    newsheadlines_dataset = newsheadlines_dataset.map(lambda x: preprocess_text(x, tokenizer), batched=True, keep_in_memory=True)
    print('Done.')
    print('\tSetting format...', end=' ')
    newsheadlines_dataset.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])
    print('Done.')
    elapsed = time.time() - start
    print('Done in {:.2f}s.'.format(elapsed))

    # Setup dataloaders
    collate_fn = DataCollatorForSeq2Seq(tokenizer, model=model)
    train_dataloader = torch.utils.data.DataLoader(newsheadlines_dataset['train'], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = torch.utils.data.DataLoader(newsheadlines_dataset[args.split], batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Setup Accelerator
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

    # Setup LR Scheduler
    num_training_steps = args.epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Setup rouge
    scorer = rouge_scorer.RougeScorer(['rougeL', 'rouge1', 'rouge2'], use_stemmer=True)
    #rouge_score = load_metric("rouge")

    # Training Loop
    print('Training...')
    results = []
    start = time.time()

    for epoch in range(args.epochs):
        # Train for an epoch
        model.train()
        average_loss = 0
        average_batch_time = 0
        for step, batch in enumerate(train_dataloader):
            batch_start = time.time()
            outputs = model(**batch)
            loss = outputs.loss
            average_loss += loss.item()
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            average_batch_time += time.time() - batch_start

        average_loss /= len(train_dataloader)
        average_batch_time /= len(train_dataloader)

        # Evaluate for an epoch
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"], attention_mask=batch["attention_mask"])

                generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
                labels = batch["labels"]

                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                rougeL_vals = []
                rouge1_vals = []
                rouge2_vals = []
                for pred, label in zip(decoded_preds, decoded_labels):
                    scores = scorer.score(pred, label)
                    rougeL_vals.append(scores['rougeL'].fmeasure)
                    rouge1_vals.append(scores['rouge1'].fmeasure)
                    rouge2_vals.append(scores['rouge2'].fmeasure)

                median_rougeL = np.median(np.array(rougeL_vals))
                median_rouge1 = np.median(np.array(rouge1_vals))
                median_rouge2 = np.median(np.array(rouge2_vals))
                #rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

        # Compute metrics
        # Extract the median ROUGE scores
        result = {}
        result['rougeL'] = median_rougeL
        result['rouge1'] = median_rouge1
        result['rouge2'] = median_rouge2
        result['loss'] = average_loss
        result['epoch'] = epoch
        result['batch_time'] = average_batch_time
        print(f"Epoch {epoch}:", result)
        results.append(result)

    elapsed = time.time() - start
    print('Done in {:.2f}s.'.format(elapsed))
    df = pd.DataFrame(results)
    df.to_csv(args.results_name + '.csv')
    tf = pd.DataFrame([vars(args)])
    tf.to_csv(args.results_name + '.params')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), args.results_name + '.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a summarization model.')
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=1)
    parser.add_argument('--model', type=str, help='base model', choices=['bart', 'pegasus', 't5'])
    parser.add_argument('--batch_size', type=int, help='batch size (per gpu)', default=8)
    parser.add_argument('--learning_rate', type=float, help='learning_rate', default=5e-6)
    parser.add_argument('--article_max_len', type=int, help='article maximum length', default=16)
    parser.add_argument('--headline_max_len', type=int, help='headline maximum length', default=16)
    parser.add_argument('--split', type=str, help='dataset split to evaluate on', choices=['val', 'test'])
    parser.add_argument('--seed', type=int, help='random seed for reproducibility', default=0)
    parser.add_argument('--results_name', type=str, help='name of results CSV file', default='foo')

    args = parser.parse_args()

    run(args)
