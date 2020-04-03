""" Train """

from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from evaluate import evaluate
from utils.utils import set_seed
from utils.log import get_logger, out_dir
from config import TRAINING_SCHEME, DEVICE
logger = get_logger(__file__.split("/")[-1])


def train(train_dataset, valid_dataset, model):
    """ Train the model """

    # Create data loader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=TRAINING_SCHEME["batch_size"])

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': TRAINING_SCHEME["weight_decay"]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=TRAINING_SCHEME["learning_rate"], eps=TRAINING_SCHEME["adam_eps"])
    t_total = int(len(train_dataloader) * TRAINING_SCHEME["num_training_epoches"])
    warmup_step = int(t_total * TRAINING_SCHEME["warmup_proportion"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.to(DEVICE)
    model.zero_grad()
    set_seed()

    best_f1 = 0
    train_iterator = trange(int(TRAINING_SCHEME["num_training_epoches"]), desc="Epoch", disable=False)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            global_step += 1
            model.train()
            batch = tuple(t.to(DEVICE) for t in batch)

            # Convert input format of BERT model
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3],
                      'label_ids': batch[4]}
            # Init
            model.batch_size = len(batch[0])
            model.lstm_hidden = model.init_hidden()
            # Fit data to model
            ouputs = model(**inputs)
            # Get loss and do backward
            loss = ouputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_SCHEME["max_grad_norm"])
            tr_loss += loss.item()
            # Update optimizer and scheduler
            optimizer.step()
            scheduler.step()
            # Zero grad
            model.zero_grad()

        # Evaluate and save best model
        valid_loss, valid_f1, _ = evaluate(model, valid_dataset, "valid")
        if (valid_f1 > best_f1):
            best_f1 = valid_f1
            model.save_pretrained(out_dir)
            logger.info("======> SAVE BEST MODEL | F1 = {}".format(valid_f1))

    return model, global_step, tr_loss / global_step