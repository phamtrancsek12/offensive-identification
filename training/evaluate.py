""" Evaluate """
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import f1_score, classification_report

from config import TRAINING_SCHEME, DEVICE
from utils.log import get_logger, out_dir
logger = get_logger(__file__.split("/")[-1])

def evaluate(model, eval_dataset, prefix):
    """ Evaluate function """

    # Create data loader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=TRAINING_SCHEME["batch_size"])

    # Log info
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", TRAINING_SCHEME["batch_size"])

    # Inference
    preds = None
    out_label_ids = None
    eval_loss = 0.0
    nb_eval_steps = 0

    epoch_iterator = tqdm(eval_dataloader, desc="Iteration", disable=False)
    for step, batch in enumerate(epoch_iterator):
        nb_eval_steps += 1
        model.eval()
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
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
            outputs = model(**inputs)
            # Get loss and logits
            loss, logits = outputs[:2]
            eval_loss += loss.mean().item()

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['label_ids'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['label_ids'].detach().cpu().numpy(), axis=0)

    # Evaluate
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = classification_report(out_label_ids, preds)
    f1 = f1_score(out_label_ids, preds, average='macro')

    # Log result
    logger.info("***** Eval results {} *****".format(prefix))
    logger.info("Loss = {}\n".format(eval_loss))
    logger.info("Macro F1 = {}\n".format(f1))
    logger.info("\n{}".format(result))
    logger.info("\n===========================\n")

    return eval_loss, f1, result
