"""
Load and cache data for BERT model
Example from https://github.com/huggingface/transformers/
"""
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from utils.utils import truncate_seq_pair
from config import DATA_DIR, FILENAME, LABEL_LIST, LABEL_SIZE, TRAINING_SCHEME
from utils.log import get_logger
logger = get_logger(__file__.split("/")[-1])


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.label_id = label_id


class OffensEvalProcessor(object):
    """Processor for the racism data set."""

    def get_examples(self, prefix):
        """Gets the list of labels for this data set."""
        return self._create_examples(
            self._read_csv(DATA_DIR, FILENAME[prefix]), prefix)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _read_csv(cls, data_dir, filename):
        """ Read data from list of CSV files """
        list_df = []
        for file in filename:
            list_df.append(pd.read_csv(os.path.join(data_dir, file), lineterminator='\n'))
        df = pd.concat(list_df)
        df = df.reset_index()
        del df["index"]
        lines = []
        for i in range(len(df)):
            if "label" in df:
                lines.append([df["text"][i], df["label"][i]])
            else:
                lines.append([df["text"][i], None])
        return lines


def convert_examples_to_features(examples, tokenizer,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (1 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(LABEL_LIST)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            truncate_seq_pair(tokens_a, tokens_b, TRAINING_SCHEME["max_seq_length"] - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > TRAINING_SCHEME["max_seq_length"] - 2:
                tokens_a = tokens_a[:(TRAINING_SCHEME["max_seq_length"] - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = TRAINING_SCHEME["max_seq_length"] - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == TRAINING_SCHEME["max_seq_length"]
        assert len(input_mask) == TRAINING_SCHEME["max_seq_length"]
        assert len(segment_ids) == TRAINING_SCHEME["max_seq_length"]

        label = example.label
        if LABEL_SIZE == 2:
            label_id = label_map[int(example.label > 0.5)] # Using threshold 0.5
        else:
            label_id = example.label

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.guid))
            logger.info("tokens: {}".format(" ".join([str(x) for x in tokens])))
            logger.info("input_ids: {}".format(" ".join([str(x) for x in input_ids])))
            logger.info("input_mask: {}".format(" ".join([str(x) for x in input_mask])))
            logger.info("segment_ids: {}".format(" ".join([str(x) for x in segment_ids])))
            logger.info("label: {} (id = {})".format(label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label=label,
                          label_id=label_id))
    return features


def load_and_cache_examples(tokenizer, prefix):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(DATA_DIR, 'cached_{}_{}_{}_{}'.format(
        FILENAME[prefix][0].split(".")[0], TRAINING_SCHEME["classifier_type"], TRAINING_SCHEME["pretrained_version"], TRAINING_SCHEME["max_seq_length"]))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", DATA_DIR)
        processor = OffensEvalProcessor()
        examples = processor.get_examples(prefix)

        features = convert_examples_to_features(examples, tokenizer,
                                                cls_token_at_end=False,
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=1,
                                                pad_on_left=False,
                                                pad_token_segment_id=0) # For BERT model only

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels, all_label_ids)
    return dataset
