import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
import json
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

ImageFile.LOAD_TRUNCATED_IMAGES = True
def _truncate_seq_pair(tokens_a, tokens_b, max_length):

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples,max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    bert_data = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label
        bert_data.append([example.guid,input_ids, input_mask,segment_ids,label_id])
    return bert_data
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

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
    def _read_json(self,file_path):
        """Reads a tab separated value file."""
        res = []
        with open(file_path, "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                res.append(line)
        return res

class Processor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self,data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(data_dir))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(data_dir))

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        vid_examples = []
        cid_examples = []
        for (i, line) in enumerate(lines):
            label = line[0]
            text_a = line[2]
            #text_b = line[2]+str(':')+line[6]
            text_b = line[5]
            vid_examples.append(
                InputExample(guid=line[1], text_a=text_a, text_b=None, label=label))
            cid_examples.append(
                InputExample(guid=line[3], text_a=text_b, text_b=None, label=label))
        return vid_examples,cid_examples


class Bert_Res_Data(Dataset):
    def __init__(self, opt, tp="train"):
        assert tp in ["train", "valid", "test"]
        self.opt = opt
        if tp == "train":
            self.data_file = opt.train_file
        elif tp == "test":
            self.data_file = opt.test_file
        elif tp == "valid":
            self.data_file = opt.val_file
           
                
        self.transform = transforms.Compose(
              [
                   transforms.Scale(opt.input_size),
                   transforms. CenterCrop(opt.input_size),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize(mean = [.5,.5,.5],std = [.5,.5,.5])
              ]
           )

        processor = Processor()
        self.vid_examples,self.cid_examples = processor.get_train_examples(data_dir=self.data_file)
        self.tokenizer = BertTokenizer.from_pretrained(opt.bert_model, do_lower_case=opt.do_lower_case)
        self.vid_data = convert_examples_to_features(
                    self.vid_examples, opt.max_seq_length, self.tokenizer)
        self.cid_data = convert_examples_to_features(
                    self.cid_examples, opt.max_seq_length, self.tokenizer)
        self.length = len(self.vid_data)
        self.vid_pic_list = self.get_images(opt.vid_pic_path)
        self.cid_pic_list = self.get_images(opt.cid_pic_path)
    def get_images(self,root):
        files = []
        exts = ['jpg', 'png', 'jpeg', 'JPG']
        for parent, dirnames, filenames in os.walk(root):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(filename[:-4])
                        break
                              # print('Find {} images'.format(len(files)))
        return files
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        vid_data = self.vid_data[index]
        cid_data = self.cid_data[index]
        
        vid_guid,vid_input_id, vid_input_mask,vid_segment_id,vid_label_id = vid_data
        cid_guid,cid_input_id, cid_input_mask,cid_segment_id,cid_label_id = cid_data
        
        vid_input_id = torch.tensor(vid_input_id, dtype=torch.long)
        vid_segment_id = torch.tensor(vid_segment_id, dtype=torch.long)
        vid_input_mask = torch.tensor(vid_input_mask, dtype=torch.long)
        
        label_id = torch.tensor(int(vid_label_id), dtype=torch.long)
        
        cid_input_id = torch.tensor(cid_input_id, dtype=torch.long)
        cid_segment_id = torch.tensor(cid_segment_id, dtype=torch.long)
        cid_input_mask = torch.tensor(cid_input_mask, dtype=torch.long)
        
        vid = vid_guid
        cid = cid_guid
        vid_pic = 'vid_pic'
        cid_pic = 'cid_pic'
        if vid in self.vid_pic_list:
            vid_pic = Image.open(os.path.join(self.opt.vid_pic_path, "%s.jpg" % vid)).convert("RGB")
        else:
            vid_pic = Image.open('/dockerdata/qinguoliu/el2/data/pic/1.jpg').convert("RGB")
        if cid in self.cid_pic_list:
            cid_pic = Image.open(os.path.join(self.opt.cid_pic_path, "%s.jpg" % cid)).convert("RGB")
        else:
            cid_pic = Image.open('/dockerdata/qinguoliu/el2/data/pic/1.jpg').convert("RGB")

        return vid_guid,vid_input_id,vid_segment_id,vid_input_mask,cid_guid,cid_input_id, cid_segment_id,cid_input_mask,label_id,self.transform(vid_pic),self.transform(cid_pic)


