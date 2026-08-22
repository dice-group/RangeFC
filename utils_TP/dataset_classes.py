from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch

# -------------MY EDITITNG-------------------------

from collections import defaultdict
def make_pairs(split_idx_list):
    """
    group by subject and return (h,p,o,t1,t2) pairs
    whenever subject appears at least twice
    """
    buckets = defaultdict(list)
    for record in split_idx_list:
        h,p,o,t, *rest = record
        buckets[(h)].append((p,o,t))

    pairs = []
    for h, items in buckets.items():
        if len(items) >= 2:
            """
            you can build all pairs, or min/max, or consecutive
            Here: pick min/max t
            """
            ts = [item[2] for item in items]
            t1 = min(ts)
            t2 = max(ts)
            #You can choose one representative (p,o)
            p,o, _ = items[0]
            pairs.append([h,p,o,t1,t2])
    return pairs

    #print(f"Generated {len(self.paired_train_idx)} train pairs")
    #print(f"Generated {len(self.paired_valid_idx)} valid pairs")
    #print(f"Generated {len(self.paired_test_idx)} test pairs")

# -------------MY EDITITNG-------------------------

class RangePredictionDataset(Dataset):
    """
    Each item is (h, r, t, year1_idx, year2_idx)
    """

    def __init__(self, triples_idx, num_entities, num_relations, num_times, neg_sample_ratio=0):
        # Handle empty split safely
        if not triples_idx:
            empty = torch.zeros(0, dtype=torch.long)
            self.head_idx = empty
            self.rel_idx = empty
            self.tail_idx = empty
            self.y1_idx = empty
            self.y2_idx = empty
            self.length = 0
            self.num_entities = num_entities
            self.num_relations = num_relations
            self.num_times = num_times
            return

        triples = torch.as_tensor(triples_idx, dtype=torch.long)
        if triples.ndim != 2 or triples.size(1) != 5:
            raise ValueError(f"RangePredictionDataset expects Nx5, got shape {tuple(triples.shape)}")

        self.head_idx = triples[:, 0]
        self.rel_idx = triples[:, 1]
        self.tail_idx = triples[:, 2]
        self.y1_idx = triples[:, 3]
        self.y2_idx = triples[:, 4]
        self.length = triples.size(0)

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_times = num_times

    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return (
            self.head_idx[idx],
            self.rel_idx[idx],
            self.tail_idx[idx],
            self.y1_idx[idx],
            self.y2_idx[idx]
        )

class StandardDataModule(pl.LightningDataModule):
    """
    train, valid and test sets are available.
    """

    def __init__(self, train_set_idx, entities_count, relations_count, times_count, batch_size, form,
                 num_workers=32, valid_set_idx=None, test_set_idx=None, neg_sample_ratio=None,
                 paired_train_idx=None, paired_valid_idx=None, paired_test_idx=None):
        super().__init__()
        self.train_set_idx = train_set_idx
        self.valid_set_idx = valid_set_idx
        self.test_set_idx = test_set_idx

        self.num_entities = entities_count
        self.num_relations = relations_count
        self.num_times = times_count

        self.form = form
        # self.batch_size = batch_size
        self.num_workers = num_workers
        self.neg_sample_ratio = neg_sample_ratio
        if self.form == 'FactChecking':  # we can name it as FactChecking
            self.dataset_type_class = FactCheckingDataset
            self.target_dim = 1
            self.neg_sample_ratio = neg_sample_ratio
        elif self.form == 'TimePrediction':  # we can name it as FactChecking
            self.dataset_type_class = TimePredictionDataset
            self.target_dim = 1
            self.neg_sample_ratio = neg_sample_ratio
        elif self.form == 'RangePrediction':
            # We handle RangePrediction with our custom RangePredictionDataset in the loaders
            self.dataset_type_class = RangePredictionDataset
            self.target_dim = 2
        # ————— Build paired lists from the flat idx lists —————
            '''self.paired_train_idx = make_pairs(self.train_set_idx)
            self.paired_valid_idx = make_pairs(self.valid_set_idx or [])
            self.paired_test_idx = make_pairs(self.test_set_idx or [])'''
        else:
            raise ValueError

    # Train, Valid, TestDATALOADERs
    def train_dataloader(self, batch_size1) -> DataLoader:
        if self.form == 'FactChecking':
            self.batch_size = batch_size1
            print("Loading Training Data...")       # my editing for checking where my code gets killed?
            train_set = FactCheckingDataset(self.train_set_idx,
                                            num_entities=self.num_entities,
                                            num_relations=self.num_relations,
                                            num_times=self.num_times)
            #print(f"Training Set Size: {len(train_set)}")        # my editing for checking where my code gets killed?
            return DataLoader(train_set, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)
        elif self.form == 'TimePrediction':
            self.batch_size = batch_size1
            train_set = TimePredictionDataset(self.train_set_idx,
                                            num_entities=self.num_entities,
                                            num_relations=self.num_relations,
                                            num_times=self.num_times)
            return DataLoader(train_set, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)
        elif self.form == 'RangePrediction':
            # uses the paired (h,p,o,t1,t2) list we built in Data.__init__
            #from utils_TP.dataset_classes import RangePredictionDataset
            #self.batch_size = batch_size1
            #train_set = RangePredictionDataset(self.paired_train_idx)
            ds = RangePredictionDataset(self.train_set_idx, num_entities=self.num_entities, num_relations=self.num_relations, num_times=self.num_times)
            if len(ds) == 0:
                print("⚠️  Train split is empty after mapping — check ID normalization & maps.")
                # return a minimal loader to avoid crashing early; training will effectively be skipped
                return DataLoader(ds, batch_size=1, shuffle=False, num_workers=self.num_workers)
            return DataLoader(ds, batch_size=batch_size1, shuffle=True,num_workers=self.num_workers)

    def val_dataloader(self, batch_size1) -> DataLoader:

        if self.form == 'FactChecking':
            self.batch_size = batch_size1
            print("Loading Validation Data...")         # my editing for checking where my code gets killed?
            val_set = FactCheckingDataset(self.valid_set_idx,
                                            num_entities=self.num_entities,
                                            num_relations=self.num_relations,
                                            num_times=self.num_times)
            #print(f"Validation Set Size: {len(val_set)}")       # my editing for checking where my code gets killed?
            return DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        elif self.form == 'TimePrediction':
            self.batch_size = batch_size1
            val_set = TimePredictionDataset(self.valid_set_idx,
                                            num_entities=self.num_entities,
                                            num_relations=self.num_relations,
                                            num_times=self.num_times)
            return DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        elif self.form == 'RangePrediction':
            # uses the paired (h,p,o,t1,t2) list we built in Data.__init__
            ds = RangePredictionDataset(self.valid_set_idx or [], num_entities=self.num_entities, num_relations=self.num_relations, num_times=self.num_times)
            if len(ds) == 0:
                print("⚠️  Train split is empty after mapping — check ID normalization & maps.")
                # return a minimal loader to avoid crashing early; training will effectively be skipped
                return DataLoader(ds, batch_size=1, shuffle=False, num_workers=self.num_workers)
            return DataLoader(ds, batch_size=batch_size1, shuffle=False,num_workers=self.num_workers)

    def dataloaders(self, batch_size1) -> DataLoader:
        if self.form == 'FactChecking':
            test_set = FactCheckingDataset(self.test_set_idx,
                                               num_entities=(self.num_entities),
                                               num_relations=(self.num_relations),
                                               num_times=(self.num_times))
            return DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        elif self.form == 'TimePrediction':
            test_set = TimePredictionDataset(self.test_set_idx,
                                               num_entities=(self.num_entities),
                                               num_relations=(self.num_relations),
                                               num_times=(self.num_times))
            return DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        elif self.form == 'RangePrediction':
            # uses the paired (h,p,o,t1,t2) list we built in Data.__init__
            ds = RangePredictionDataset(self.test_set_idx, num_entities=self.num_entities, num_relations=self.num_relations, num_times=self.num_times)
            return DataLoader(ds, batch_size=batch_size1, shuffle=False,num_workers=self.num_workers)

    def setup(self, *args, **kwargs):
        pass

    def transfer_batch_to_device(self, *args, **kwargs):
        pass

    def prepare_data(self, *args, **kwargs):
        # Nothing to be prepared for now.
        pass


class TimePredictionDataset(Dataset):
    """
    Similar Issue =
    https://github.com/pytorch/pytorch/issues/50089
    https://github.com/PyTorchLightning/pytorch-lightning/issues/538
    """
    def __init__(self, triples_idx, num_entities, num_relations, num_times, neg_sample_ratio=0):
        self.neg_sample_ratio = neg_sample_ratio  # 0 Implies that we do not add negative samples. This is needed during testing and validation
        triples = torch.LongTensor(triples_idx)
        self.head_idx = triples[:, 0]
        self.rel_idx = triples[:, 1]
        self.tail_idx = triples[:, 2]
        self.y1_idx = triples[:, 3]
        self.y2_idx = triples[:, 4]

        # enforce y1 <= y2 for training targets
        swap = self.y1_idx > self.y2_idx
        if swap.any():
            y1c = self.y1_idx.clone()
            self.y1_idx[swap] = self.y2_idx[swap]
            self.y2_idx[swap] = y1c[swap]

        self.length = len(triples)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_times = num_times

        #--------------------my editing--------------

        #if there are no examples, build seven empty tensors and bail out -
        if not triples_idx:
            empty = torch.zeros(0, dtype=torch.long)
            self.head_idx = empty
            self.rel_idx = empty
            self.tail_idx = empty
            self.time_idx = empty
            self.sent_idx = empty
            self.score_idx = empty
            self.lbl_idx = empty
            self.length = 0
            self.num_entities = num_entities
            self.num_relations = num_relations
            self.num_times = num_times
            return

        #turn you list-of-lists into a tensor
        triples_tensor = torch.LongTensor(triples_idx)

        #at this point we know its >= 2-D
        n_cols = triples_tensor.size(1)

        if n_cols == 5:
            # [head, rel, tail, time, label]
            self.head_idx = triples_tensor[:, 0]
            self.rel_idx = triples_tensor[:, 1]
            self.tail_idx = triples_tensor[:, 2]
            self.time_idx = triples_tensor[:, 3]
            #you dont have sentence - or score-coloms, so fill with dummies
            N = self.head_idx.size(0)
            self.sent_idx = torch.zeros(N, dtype=torch.long)
            self.score_idx = torch.zeros(N, dtype=torch.long)
            self.lbl_idx = triples_tensor[:, 4]

        elif n_cols == 7:
            #original format: [h, r, t, time, sent_i, score_i, lbl]
            self.head_idx = triples_tensor[:, 0]
            self.rel_idx = triples_tensor[:, 1]
            self.tail_idx = triples_tensor[:, 2]
            self.time_idx = triples_tensor[:, 3]
            self.sent_idx = triples_tensor[:, 4]
            self.score_idx = triples_tensor[:, 5]
            self.lbl_idx = triples_tensor[:, 6]

        else:
            raise ValueError(f"Expected 5 or 7 columns in your split files, got {n_cols}")

        #now these all share the same length
        #assert (self.head_idx.size(0) == self.rel_idx.size(0) == self.tail_idx.size(0) == self.time_idx.size(0) == self.lbl_idx.size(0))

        # assert self.sent_idx == self.head_idx.shape == self.rel_idx.shape == self.tail_idx.shape == self.lbl_idx.shape == self.score_idx.shape == self.time_idx.shape
        #assert self.head_idx.shape == self.rel_idx.shape == self.tail_idx.shape == self.lbl_idx.shape  == self.time_idx.shape
        #self.length = len(triples_tensor)

        #self.num_entities = num_entities
        #self.num_relations = num_relations
        #self.num_times = num_times

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            self.head_idx[idx],
            self.rel_idx[idx],
            self.tail_idx[idx],
            self.y1_idx[idx],
            self.y2_idx[idx],
        )

class FactCheckingDataset(Dataset):
    """
    Similar Issue =
    https://github.com/pytorch/pytorch/issues/50089
    https://github.com/PyTorchLightning/pytorch-lightning/issues/538
    """
    def __init__(self, triples_idx, num_entities, num_relations, num_times, neg_sample_ratio=0):
        triples = torch.LongTensor(triples_idx)
        self.head_idx = triples[:, 0]
        self.rel_idx = triples[:, 1]
        self.tail_idx = triples[:, 2]
        self.y1_idx = triples[:, 3]
        self.y2_idx = triples[:, 4]
        self.length = len(triples)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_times = num_times

        #assert (self.sent_idx.shape == self.head_idx.shape == self.rel_idx.shape == self.tail_idx.shape ==self.lbl_idx.shape == self.score_idx.shape == self.time_idx.shape)
        #self.length = len(triples_idx)

        #self.num_entities = num_entities
        #self.num_relations = num_relations
        #self.num_times = num_times

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            self.head_idx[idx],
            self.rel_idx[idx],
            self.tail_idx[idx],
            self.y1_idx[idx],
            self.y2_idx[idx]
        )

    class FactCheckingDataset(Dataset):
        """
        Similar Issue =
        https://github.com/pytorch/pytorch/issues/50089
        https://github.com/PyTorchLightning/pytorch-lightning/issues/538
        """

        def __init__(self, triples_idx, num_entities, num_relations, num_times, neg_sample_ratio=0):
            triples = torch.LongTensor(triples_idx)
            self.head_idx = triples[:, 0]
            self.rel_idx = triples[:, 1]
            self.tail_idx = triples[:, 2]
            self.y1_idx = triples[:, 3]
            self.y2_idx = triples[:, 4]
            self.length = len(triples)
            self.num_entities = num_entities
            self.num_relations = num_relations
            self.num_times = num_times

            #assert self.head_idx.shape == self.rel_idx.shape == self.tail_idx.shape == self.lbl_idx.shape == self.score_idx.shape == self.time_idx.shape
            #self.length = len(triples_idx)

            #self.num_entities = num_entities
            #self.num_relations = num_relations
            #self.num_times = num_times

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return (
                self.head_idx[idx],
                self.rel_idx[idx],
                self.tail_idx[idx],
                self.y1_idx[idx],
                self.y2_idx[idx]
            )



    # def collate_fn(self, batch):
    #     batch = torch.LongTensor(batch)
    #     h, r, t, time, veracity, label = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
    #     # h, r, t, label = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]
    #     size_of_batch, _ = batch.shape
    #     assert size_of_batch > 0
    #     label = torch.ones((size_of_batch,))
    #     # # Generate Negative Triples
    #     corr = torch.randint(0, self.num_entities, (size_of_batch * self.neg_sample_ratio, 2))
    #     #
    #     # # 2.1 Head Corrupt:
    #     h_head_corr = corr[:, 0]
    #     r_head_corr = r.repeat(self.neg_sample_ratio, )
    #     t_head_corr = t.repeat(self.neg_sample_ratio, )
    #     label_head_corr = torch.zeros(len(t_head_corr), )
    #
    #     # 2.2. Tail Corrupt
    #     h_tail_corr = h.repeat(self.neg_sample_ratio, )
    #     r_tail_corr = r.repeat(self.neg_sample_ratio, )
    #     t_tail_corr = corr[:, 1]
    #     label_tail_corr = torch.zeros(len(t_tail_corr), )
    #     #
    #     # # 3. Stack True and Corrupted Triples
    #     # h = torch.cat((h, h_head_corr, h_tail_corr), 0)
    #     # r = torch.cat((r, r_head_corr, r_tail_corr), 0)
    #     # t = torch.cat((t, t_head_corr, t_tail_corr), 0)
    #     label = torch.cat((label, label_head_corr, label_tail_corr), 0)
    #
    #     return (h, r, t), label
