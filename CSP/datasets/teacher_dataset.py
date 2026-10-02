from itertools import product

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop

BICUBIC = InterpolationMode.BICUBIC
n_px = 224

def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


class CompositionDataset(Dataset):
    @staticmethod
    def import_dataset_paths(dataset=None):

        if dataset == 'cgqa':
            from datasets import cgqa_read_datasets
            return cgqa_read_datasets

        elif dataset == 'mit-states':
            from datasets import mit_read_datasets
            return mit_read_datasets

        elif dataset == 'utzappos':
            from datasets import utzappos_read_datasets
            return utzappos_read_datasets

        else:
            raise ValueError(f"Unknown dataset: {dataset}")
            
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False,
            session= "",
            teacher_session= "",
            dataset_name = None
            
            # inductive=True
    ):
        self.paths = CompositionDataset.import_dataset_paths(dataset_name)        
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world
        self.session = session
        self.teacher_session = teacher_session

        # new addition
        # if phase == 'train':
        #     self.inductive = inductive
        # else:
        #     self.inductive = False

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        print(f"root path: {self.root}")
        self.loader = ImageLoader(self.root + '/images/')
        
        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            print(f"in open world: {self.open_world}")
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        # print(f"self.train_data: {self.train_data}")
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data
        
        selected_session = self.paths.selected_session_g_pair2_idx 
        g_obj2idx = self.paths.g_objattrpair2idx_path[selected_session]['g_obj2idx']
        g_obj2idx = {v: i for i, v in enumerate(g_obj2idx)}

        g_attr2idx = self.paths.g_objattrpair2idx_path[selected_session]['g_attr2idx']
        g_attr2idx = {v: i for i, v in enumerate(g_attr2idx)}

        g_pair2idx = self.paths.g_objattrpair2idx_path[selected_session]['g_pair2idx']
        print(f"g_pair2idx without mapping: {g_pair2idx}")

        g_pair2idx = {v: i for i, v in enumerate(g_pair2idx)}
        print("#######################################")
        print(f"g_pair2idx: {g_pair2idx}")
        print("#######################################")

        print(f"g_obj2idx: {g_obj2idx}")

        g_train_pair = {
            tuple(line.strip().split()): int(idx)
            for idx, line in enumerate(open(self.paths.train_val_test_txtfiles[selected_session]+ "/train_pairs.txt", encoding="utf-8"))
            if line.strip()
        }

        # print(f"g_train_pair: {g_train_pair}")

        g_val_pair = {
            tuple(line.strip().split()): int(idx)
            for idx, line in enumerate(open(self.paths.train_val_test_txtfiles[selected_session]+ "/val_pairs.txt", encoding="utf-8"))
            if line.strip()
        }
        # print(f"g_val_pair: {g_val_pair}\n")
        g_test_pair = {
            tuple(line.strip().split()): int(idx)
            for idx, line in enumerate(open(self.paths.train_val_test_txtfiles[selected_session]+ "/test_pairs.txt", encoding="utf-8"))
            if line.strip()
        }

        #old code 
        self.obj2idx = {obj: g_obj2idx[obj] for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: g_attr2idx[attr] for idx, attr in enumerate(self.attrs)}
        # print(f"self.pairs: {self.pairs}")
        type(self.pairs)
        self.pair2idx = {pair: g_pair2idx[pair] for idx, pair in enumerate(self.pairs)}

       

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, g_train_pair[pair]) for idx, pair in enumerate(self.train_pairs)]
        )
        self.val_pair_to_idx = dict(
            [(pair, g_val_pair[pair]) for idx, pair in enumerate(self.val_pairs)]
        )
        self.test_pair_to_idx = dict(
            [(pair, g_test_pair[pair]) for idx, pair in enumerate(self.test_pairs)]
        )
       

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

    def get_split_info(self):
        t7path = self.paths.teacher_t7_files[self.session] 
        print(f"t7 path: {t7path}")
        data = torch.load(t7path)

        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            # print(f"data_i: {data_i}")
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        # print(f'train_data: {train_data}')
        # print(f'val_data: {val_data}')
        # print(f'test_data: {test_data}')
        
        return train_data, val_data, test_data


    def parse_split(self):
        def parse_pairs(pair_list):
            print(f"pairs list: {pair_list}")
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = f.read().splitlines()
                print(f"teacher pairs: {pairs}")

                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%strain_pairs.txt' % (self.paths.train_val_test_txtfiles[self.teacher_session], ""))
            # '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%sval_pairs.txt' % (self.paths.train_val_test_txtfiles[self.teacher_session], ""))
            # '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%stest_pairs.txt' % (self.paths.train_val_test_txtfiles[self.teacher_session], ""))
            # '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))
        print(all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs)
        print("\n\n")
       
        all_attrs = self.paths.ALL_ATTR_OBJ[self.teacher_session]['all_attrs']
        all_objs = self.paths.ALL_ATTR_OBJ[self.teacher_session]['all_objs']
        all_pairs = self.paths.ALL_ATTR_OBJ[self.teacher_session]['all_pairs']
        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        return data

    def __len__(self):
        return len(self.data)