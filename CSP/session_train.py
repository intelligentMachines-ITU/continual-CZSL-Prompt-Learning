import argparse
import os
import pickle
import pprint
import torch.nn.functional as F
import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader

from datasets.composition_dataset import CompositionDataset

from models.compositional_modules import get_model
from utils import set_seed

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

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

def train_model(model, optimizer, train_dataset, config, device, teacher_model, paths):
   
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    model.train()

    loss_fn = CrossEntropyLoss()

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx
    print(attr2idx)
    print(obj2idx)

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).to(device)

    teacher_pairs = train_pairs[:paths.TRAIN_PAIRS[config.teacher_session]]
    print(f"Teacher pairs: {paths.TRAIN_PAIRS[config.teacher_session]}")

    # print("Teacher pairs")
    # print(teacher_pairs)
    pair_end = paths.TRAIN_PAIRS[config.teacher_session] 

    print(f"config.session: {config.session}, config.teacher_session {config.teacher_session}")

    alpha = config.alpha
    i = 0
    train_losses = []

    torch.autograd.set_detect_anomaly(True)

    for i in range(config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):
            batch_img, batch_target = batch[0], batch[3]
            batch_target = batch_target.to(device)
            batch_img = batch_img.to(device)
            batch_feat = model.encode_image(batch_img)
            with torch.no_grad():
                t_pair_logits = teacher_model(batch_feat, teacher_pairs)

            logits = model(batch_feat, train_pairs)
            
            distill_loss = get_distillation_loss(logits, t_pair_logits, pair_end, temperature=2.5)

            task_loss = loss_fn(logits, batch_target)
            loss = (1-alpha)*task_loss + alpha * distill_loss

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            loss.backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or \
                    (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix(
                {"train loss": np.mean(epoch_train_losses[-50:])}
            )

            progress_bar.update()

        progress_bar.close()
        progress_bar.write(
            f"epoch {i +1} train loss {np.mean(epoch_train_losses)}"
        )
        train_losses.append(np.mean(epoch_train_losses))

        if (i + 1) % config.save_every_n == 0:
            save_soft_embeddings(model, config, epoch=i + 1)

    return model, optimizer

def compute_kl(student_logits, teacher_logits, temperature=1.0):
    
    s_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    t_soft = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (temperature ** 2)


def get_distillation_loss(student_logits, teacher_logits, pair_end, temperature=2.5):
   
    s_pair_logits = student_logits
    t_pair_logits  = teacher_logits


    s_pair_aligned = s_pair_logits[:, :pair_end]
    t_pair_aligned = t_pair_logits[:, :pair_end]
    
    
    pair_loss = compute_kl(s_pair_aligned, t_pair_aligned, temperature)

    return pair_loss
    
def save_soft_embeddings(model, config, epoch=None):
    
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # save the soft embedding
    with torch.no_grad():
        if epoch:
            soft_emb_path = os.path.join(
                config.save_path, f"soft_embeddings_epoch_{epoch}.pt"
            )
        else:
            soft_emb_path = os.path.join(
                config.save_path, "soft_embeddings.pt"
            )

        torch.save({"soft_embeddings": model.soft_embeddings}, soft_emb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str,
    )
    parser.add_argument("--dataset", help="name of the dataset", type=str)
    parser.add_argument(
        "--lr", help="learning rate", type=float, default=5e-05
    )
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float, default=1e-05
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-B/32"
    )
    parser.add_argument(
        "--epochs", help="number of epochs", default=20, type=int
    )
    parser.add_argument(
        "--train_batch_size", help="train batch size", default=64, type=int
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=1024, type=int
    )
    parser.add_argument(
        "--evaluate_only",
        help="directly evaluate on the" "dataset without any training",
        action="store_true",
    )
    parser.add_argument(
        "--context_length",
        help="sets the context length of the clip model",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--attr_dropout",
        help="add dropout to attributes",
        type=float,
        default=0.0,
    )
    parser.add_argument("--save_path", help="save path", type=str)
    parser.add_argument(
        "--save_every_n",
        default=1,
        type=int,
        help="saves the model every n epochs; "
        "this is useful for validation/grid search",
    )
    parser.add_argument(
        "--save_model",
        help="indicate if you want to save the model state dict()",
        action="store_true",
    )

    parser.add_argument(
        "--soft_embeddings",
        help="location for softembeddings",
        type=str,
        default="./soft_embeddings.pt",
    )
    
    parser.add_argument("--seed", help="seed value", default=0, type=int)

    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=1,
        type=int
    )

    parser.add_argument(
        "--session",
        help="current config.session",
        default="mit-states-s1" ,
        type=str
    )

    parser.add_argument(
        "--teacher_session",
        help="previous(teacher) config.session",
        default="mit-states-s0" ,
        type=str
    )

    config = parser.parse_args()

    # set the seed value
    set_seed(config.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("training details")
    pprint.pprint(config)

    if os.path.exists(config.save_path):
        print('file already exists')
        print('exiting!')
        exit(0)

    # This should work for mit-states, ut-zappos, and maybe c-gqa.
    paths = import_dataset_paths(config.dataset)        

    print("Loading Teacher Model")
    print("**********************************************************************")
    if config.soft_embeddings:
        def load_teacher_data():
            from datasets.teacher_dataset import CompositionDataset
            dataset_path = paths.DATASET_PATHS[config.dataset]
            train_dataset = CompositionDataset(dataset_path,
                                               phase='train',
                                               split='compositional-split-natural',
                                               session=config.session,
                                               teacher_session=config.teacher_session,
                                               dataset_name= config.dataset)
           
            return train_dataset
        train_dataset = load_teacher_data()
        
        
        teacher_model, optimizer = get_model(train_dataset, config, device)
        soft_embs = torch.load(config.soft_embeddings)['soft_embeddings']
        teacher_model.set_soft_embeddings(soft_embs)
        teacher_state_dict = teacher_model.state_dict()
        print("model dtype", teacher_model.dtype)
        print("soft embedding dtype", teacher_model.soft_embeddings.dtype)
        # print("teacher_state_dict", teacher_model.state_dict().keys())
        
    print("Loading Student Model")
    print("**********************************************************************")
    from datasets.composition_dataset import CompositionDataset  
    dataset_path = paths.DATASET_PATHS[config.dataset]
    train_dataset = CompositionDataset(dataset_path,
                                               phase='train',
                                               split='compositional-split-natural',
                                               session=config.session,
                                               teacher_session=config.teacher_session,
                                               dataset_name= config.dataset
                                               )
    
    
    model, optimizer = get_model(train_dataset, config, device)
    student_state_dict = model.state_dict()
    # print("State dict of student model", student_state_dict.keys())
    matched = {}
    
    for k, v in teacher_state_dict.items():
        if k in student_state_dict:
            # if same shape, copy directly
            if student_state_dict[k].shape == v.shape:
                matched[k] = v
            else:
                print(f"⚠️ Shape mismatch for key '{k}': "
                      f"teacher {tuple(v.shape)} vs student {tuple(student_state_dict[k].shape)} (skipped)")
                t = teacher_state_dict["soft_embeddings"]
                s = student_state_dict["soft_embeddings"].clone()

                T_ATTRS, T_OBJS = paths.ALL_ATTR_OBJ_PAIRS[config.teacher_session][0], paths.ALL_ATTR_OBJ_PAIRS[config.teacher_session][1]
                S_ATTRS, S_OBJS = paths.ALL_ATTR_OBJ_PAIRS[config.session][0], paths.ALL_ATTR_OBJ_PAIRS[config.session][1]

                # ✅ copy attributes (first 8 rows)
                s[0:T_ATTRS, :] = t[0:T_ATTRS, :]
            
                # ✅ copy objects (rows after attr block)
                s[S_ATTRS:S_ATTRS + T_OBJS, :] = t[T_ATTRS:T_ATTRS + T_OBJS, :]
            
                student_state_dict["soft_embeddings"] = s
            
                print(f"✅ Copied attributes [0:{T_ATTRS}) and objects [{S_ATTRS}:{S_ATTRS + T_OBJS}) "
                      f"from teacher to student (out of total {s.shape[0]} rows).")

                      
        else:
            print(f"⛔ Key '{k}' not found in student model (skipped)")
    
    # update matching weights
    student_state_dict.update(matched)
    
    # load them into student (non-strict to allow missing keys)
    model.load_state_dict(student_state_dict, strict=False)
    
    print(f"✅ Loaded {len(matched)} matching parameters from teacher into student.")
    

    if not config.evaluate_only:
        model, optimizer = train_model(
            model,
            optimizer,
            train_dataset,
            config,
            device,
            teacher_model, paths
        )

    save_soft_embeddings(
        model,
        config,
    )

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)

    if config.save_model:
        torch.save(
            model.dict(),
            os.path.join(
                config.save_path,
                'final_model.pt'))

    print("done!")