
#
import os
import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model.model_factory import get_model
from parameters import parser

# from test import *
import test as test
from dataset import CompositionDataset
from utils import *

def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset,teacher_model):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    model.train()
    best_metric = 0
    best_loss = 1e5
    best_epoch = 0
    final_model_state = None
    
    val_results = []
    
    scheduler = get_scheduler(optimizer, config, len(train_dataloader))
    attr2idx = train_dataset.attr2idx
   
    obj2idx = train_dataset.obj2idx

    
    
    
    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()
                                
                            
    
 
    
    teacher_pairs = train_pairs[:config.T_PAIRS]
   
                                
    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )
        
        if i > 9:
            alpha = config.alpha
        else:
           alpha = config.alpha
       
        epoch_train_losses = []
        epoch_orhto = []
        epoch_cosine = []
        attr_end = config.T_ATTRS
        obj_end = config.T_OBJS
        pair_end = config.T_PAIRS


        lam_anchor = 1e-3  # tune in [3e-4, 3e-3]
        for bid, batch in enumerate(train_dataloader):
        
            with torch.no_grad():
                
                teacher_output = teacher_model(batch, teacher_pairs)
                      
            predict = model(batch, train_pairs)

            # Unpack logits
            s_pair_logits, s_attr_logits, s_obj_logits = predict
            t_pair_logits, t_attr_logits, t_obj_logits = teacher_output
        
            
            
            distill_loss = get_distillation_loss(predict, teacher_output, attr_end, obj_end, pair_end, temperature=2.5)
            
            task_loss = model.loss_calu(predict, batch)
            loss = (1-alpha)*task_loss + alpha * distill_loss
            

            s_soft = model.soft_att_obj                               
            t_soft = teacher_model.soft_att_obj.detach().to(s_soft.device, s_soft.dtype)  
          

            # L_anchor_soft = cosine_anchor(s_soft[0:8],     t_soft[0:8])               #               + cosine_anchor(s_soft[12:18],   t_soft[8:14])

            L_anchor_soft = cosine_anchor(s_soft[0:T_ATTRS],     t_soft[0:T_ATTRS])                             + cosine_anchor(s_soft[S_ATTRS:S_ATTRS+T_OBJS],   t_soft[T_ATTRS:T_ATTRS+T_OBJS])

            
            loss = loss + config.L_CAL * L_anchor_soft

            s_old_attrs = s_soft[0:T_ATTRS]                
            s_new_attrs = s_soft[T_ATTRS:S_ATTRS]       
            s_old_objs  = s_soft[S_ATTRS:S_ATTRS+T_OBJS] 
            s_new_objs  = s_soft[S_ATTRS+T_OBJS:]      
            
            s_old_all = torch.cat([s_old_attrs, s_old_objs], dim=0)    
            s_new_all = torch.cat([s_new_attrs, s_new_objs], dim=0)   
            
           
            L_ortho_attr = ortho_new_vs_old(s_new_attrs, s_old_attrs)
            L_ortho_obj  = ortho_new_vs_old(s_new_objs,  s_old_objs)
            L_ortho_prompts = L_ortho_attr + L_ortho_obj

                    
            
            loss = loss + config.L_ORTHO * L_ortho_prompts
           
            lambda_div = config.L_DIV  # gentle, safe value
            L_div_attr = diversity_loss(s_new_attrs)
            L_div_obj  = diversity_loss(s_new_objs)
            loss += lambda_div * (L_div_attr + L_div_obj)

            
            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            loss.backward()
            
            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))

            epoch_train_losses.append(loss.item())
            epoch_cosine.append((config.L_CAL * L_anchor_soft).item())
            epoch_orhto.append(config.L_ORTHO)

            
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:]),"Cossloss" : np.mean(epoch_cosine[-50:]),"Ortholoss" : np.mean(epoch_orhto[-50:]) })

            progress_bar.update()
        

        progress_bar.close()
        progress_bar.write(f"epoch {i+1} train loss {np.mean(epoch_train_losses)}")
        progress_bar.write(f"epoch {i+1} Cosine loss {np.mean(epoch_cosine)}")
        progress_bar.write(f"epoch {i+1} Ortho loss {np.mean(epoch_orhto)}")
        
        
        train_losses.append(np.mean(epoch_train_losses))
        
        

        if (i + 1) % config.save_every_n == 0:
            torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))

        print("Evaluating val dataset:")
        val_result = evaluate(model, val_dataset, config)
        val_results.append(val_result)

        if config.val_metric == 'best_loss' and val_result[config.val_metric] < best_loss:
            best_loss = val_result['best_loss']
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(
                config.save_path, "val_best.pt"))
        if config.val_metric != 'best_loss' and val_result[config.val_metric] > best_metric:
            best_metric = val_result[config.val_metric]
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(
                config.save_path, "val_best.pt"))

        final_model_state = model.state_dict()
        if i + 1 == config.epochs:
            print("--- Evaluating test dataset on Closed World ---")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "val_best.pt"
            )))
            evaluate(model, test_dataset, config)

    if config.save_final_model:
        torch.save(final_model_state, os.path.join(config.save_path, f'final_model.pt'))


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
            model, dataset, config)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    test_saved_results = dict()
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
  
    test_saved_results['loss'] = loss_avg
    return test_saved_results


def _offdiag(M: torch.Tensor):
    n = M.size(0)
    return M.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()

    
def diversity_loss(S_new, eps=1e-8):
    if S_new.numel() == 0:
        return S_new.new_tensor(0.0)
    S = F.normalize(S_new, dim=-1, eps=eps)
    sim = torch.matmul(S, S.t())       # pairwise cosine similarities
    I = torch.eye(sim.size(0), device=sim.device)
    sim = sim * (1 - I)                # ignore self-similarity
    return sim.abs().mean()


def gram_diversity(W: torch.Tensor, eps: float = 1e-8):
    # Encourage rows of W to be mutually orthogonal (spread out)
    if W.numel() == 0 or W.size(0) <= 1:
        return W.new_tensor(0.0)
    Wn = F.normalize(W, dim=-1, eps=eps)
    G  = Wn @ Wn.t()
    return (_offdiag(G)**2).mean()

def ortho_new_vs_old(S_new, S_old, eps=1e-8):
    if S_new.numel() == 0 or S_old.numel() == 0:
        return S_new.new_tensor(0.0)
    Sn = F.normalize(S_new, dim=-1, eps=eps)
    So = F.normalize(S_old,  dim=-1, eps=eps)
    proj = Sn @ So.t()
    return (proj.abs()).mean()  # soft constraint, not squared


def cosine_anchor(a, b, eps=1e-8):
    # a, b: [N, D]
    a_n = F.normalize(a, dim=-1, eps=eps)
    b_n = F.normalize(b, dim=-1, eps=eps)
    return (1.0 - (a_n * b_n).sum(dim=-1)).mean()

    
def compute_kl(student_logits, teacher_logits, temperature=1.0):
    
    s_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    t_soft = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (temperature ** 2)

def get_distillation_loss(student_logits, teacher_logits, attr_end, obj_end, pair_end, temperature=2.5):
    # 
    # Compute distillation loss across attribute, object, and pair outputs.
    
    # Args:
    #     student_logits: Tuple of logits from student model (pair, attr, obj)
    #     teacher_logits: Tuple of logits from teacher model (pair, attr, obj)
    #     attr_end: int, number of attributes (shared with teacher)
    #     obj_end: int, number of objects (shared with teacher)
    #     pair_end: int, number of pairs (shared with teacher)
    #     temperature: float, softening temperature for KL

    # Returns:
    #     Total KL-based distillation loss
    # 
    s_pair_logits, s_attr_logits, s_obj_logits = student_logits
    t_pair_logits, t_attr_logits, t_obj_logits = teacher_logits

    # Align the logits using fixed known output ranges
    s_attr_aligned = s_attr_logits[:, :attr_end] 
    t_attr_aligned = t_attr_logits[:, :attr_end]
  

    s_obj_aligned = s_obj_logits[:, :obj_end]
    t_obj_aligned = t_obj_logits[:, :obj_end]
    

    s_pair_aligned = s_pair_logits[:, :pair_end]
    t_pair_aligned = t_pair_logits[:, :pair_end]
    
    

    # Compute KL divergence loss for each head
    attr_loss = compute_kl(s_attr_aligned, t_attr_aligned, temperature)
    obj_loss = compute_kl(s_obj_aligned, t_obj_aligned, temperature)
    pair_loss = compute_kl(s_pair_aligned, t_pair_aligned, temperature)

    return attr_loss + obj_loss + pair_loss



if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)
    print(config)
    # set the seed value
    set_seed(config.seed)

    dataset_path = config.dataset_path

    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural',
                                       same_prim_sample=config.same_prim_sample)

    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural')

    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural')

    allattrs = train_dataset.attrs
  
    
   
    allobj = train_dataset.objs
    
  
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    
    offset = len(attributes)
  
    attr_idx_path = os.path.join(dataset_path, 'compositional-split-natural', f"{config.T_ATTR_IDX}.json")

    with open(attr_idx_path, 'r') as f:
        s0_attr2idx = json.load(f)
    attributes_session0 = list(s0_attr2idx.keys())
   
    
    obj_idx_path = os.path.join(dataset_path, 'compositional-split-natural', f"{config.T_OBJ_IDX}.json")

    with open(obj_idx_path, 'r') as f:
        s0obj2idx = json.load(f)
    classes_session0 = list(s0obj2idx.keys())

    
 
    teacher_model = get_model(config, attributes=attributes_session0, classes=classes_session0, offset=len(attributes_session0)).cuda()
   
    if config.load_model:
      
        state_dict = torch.load(config.load_model)
        teacher_model.load_state_dict(state_dict)
        print("Loaded teacher model successfully")
    
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()

    if config.load_model:
        teacher_state_dict = torch.load(config.load_model)
        print("teacher_state_dict")
       
        student_state_dict = model.state_dict()
        
        matched = {}
       
    
        for k, v in teacher_state_dict.items():
            if k in student_state_dict:
                if student_state_dict[k].shape == v.shape:
                    
                    matched[k] = v
               
                elif 'soft_att_obj' in k:
                                       
                    T_ATTRS, T_OBJS = config.T_ATTRS, config.T_OBJS
                    S_ATTRS, S_OBJS = config.S_ATTRS, config.S_OBJS
                    T_SHAPE = config.T_SHAPE
                    S_SHAPE = config.S_SHAPE
                  

                    t = v.to(dtype=student_state_dict[k].dtype)          
                    s = student_state_dict[k].clone()
                    
                    assert t.ndim == 2 and s.ndim == 2, f"{k} must be 2D"
                    assert t.shape[1] == s.shape[1], f"{k} dim mismatch on D: {t.shape[1]} vs {s.shape[1]}"
                    assert t.shape[0] == T_ATTRS + T_OBJS == T_SHAPE, f"teacher rows={t.shape[0]} unexpected"
                    assert s.shape[0] == S_ATTRS + S_OBJS == S_SHAPE, f"student rows={s.shape[0]} unexpected"
   
                    s[0:T_ATTRS, :] = t[0:T_ATTRS, :]
                
                    s[S_ATTRS:S_ATTRS + T_OBJS, :] = t[T_ATTRS:T_ATTRS + T_OBJS, :]    
                    matched[k] = s
                    print(f"Hard-loaded {k}: attrs 0:{T_ATTRS}, objs {S_ATTRS}:{S_ATTRS + T_OBJS} from teacher.")
                
                elif any(x in k for x in ['comp_ctx_vectors', 'attr_ctx_vectors', 'obj_ctx_vectors']):
                   
                    t = v.to(dtype=student_state_dict[k].dtype)
                    s = student_state_dict[k].clone()
                    m = min(t.shape[0], s.shape[0])
                    s[:m, :] = t[:m, :]
                    matched[k] = s
                    print(f"Partially loaded {k}: rows 0:{m} copied, rest kept from student.")

                else:
                    print(f"Skipped {k}: shape mismatch {v.shape} vs {student_state_dict[k].shape}")
    
        student_state_dict.update(matched)
        model.load_state_dict(student_state_dict, strict=False)
        print(f"Loaded {len(matched)} shared/partial parameters from teacher into student.")
        with torch.no_grad():
            s = model.state_dict()['soft_att_obj']
            # compare copied rows
            assert torch.allclose(s[0:T_ATTRS], t[0:T_ATTRS], atol=1e-6)
            assert torch.allclose(s[S_ATTRS:S_ATTRS+T_OBJS], t[T_ATTRS:T_ATTRS+T_OBJS], atol=1e-6)

        print("soft_att_obj rows match where expected.")

    
        

    
    optimizer = get_optimizer(model, config)

    os.makedirs(config.save_path, exist_ok=True)

    train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset, teacher_model)

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    write_json(os.path.join(config.save_path, "config.json"), vars(config))
    print("done!")

 
