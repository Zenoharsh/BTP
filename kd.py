"""
Sparse knowledge distillation shared by scripts/cache_teacher.py (T6) and train.py (T7).

Teacher targets are stored for ANSWER tokens only (the positions get_answer_labels supervises,
including <|im_end|>), shifted for next-token prediction exactly like the CE loss:
    logits[:, :-1]  predict  input_ids[:, 1:]   ->   shift_mask = answer_mask[1:]
"""
import torch
import torch.nn.functional as F

from utils import get_answer_labels


def answer_shift_mask(input_ids_1d, token_ids):
    """[L] input ids of ONE unpadded sequence -> [L-1] bool mask of logit positions that predict an
    answer token."""
    labels = get_answer_labels(input_ids_1d.unsqueeze(0), token_ids["im_start"],
                               token_ids["assistant"], token_ids["im_end"])[0]
    return (labels != -100)[1:]


@torch.no_grad()
def teacher_targets(logits_1d, shift_mask, temperature=2.0, k=50):
    """logits_1d: [L, V] teacher logits of one unpadded sequence. Returns the cache entry."""
    sel = logits_1d[:-1][shift_mask].float()                          # [n, V]
    probs = F.softmax(sel / temperature, dim=-1)
    top = probs.topk(k, dim=-1)
    return {"probs": top.values.half().cpu(),                         # [n, k] (NOT renormalised)
            "indices": top.indices.to(torch.int32).cpu(),             # [n, k]
            "mask": shift_mask.bool().cpu(),                          # [L-1]
            "seq_len": int(logits_1d.shape[0]),                       # L
            "topk_mass": top.values.sum(-1).float().cpu(),            # [n] mass before renorm
            "teacher_top1": sel.argmax(-1).to(torch.int32).cpu(),     # [n] for sanity checks
            "temperature": float(temperature), "k": int(k)}


def sparse_kd(student_logits, indices, probs, temperature=2.0):
    """KL(teacher_topk_renormalised || student) on the teacher's top-k support, T^2 scaled.
    student_logits: [n, V] (any dtype; cast to fp32 here), indices: [n, k], probs: [n, k]."""
    s = student_logits.float() / temperature
    s_logp = s.gather(-1, indices.long()) - torch.logsumexp(s, dim=-1, keepdim=True)
    t = probs.float()
    t = t / t.sum(-1, keepdim=True).clamp_min(1e-8)
    kl = (t * (t.clamp_min(1e-8).log() - s_logp)).sum(-1).mean()
    return kl * temperature ** 2
