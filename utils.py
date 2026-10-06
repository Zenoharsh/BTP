import torch

def get_answer_labels(input_ids, im_start_id, assistant_id, im_end_id):
    """
    Centralized helper for strict answer-token masking.
    Returns a label tensor where everything is -100 EXCEPT the actual 
    assistant answer tokens.
    Excludes:
      - System/User prompt tokens
      - Padding
      - <|im_start|> assistant \\n
      - <|im_end|>
    """
    labels = input_ids.clone()
    labels[:] = -100
    
    for i in range(labels.size(0)):
        seq = input_ids[i]
        start_indices = (seq == im_start_id).nonzero(as_tuple=True)[0]
        for start_idx in start_indices:
            if start_idx + 1 < len(seq) and seq[start_idx + 1] == assistant_id:
                end_idx_candidates = (seq[start_idx:] == im_end_id).nonzero(as_tuple=True)[0]
                if len(end_idx_candidates) > 0:
                    end_idx = start_idx + end_idx_candidates[0]
                    # We want tokens strictly AFTER '<|im_start|> assistant \\n' (which is start_idx + 3)
                    # And including '<|im_end|>' (which is end_idx)
                    if end_idx >= start_idx + 3:
                        labels[i, start_idx+3 : end_idx+1] = seq[start_idx+3 : end_idx+1]
                        
    return labels
