import torch
from utils import get_answer_labels
from prompts import build_prompt

def test_labels():
    # [.., im_start, assistant, \n, 10, 11, 12, im_end, \n]
    im_start = 151644
    assistant = 77091
    im_end = 151645
    
    seq = torch.tensor([[100, 101, im_start, assistant, 198, 10, 11, 12, im_end, 198]])
    labels = get_answer_labels(seq, im_start, assistant, im_end)
    
    expected = torch.tensor([[-100, -100, -100, -100, -100, 10, 11, 12, im_end, -100]])
    
    assert torch.equal(labels, expected), f"Expected {expected}, got {labels}"

def test_prompts():
    p = build_prompt("spatial_reasoning", "The dog is on the left.")
    assert p == "Statement: The dog is on the left.\nIs this statement true or false about the image? Answer True or False."

if __name__ == "__main__":
    test_labels()
    test_prompts()
    print("All tests passed.")
