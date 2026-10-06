TEMPLATES = {
  "document_ocr":      "{q}\nAnswer the question using a single word or phrase.",
  "chart_qa":          "{q}\nAnswer the question using a single word or phrase.",
  "spatial_reasoning": "Statement: {q}\nIs this statement true or false about the image? Answer True or False.",
}

def build_prompt(task, question): 
    return TEMPLATES[task].format(q=question.strip())
