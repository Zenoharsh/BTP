from pptx import Presentation
from pptx.util import Inches, Pt

def add_slide(prs, title, content_lines, is_title_slide=False):
    if is_title_slide:
        slide_layout = prs.slide_layouts[0] # Title Slide layout
        slide = prs.slides.add_slide(slide_layout)
        title_placeholder = slide.shapes.title
        subtitle_placeholder = slide.placeholders[1]
        
        title_placeholder.text = title
        subtitle_placeholder.text = "\n".join(content_lines)
    else:
        slide_layout = prs.slide_layouts[1] # Title and Content layout
        slide = prs.slides.add_slide(slide_layout)
        title_placeholder = slide.shapes.title
        content_placeholder = slide.placeholders[1]
        
        title_placeholder.text = title
        tf = content_placeholder.text_frame
        tf.clear()
        
        for line in content_lines:
            p = tf.add_paragraph()
            p.text = line
            p.font.size = Pt(18)
            p.space_before = Pt(10)
            
    return slide

def generate_presentation():
    prs = Presentation()

    # Slide 1: Title Slide
    add_slide(prs, 
              "Resource-Constrained Sparse Upcycling", 
              ["Evaluating Top-K Routing Stability and Visual Modality Clustering in Quantized Edge-VLMs",
               "Presenter: [Your Name / Harsh Raj] & Team",
               "Roles: ML Architect, Data & DevOps Engineer, Edge Systems Developer"], 
              is_title_slide=True)

    # Slide 2: The Problem Statement
    add_slide(prs,
              "The Bottleneck of Dense Multimodal AI on Edge",
              ["The Compute Crisis: Modern Vision-Language Models (VLMs) activate 100% of their parameters for every single token (text or image patch).",
               "Edge Hardware Failure: Deploying standard dense models on local, constrained hardware (drones, mobile, robotics) causes Out-Of-Memory (OOM) crashes and extreme latency.",
               "The Research Gap: While Mixture-of-Experts (MoE) architectures solve compute bottlenecks for massive cloud servers, their behavior and stability under extreme low-bit quantization on edge devices remains largely unexplored."])

    # Slide 3: Our Novel Solution (The Conference Hook)
    add_slide(prs,
              "Edge-Optimized MoE via Sparse Upcycling",
              ["The Objective: We are taking a dense VLM, upcycling it into a 4x MoE architecture, and deploying it on constrained edge hardware.",
               "Target Metrics: Achieve a 40% reduction in active inference latency while maintaining 95% of the zero-shot accuracy on domain-specific multimodal tasks.",
               "Novelty: Unlike natively trained models, we are utilizing 'Sparse Upcycling'—cloning pre-trained dense weights and inducing sparsity at a fraction of the traditional compute cost."])

    # Slide 4: Phase 1 - Architectural Surgery
    add_slide(prs,
              "Replacing Dense MLPs with Sparse Routing",
              ["Base Model: Qwen1.5-0.5B (acting as our initial sandbox baseline before scaling).",
               "Expert Initialization: We replaced the standard linear layers with a custom Sparse Routing mechanism. We initialized 4 identical 'Experts' by copying the original dense weights.",
               "Hardware Constraint Logic: We engineered strict Token Dropping (capacity limits). If an expert is overloaded, overflow tokens bypass the expert via residual connections to prevent edge VRAM crashes."])

    # Slide 5: Phase 2 - Distillation & Router Training
    add_slide(prs,
              "Teaching the Router without Catastrophic Forgetting",
              ["The Strategy: We froze all the expert weights and are currently only training the Gating Network (the router).",
               "Knowledge Distillation: We use the original dense model as a frozen 'Teacher' and train our upcycled MoE 'Student' to mimic its logit distributions using KL-Divergence.",
               "Load Balancing Loss: We implemented a critical mathematical constraint to force the router to distribute the computational load evenly across experts, preventing 'expert collapse'.",
               "We also integrated a Router Z-Loss to prevent fp16 logit overflow."])

    # Slide 6: Current Progress & Codebase
    add_slide(prs,
              "What We Have Built So Far",
              ["Core Architecture Locked: The foundational PyTorch scripts are fully written and verified.",
               "layers.py: Contains our custom MoELayer with the cloned experts and strict token-dropping algorithms.",
               "router.py: Contains the TopKRouter gating network and auxiliary loss functions.",
               "train.py: The active distillation training loop utilizing PEFT (QLoRA) parameter targeting."])

    # Slide 7: Next Steps (Phase 3)
    add_slide(prs,
              "Domain-Specific Expert Tuning",
              ["Diverse Datasets: We will feed the model highly diverse multimodal data to force expert specialization (e.g., Expert A on document OCR, Expert B on spatial reasoning, Expert C on scientific diagrams).",
               "QLoRA Tuning: We will apply Parameter-Efficient Fine-Tuning (specifically QLoRA) to the individual experts.",
               "Evaluation: We will monitor the router to verify that it naturally learns to send text-heavy images to Expert A and charts to Expert C."])

    # Slide 8: The Final Deliverable (Phase 4)
    add_slide(prs,
              "Quantization & Edge Deployment",
              ["Model Compression: Convert the final MoE weights to GGUF format using llama.cpp.",
               "Quantization: Apply 4-bit quantization to shrink the memory footprint drastically.",
               "Hardware Benchmarking: Deploy the model onto an edge device (such as a Raspberry Pi 5 or Jetson Nano) and measure the real-world performance gains against our Phase 1 baseline.",
               "Tip for your presentation: Show the professor your GitHub repository file tree (layers.py, router.py, train.py) as proof of active progress!"])

    output_path = "BTP_Project_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}")

if __name__ == "__main__":
    generate_presentation()
