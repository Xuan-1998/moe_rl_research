import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def main():
    model_name = "deepseek-ai/deepseek-moe-16b-chat"
    print(f"Loading {model_name} for precomputation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
    model.config.use_cache = False # Disable cache to avoid compatibility issues with remote code
    
    # Data to run
    try:
        from datasets import load_dataset
        print("Loading wikitext dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Filter for reasonable length and take a subset
        prompts = [x['text'] for x in dataset if len(x['text'].strip()) > 100][:2000]
        print(f"Loaded {len(prompts)} prompts from wikitext.")
    except ImportError:
        print("Warning: 'datasets' library not found. Falling back to dummy prompts.")
        prompts = [
            "Explain the importance of topology-aware routing in distributed systems. " * 2,
            "The history of the Roman Empire is characterized by rise and fall of power.",
            "DeepSeek-MoE is a mixture of experts model designed for efficient inference.",
        ] * 100
    
    inputs_list = [tokenizer(p, return_tensors="pt", truncation=True, max_length=512).to(model.device) for p in prompts]
    
    print("Running forward passes to capture hidden states...")
    
    all_inputs = []
    all_outputs = []
    
    def hook_fn(module, args, output):
        # args[0] is hidden_states [batch, seq, dim]
        all_inputs.append(args[0].detach().cpu())
        
        # Check if output is tuple
        out_tensor = output[0] if isinstance(output, tuple) else output
        all_outputs.append(out_tensor.detach().cpu())
        
    # Find first MoE block (using hooking logic from before)
    target_layer = None
    for n, m in model.named_modules():
        if n.endswith("model.layers.0.mlp"): 
            target_layer = m
            break
            
    if target_layer is None:
        for name, module in model.named_modules():
             if hasattr(module, 'experts') or hasattr(module, 'routed_experts'): 
                  target_layer = module
                  print(f"Hooking into {name}")
                  break
    
    if target_layer is None:
         print("CRITICAL: No MoE layer found.")
         return

    handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for inp in inputs_list:
            model(**inp)
        
    handle.remove()
    
    # Concatenate
    final_input = torch.cat(all_inputs, dim=1) # Concat along seq dim (batch=1)
    final_output = torch.cat(all_outputs, dim=1)
    
    # Flatten batch and seq
    final_input = final_input.view(-1, final_input.shape[-1])
    final_output = final_output.view(-1, final_output.shape[-1])
    
    print(f"Captured Total Input Shape: {final_input.shape}")
    print(f"Captured Total Output Shape: {final_output.shape}")
    
    captured = {
        'input': final_input.unsqueeze(0), # Add batch dim back for env compatibility if needed?
        # Env expects [batch, seq, dim] or [total_seq, dim]?
        # Env treats it as one long sequence or batch=1.
        # Let's keep it [1, Total_Seq, Dim]
    }
    # Reshape back to [1, Total, Dim]
    captured['input'] = final_input.unsqueeze(0)
    captured['output'] = final_output.unsqueeze(0)
    
    os.makedirs("data", exist_ok=True)
    torch.save(captured, "data/calibration_data.pt")
    print("Saved data/calibration_data.pt")

if __name__ == "__main__":
    main()
