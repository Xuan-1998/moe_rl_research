
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os

def prepare_dataset(name, config_name, split, prompt_field='text', limit=200):
    print(f"Processing {name}/{config_name}...")
    try:
        if config_name:
            ds = load_dataset(name, config_name, split=split)
        else:
            ds = load_dataset(name, split=split)
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return None

    # Extract text based on dataset structure
    texts = []
    for item in ds:
        if len(texts) >= limit: break
        
        text = ""
        if name == 'winogrande':
            text = item['sentence']
        elif name == 'piqa':
            text = item['goal'] + " " + item['sol1']
        elif name == 'super_glue':
            if config_name == 'wsc':
                text = item['text']
            elif config_name == 'rte':
                text = item['premise'] + " " + item['hypothesis']
        elif name == 'race':
            # article + question
            text = item['article'] + "\n" + item['question']
        elif name == 'math_qa':
            # problem + options
            text = item['Problem'] + "\n" + item['options']
        else:
            text = item.get(prompt_field, "")
            
        if len(text.strip()) > 10:
            texts.append(text)
            
    print(f"Extracted {len(texts)} samples from {name}")
    return texts

def main():
    model_name = "deepseek-ai/deepseek-moe-16b-chat"
    # model_name = "deepseek-ai/deepseek-moe-16b-base" # if chat not available
    
    print(f"Loading Model {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            trust_remote_code=True, 
            torch_dtype=torch.float16
        )
        model.config.use_cache = False
    except Exception as e:
        print(f"Model load failed: {e}. Using dummy tensors for structure if allowed, but better to fail.")
        return

    # Tasks from standard LLM benchmarks
    # Winogrande, WSC, PIQA, RTE, RACE, MathQA
    tasks = [
        ('winogrande', 'winogrande_xl', 'validation'),
        # ('piqa', None, 'validation'), # Still failing
        ('super_glue', 'rte', 'validation'),
        ('super_glue', 'wsc', 'validation'),
        ('race', 'high', 'validation'),
        ('math_qa', None, 'validation')
    ]
    
    os.makedirs("data/benchmarks", exist_ok=True)
    
    # Hook to capture input to MoE
    captured_inputs = []
    
    def hook_fn(module, args, output):
        # valid for DeepSeek MoE layer input
        captured_inputs.append(args[0].detach().cpu())

    # Find first MoE block
    target_layer = None
    for n, m in model.named_modules():
        # Look for the characteristic 'experts' module
        if hasattr(m, "experts") and isinstance(m.experts, torch.nn.ModuleList):
            target_layer = m
            print(f"Hooking MoE Layer at: {n}")
            break
            
    if target_layer:
        print(f"Target Layer: {target_layer}")
        handle = target_layer.register_forward_hook(hook_fn)
    else:
        print("Error: No MoE layer found to hook.")
        return

    for t_name, t_conf, t_split in tasks:
        texts = prepare_dataset(t_name, t_conf, t_split, limit=200) # 200 samples per task
        if not texts: continue
        
        captured_inputs.clear()
        
        print(f"Running inference for {t_name}...")
        with torch.no_grad():
            for txt in texts:
                inputs = tokenizer(txt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                model(**inputs)
        
        # Aggregate
        if captured_inputs:
            full_tensor = torch.cat(captured_inputs, dim=1) # [1, TotalSeq, Dim]
            full_tensor = full_tensor.view(-1, full_tensor.shape[-1]) # [TotalSeq, Dim]
            
            # Compute original logits
            logits = None
            if hasattr(target_layer, 'gate') and hasattr(target_layer.gate, 'wg'):
                # DeepSeek uses 'wg' for weight in MoEGate typically or 'weight'
                # Let's try to access the linear weight.
                # Based on previous output, MoEGate structure wasn't fully shown, but often it's 'weight' or 'wg'.
                # Let's try 'weight' first, if fails, catch.
                try:
                     w = target_layer.gate.weight
                     print("Computing original router logits via 'gate.weight'...")
                     logits = torch.nn.functional.linear(full_tensor, w)
                except AttributeError:
                     # Maybe it's 'wg' (DeepSeek V2/V3 uses this, 16B might differ)
                     pass
            
            if logits is None and hasattr(target_layer, 'gate'):
                 # Fallback: Run forward and inspect
                 try:
                     print("Running gate forward for logits...")
                     out = target_layer.gate(full_tensor.unsqueeze(0))
                     if isinstance(out, tuple):
                         # Usually (topk_scores, topk_indices) or (logits, ...)
                         # DeepSeek gate returns (topk_prob, topk_idx) typically?
                         # If so, we can't recover full logits easily.
                         # But let's check one more attribute: 'gate.weight' should exist!
                         # Let's Assume getattr(target_layer.gate, 'weight') failed?
                         pass
                 except Exception as e:
                     print(f"Gate forward failed: {e}")

            # Hard fallback: If we can't get logits, we can't do Naive comparison accurately.
            # But let's try to find the weight parameter by name.
            if logits is None:
                 for name, param in target_layer.gate.named_parameters():
                     if 'weight' in name or 'wg' in name:
                         print(f"Found weight param {name} in gate. Computing logits.")
                         logits = torch.nn.functional.linear(full_tensor.to(param.device), param).cpu()
                         break
            
            save_name = f"{t_name}_{t_conf}" if t_conf else t_name
            save_path = f"data/benchmarks/{save_name}.pt"
            data_dict = {
                'input': full_tensor.unsqueeze(0), # [1, N, D]
                'name': t_name
            }
            if logits is not None:
                data_dict['logits'] = logits.unsqueeze(0)
                
            torch.save(data_dict, save_path)
            print(f"Saved {save_path}: Input {full_tensor.shape}, Logits {logits.shape if logits is not None else 'None'}")
        
    handle.remove()
    print("Benchmark preparation complete.")

if __name__ == "__main__":
    main()
