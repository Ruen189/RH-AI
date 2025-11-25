from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", local_files_only=True)
    print("Tokenizer loaded locally!")
except:
    print("Tokenizer NOT found locally")

try:
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", local_files_only=True)
    print("Model loaded locally!")
except Exception as e:
    print("Model NOT found locally:", e)
