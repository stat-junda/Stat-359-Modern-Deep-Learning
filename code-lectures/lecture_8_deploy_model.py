from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from transformers import TextStreamer
import sentencepiece
from peft import PeftModel, PeftConfig
from transformers import StoppingCriteria, StoppingCriteriaList

def get_sampler_params(version='v1'):
    if version == 'v1':
        params = {
            'temperature': 0.72,
            'top_p': 0.73,
            'top_k': 30,
            'repetition_penalty': 1.15
        }
        return params
    elif version == 'v2':
        params = {
            'temperature': 0.7,
            'top_p': 0.1,
            'top_k': 40,
            'repetition_penalty': 1.0/0.85
        }
        return params
    elif version == 'v3':
        params = {
                    'temperature': 0.72,
                    'top_p': 0.73,
                    'top_k': 0,
                    'repetition_penalty': 0.73
                }
        return params


class ChatModel:
    def __init__(self, fine_tune_model_id, base_model_path, max_context_length):
        self.max_context_length = max_context_length
        model = LlamaForCausalLM.from_pretrained(base_model_path, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16, use_safetensors=False)
        fine_tuned_model = PeftModel.from_pretrained(model, fine_tune_model_id)
        model = fine_tuned_model
        self.model = model

        tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
        self.tokenizer = tokenizer

        self.sampler = get_sampler_params()
        self.streamer = TextStreamer(self.tokenizer)

    
    def query_model(self, query, max_tokens=250):
        if max_tokens is None:
            max_tokens = self.max_context_length - 100
        
        model_input = self.tokenizer(query, return_tensors='pt').to('cuda')
        self.model.eval()

        with torch.no_grad():
            ret = self.tokenizer.decode(self.model.generate(**model_input, 
                                                            #stopping_criteria= MyStoppingCriteria("User:", prompt=query, tokenizer=self.tokenizer),
                                                            max_new_tokens=max_tokens,
                                                            streamer=self.streamer,
                                                            **self.sampler)[0],
                                                            skip_special_tokens=True)
            print(F"Model return is: \n\n\n\n\n\n{ret}\n\n\n\n\n")
        
        return ret


class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, target_sequence, prompt, tokenizer):
        self.target_sequence = target_sequence
        self.prompt = prompt
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt, '')
        if self.target_sequence in generated_text:
            return True
        
        return False

    def __len__(self):
        return 1
    
    def __iter__(self):
        yield self