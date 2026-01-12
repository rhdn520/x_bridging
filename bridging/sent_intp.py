
import os
from dotenv import load_dotenv

# 상위 폴더(..)에 있는 .env 파일을 로드
load_dotenv('../.env', override=True)

print(os.environ.get("HF_HOME"))
# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from transformers import pipeline
import torch
from torch.utils.data import DataLoader
load_dotenv()
from hf_dataset import TinyStoriesDataset
from itertools import combinations
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}", flush=True)

class VllmInterpolator():
    """
    Interpolate two sentences with vllm
    """
    def __init__(self, model_id, tokenizer_id=None):
        import vllm
        self.llm = vllm.LLM(model_id, max_model_len=2048, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.8)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id) if tokenizer_id is None else AutoTokenizer.from_pretrained(tokenizer_id)
        self.sampling_params = vllm.SamplingParams(temperature=0.3, top_p=0.95, top_k=0, max_tokens=512)

    def _generate_output(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)
        return outputs

    def _get_fewshot_example(self):
        user_prompt = (
            """
            Sentence 1: Surprisingly often, animals show up uninvited at sporting events. Sometimes, it gets a little weird.
            Sentence 2: D. Cohen tried to parry attacks on his honesty as Todd Blanche, Donald J. Trump’s lawyer, tried to destroy his credibility with jurors.
            """
        )

        assistant_answer = (
            "Surprisingly often, animals show up uninvited at sporting events. Sometimes, it gets a little weird.\n"
            "Often, unexpected events occur during live events, and sometimes these can get quite weird.\n"
            "At public events, such as sports or trials, unexpected guests often cause disturbances, leading to weird situations.\n"
            "In public spectacles, like sports or courtrooms, unexpected participants can disrupt the normal proceedings in strange ways.\n"
            "During public hearings, like in court, surprising accusations and defenses can emerge, often causing odd disruptions.\n"
            "In courtroom battles, unexpected arguments and witnesses can often turn the proceedings weird as each side tries to undermine the other.\n"
            "In legal battles, lawyers frequently confront each other with surprising tactics to challenge credibility, which can make the proceedings seem strange.\n"
            "Michael D. Cohen, during his trial, encountered surprising tactics by Todd Blanche, Donald J. Trump’s side tries to undermine the other.\n"
            "In legal battles, lawyers frequently confront each other with surprising tactics to challenge credibility, which can make the proceedings seem strange.\n"
            "Michael D. Cohen, during his trial, encountered surprising tactics by Todd Blanche, Donald J. Trump’s"
        )
        return user_prompt, assistant_answer

    def make_interpolate_prompt(self, sent0, sent1):
        fewshot_user_prompt, fewshot_assistant_answer = self._get_fewshot_example()

        system_prompt = (
            "You are a writer who is good at writing sentences.\n" 
            "You will be given two sentences." 
            "You should gradually change the first sentence to make it exactly the same as the second sentence.\n" 
            "Just give the sentences and don’t provide additional comments. Sentences should be separated by a line break.\n"
        )

        user_prompt = (
            "Sentence 1: " + sent0 + "\n"
            "Sentence 2: " + sent1
        )

        messages = [
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": fewshot_user_prompt},
            {"role":"assistant", "content": fewshot_assistant_answer},
            {"role":"user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return text

    def interpolate(self, sent0, sent1):
        text = self.make_interpolate_prompt(sent0, sent1)

        outputs = self._generate_output([text])
        output = outputs[0].outputs[0].text
        # print(inspect.getmembers(outputs))

        # output = outputs['text']
        # print(outputs)
        output = output.replace("assistant","").strip()
        output = sent0.strip() + "\n" + output + "\n" + sent1.strip()

        return output
    
    def interpolate_batch(self, sent_list):
        #iterate combination of sent_list_0 and sent_list_1
        texts = []
        for sent0, sent1 in combinations(sent_list, 2):
            # if(sent0 != sent1): 
            texts.append(self.make_interpolate_prompt(sent0, sent1))

        #texts가 너무 크면 나눠서 처리
        batch_size = 10000
        all_outputs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_outputs = self._generate_output(batch_texts)
            all_outputs.extend(batch_outputs)
        
        outputs = all_outputs
        outputs = [output.outputs[0].text for output in outputs]
        outputs = [output.replace("assistant","").strip() for output in outputs]
        outputs = [sent0.strip() + "\n" + output + "\n" + sent1.strip() for output, (sent0, sent1) in zip(outputs, combinations(sent_list, 2))]
        return outputs
        


if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    tokenizer_id = "meta-llama/Llama-3.3-70B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tsDataset = TinyStoriesDataset(tokenizer, split="train", dataset_size=2000)
    print(f"Loaded TinyStories dataset with {len(tsDataset)} samples.")

    sents = []

    tsDataLoader = DataLoader(tsDataset, batch_size=100, shuffle=True)

    for batch in tsDataLoader:
        sents.extend(batch['text'])

    vllm_interpolater = VllmInterpolator(model_id, tokenizer_id)
    outputs = vllm_interpolater.interpolate_batch(sents)
    print(outputs)        

    with open("vllm_interpolation_outputs.json", "w") as f:
        import json
        json.dump(outputs, f, indent=4)