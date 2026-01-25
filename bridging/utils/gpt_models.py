import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class GptSentenceRefiner:
    """
    Load GPT model for refine sentences with grammar errors.
    Reference: VllmSentenceRefiner in vllm_models.py
    """

    def __init__(self, model_id="gpt-4o-mini", cache_file="gpt_refinement_cache.json"):
        self.client = OpenAI()
        self.model_id = model_id
        self.cache_file = cache_file
        self.refinement_cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.refinement_cache, f, indent=4)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _get_fewshot_example(self):
        fewshot_user_prompt = "it has out things, and you can to each other."  # Grammatically corrupted sentence
        fewshot_assistant_answer = "it has two things, and you can talk to each other."  # Fixed Sentence
        return fewshot_user_prompt, fewshot_assistant_answer

    def make_refine_messages(self, sent):
        fewshot_user_prompt, fewshot_assistant_answer = self._get_fewshot_example()
        system_prompt = (
            "You will receive a perturbed sentence. "
            "This text involves alterations to the original sentence, such as word substitutions, additions, or the introduction of grammatical errors. "
            "Reconstruct the original sentence. "
            "Just give the sentences and don’t provide additional comments."
        )

        user_prompt = sent

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": fewshot_user_prompt},
            {"role": "assistant", "content": fewshot_assistant_answer},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def refine(self, sent):
        if sent in self.refinement_cache:
            return self.refinement_cache[sent]
        
        messages = self.make_refine_messages(sent)

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.3, 
            max_tokens=512,
            top_p=0.95
        )
        
        output = response.choices[0].message.content.strip()
        self.refinement_cache[sent] = output
        self._save_cache()
        return output

    def refine_batch(self, sent_list):
        outputs = []
        is_updated = False
        for sent in sent_list:
            if sent in self.refinement_cache:
                outputs.append(self.refinement_cache[sent])
            else:
                messages = self.make_refine_messages(sent)
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0.3, 
                    max_tokens=512,
                    top_p=0.95
                )
                output = response.choices[0].message.content.strip()
                self.refinement_cache[sent] = output
                outputs.append(output)
                is_updated = True
        
        if is_updated:
            self._save_cache()
        return outputs

if __name__ == "__main__":
    refiner = GptSentenceRefiner()
    # Test examples
    print(refiner.refine("as she and walked away, she heard a noise, but the had'to put its way. home."))
    print(refiner.refine("he was many special george on he nation going to the beach."))
