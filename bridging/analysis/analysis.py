import json
import numpy as np
from sacrebleu import CHRF
from sentence_transformers import SentenceTransformer
from levenshtein_distance import Levenshtein


class SentenceSimilarity:
    def __init__(self, model_name="all-MiniLM-L6-v2", model=None):
        self.model = model or SentenceTransformer(model_name)
        self._cache = {}

    def _encode_and_cache(self, sentences):
        missing = [s for s in sentences if s not in self._cache]
        if missing:
            embeddings = self.model.encode(
                missing,
                convert_to_numpy=True
            )
            for sent, emb in zip(missing, embeddings):
                self._cache[sent] = emb

    def embedding(self, sentence):
        self._encode_and_cache([sentence])
        return self._cache[sentence]

    def similarity(self, sentence_a, sentence_b):
        self._encode_and_cache([sentence_a, sentence_b])
        emb_a = self._cache[sentence_a]
        emb_b = self._cache[sentence_b]

        res = round(float(np.dot(emb_a,emb_b)), 2)
        return res



if __name__=="__main__":
    file_name = "diffusion_intps_99.json"
    result_dir = "../inference/inference_result"
    
    chrf = CHRF()
    sbert_sim = SentenceSimilarity("all-MiniLM-L6-v2")

    with open(result_dir+"/"+file_name, 'r') as f:
        data = json.load(f)
    
    analysis_result = {
        'chrf_from_x0':[],
        'chrf_from_x1':[],
        'sbert_from_x0':[],
        'sbert_from_x1':[],
        'lev_from_x0':[],
        'lev_from_x1':[],
    }

    for intp_path in data:

        chrf_from_x0 = []
        chrf_from_x1 = []
        sbert_from_x0 = []
        sbert_from_x1 = []
        lev_from_x0 = []
        lev_from_x1 = []

        for sent in intp_path:
            chrf_from_x0.append(chrf.sentence_score(sent, [intp_path[0]]).score)
            chrf_from_x1.append(chrf.sentence_score(sent, [intp_path[-1]]).score)
            sbert_from_x0.append(sbert_sim.similarity(sent, intp_path[0]))
            sbert_from_x1.append(sbert_sim.similarity(sent, intp_path[-1]))
            lev_from_x0.append(Levenshtein(sent, intp_path[0]).distance())
            lev_from_x1.append(Levenshtein(sent, intp_path[-1]).distance())

        analysis_result['chrf_from_x0'].append(chrf_from_x0)
        analysis_result['chrf_from_x1'].append(chrf_from_x1)
        analysis_result['sbert_from_x0'].append(sbert_from_x0)
        analysis_result['sbert_from_x1'].append(sbert_from_x1)
        analysis_result['lev_from_x0'].append(lev_from_x0)
        analysis_result['lev_from_x1'].append(lev_from_x1)

    with open(f"analysis_result/progress_analysis_{file_name[-8:-5]}.json", 'w') as f:
        json.dump(analysis_result, f, indent=3)

        




