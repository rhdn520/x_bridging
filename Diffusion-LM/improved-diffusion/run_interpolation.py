import argparse
import os
import json
import torch as th
import numpy as np

# 필요한 custom 모듈 import
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
# [변경점 1] 프로젝트의 커스텀 토크나이저 로더를 사용합니다.
from improved_diffusion.rounding import load_tokenizer
from improved_diffusion.resample import create_named_schedule_sampler
# from transformers import AutoTokenizer # 더 이상 사용하지 않습니다.

class Interpolator:
    """
    모델 로딩, 문장 보간, 역전파를 수행하는 핵심 로gic을 담은 클래스
    """
    def __init__(self, args, model, diffusion):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        
        # [변경점 2] 'text_sample.py'와 동일한 방식으로 모델에 맞는 커스텀 토크나이저를 로드합니다.
        # 이 토크나이저는 단순한 단어-ID 매핑 딕셔너리일 수 있습니다.
        logger.log("loading tokenizer...")
        self.tokenizer = load_tokenizer(
            args.modality,
            args.experiment,
            os.path.split(args.model_path)[0]
        )
        # UNK (Unknown) 토큰의 ID를 찾아둡니다. 없을 경우를 대비해 기본값(e.g., 0)을 설정할 수 있습니다.
        self.unk_token_id = self.tokenizer.get('[UNK]', 0) 

        self.schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    def _preprocess_sentences(self, sentence1, sentence2):
        """
        두 개의 문자열 문장을 모델 입력에 맞는 텐서 형태로 변환합니다.
        (토큰화 -> 임베딩 -> 패딩)
        """
        seq_len = self.args.image_size ** 2 # 이 프로젝트는 시퀀스 길이를 image_size^2로 다룹니다.

        # [변경점 3] 커스텀 토크나이저(딕셔너리)를 사용하여 문장을 토큰 ID 리스트로 변환합니다.
        def sentence_to_ids(sentence):
            words = sentence.lower().strip().split()
            # 딕셔너리에서 단어에 해당하는 ID를 찾고, 없는 단어는 UNK 토큰 ID로 처리합니다.
            return [self.tokenizer.get(word, self.unk_token_id) for word in words]

        tokens1 = sentence_to_ids(sentence1)
        tokens2 = sentence_to_ids(sentence2)

        # 문장 길이를 args.image_size (시퀀스 길이)에 맞게 패딩/자르기
        def pad_or_truncate(tokens, length):
            if len(tokens) < length:
                # pad 토큰 ID가 0이라고 가정합니다.
                return tokens + [0] * (length - len(tokens))
            return tokens[:length]

        tokens1 = pad_or_truncate(tokens1, seq_len)
        tokens2 = pad_or_truncate(tokens2, seq_len)

        # 토큰 ID 텐서 생성 (batch_size=1)
        input_ids1 = th.tensor([tokens1], dtype=th.long).to(dist_util.dev())
        input_ids2 = th.tensor([tokens2], dtype=th.long).to(dist_util.dev())

        # 모델의 임베딩 레이어를 사용해 x_0 (초기 임베딩) 생성
        # 이 과정에서 CUDA 에러가 발생했었지만, 이제 ID가 범위 내에 있으므로 정상 작동합니다.
        with th.no_grad():
            embedding_layer = self.model.word_embedding
            x_start1 = embedding_layer(input_ids1)
            x_start2 = embedding_layer(input_ids2)

        # cond (model_kwargs) 생성 - attention_mask 등
        # 패딩된 부분은 0, 아닌 부분은 1
        mask1 = (input_ids1 != 0).to(dist_util.dev())
        mask2 = (input_ids2 != 0).to(dist_util.dev())
        
        # 모델은 attention_mask만 필요로 할 수 있습니다. text_sample.py를 참고하여 단순화합니다.
        cond1 = {"attention_mask": mask1}
        cond2 = {"attention_mask": mask2}
        
        return x_start1, cond1, x_start2, cond2

# run_interpolation_and_backward 함수를 아래와 같이 수정하세요.

    def run_interpolation_and_backward(self, sentence1, sentence2, alpha=0.5):
        """
        메인 로직: 문장 전처리, 노이즈 추가, 보간, 생성, forward/backward pass 수행
        """
        print("1. 두 문장을 모델 입력 텐서로 전처리 중...")
        x_start1, cond1, x_start2, cond2 = self._preprocess_sentences(sentence1, sentence2)

        print("2. 노이즈 추가 및 보간 수행 중...")
        t, weights = self.schedule_sampler.sample(x_start1.shape[0], dist_util.dev())
        noise1 = th.randn_like(x_start1)
        noise2 = th.randn_like(x_start2)
        xt_1 = self.diffusion.q_sample(x_start=x_start1, t=t, noise=noise1)
        xt_2 = self.diffusion.q_sample(x_start=x_start2, t=t, noise=noise2)
        xt_interp = (1 - alpha) * xt_1 + alpha * xt_2
        noise_interp_target = (1 - alpha) * noise1 + alpha * noise2
        cond_interp = cond1

        # =================== [ ✅ 기능 추가 부분 ] ===================
        # 모델을 평가 모드로 잠시 변경하여 문장 생성
        self.model.eval() 
        with th.no_grad():
            generated_sentence = self.denoise_and_decode(xt_interp.detach(), cond_interp)
        self.model.train() # 다시 학습 모드로 복귀

        print("="*50)
        print(f"Noise Interpolation을 통해 생성된 문장:")
        print(generated_sentence)
        print("="*50)
        # ==========================================================

        print("3. 보간된 노이즈로 Forward Pass 및 Backward Pass 수행 중...")
        self.model.zero_grad()
        
        losses = self.diffusion.training_losses(
            model=self.model,
            x_start=x_start1, 
            t=t,
            model_kwargs=cond_interp,
        )
        loss = (losses["loss"] * weights).mean()
        loss.backward()

        print("="*40)
        print(f"최종 계산된 Loss: {loss.item()}")
        grad_norm = th.linalg.norm(next(self.model.parameters()).grad).item()
        print(f"모델의 첫 번째 파라미터 Gradient Norm: {grad_norm}")
        print("="*40)


    def denoise_and_decode(self, noise_tensor, cond):
        """
        주어진 노이즈 텐서로부터 전체 샘플링 루프를 거쳐 문장을 생성합니다.
        """
        logger.log("starting denoising process to generate sentence...")
        
        # p_sample_loop를 사용하여 노이즈를 점진적으로 제거합니다.
        # 이 함수는 최종적으로 예측된 x_0 (노이즈가 제거된 임베딩) 텐서를 반환합니다.
        sample_emb = self.diffusion.p_sample_loop(
            self.model,
            noise_tensor.shape,
            noise=noise_tensor, # 시작점으로 보간된 노이즈를 제공
            clip_denoised=self.args.clip_denoised,
            model_kwargs=cond,
        )

        logger.log("decoding the generated embeddings...")
        
        # 임베딩에서 로짓(단어 확률)을 계산합니다.
        logits = self.model.get_logits(sample_emb)
        
        # 각 위치에서 가장 확률이 높은 단어의 인덱스를 선택합니다 (greedy decoding).
        top_indices = th.topk(logits, k=1, dim=-1).indices.squeeze(-1)

        # ID를 다시 단어로 변환하기 위해 역방향 토크나이저(ID -> 단어)를 만듭니다.
        rev_tokenizer = {v: k for k, v in self.tokenizer.items()}
        
        # 배치 내 첫 번째 문장을 디코딩합니다.
        decoded_words = []
        for token_id in top_indices[0].tolist():
            word = rev_tokenizer.get(token_id, '[UNK]')
            if word == '[PAD]' or token_id == 0: # 패딩 토큰은 건너뜁니다.
                continue
            decoded_words.append(word)
        
        return " ".join(decoded_words)

def create_argparser():
    """
    스크립트 실행에 필요한 인자들을 설정합니다.
    """
    defaults = model_and_diffusion_defaults()
    text_defaults = dict(
        modality='text',
        schedule_sampler="uniform",
        model_arch='transformer',
        experiment='gpt2_pre_compress' 
    )
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pt file)")
    parser.add_argument("--sentence1", type=str, required=True, help="First sentence for interpolation")
    parser.add_argument("--sentence2", type=str, required=True, help="Second sentence for interpolation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing (default: 1)")
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    # set_seed(101) # 필요시 시드 고정
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    
    # batch_size는 interpolation 스크립트의 인자를 따르도록 유지
    current_batch_size = args.batch_size
    training_args['batch_size'] = current_batch_size
    args.__dict__.update(training_args)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.train() # backward pass를 위해 train 모드로 설정

    interpolator = Interpolator(args, model, diffusion)
    interpolator.run_interpolation_and_backward(args.sentence1, args.sentence2)
    
    logger.log("interpolation and backward pass complete.")

if __name__ == "__main__":
    main()