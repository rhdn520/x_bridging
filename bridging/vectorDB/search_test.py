import sys
import os
import torch
import numpy as np
import faiss
import textwrap

# 로컬 모듈 import 경로 설정
sys.path.append("..")

from langchain_community.vectorstores import FAISS
from transformers import BertTokenizer
from model import DiffusionLM
from inference.get_latent_path import get_latent_from_sent

# ==========================================
# 1. Embedding Class Definition (DB 로드에 필요)
# ==========================================
class DiffusionEmbeddings:
    def __init__(self, model, tokenizer, device, t=499):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.t = t

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            with torch.no_grad():
                latent = get_latent_from_sent(text, self.model, self.tokenizer, self.device)
                t_tensor = torch.tensor([self.t]).to(self.device)
                latent = self.model.q_sample_no_stochastic(latent, t_tensor)
                latent = latent.squeeze(0).cpu().numpy()
                embeddings.append(latent.flatten())
        return embeddings        

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# ==========================================
# 2. Helper: Load Model (벡터 생성을 위해 필요)
# ==========================================
def load_diffusion_model(model_path, device):
    print(f">> Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = DiffusionLM(
        bert_model_name=config.get('bert_model_name'),
        max_seq_len=config.get('max_seq_len'),
        latent_channels=config.get('latent_channels'),
        latent_width=config.get('latent_width'),
        timesteps=config.get('timesteps'),
        num_diffu_layers=config.get('num_diffu_layers'),
        kernel_size=config.get('kernel_size', 3),
        model_type=config.get('model_type', 'conv'),
        transformer_config=config.get('transformer_config', None),
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(config.get('bert_model_name'))
    return model, tokenizer

# ==========================================
# 3. Main Test Logic
# ==========================================
if __name__ == "__main__":
    # --- 설정 ---
    MODEL_PATH = "../model_outputs/transformer_1024_1_8_1000_d512.pth"
    VECTOR_DB_PATH = "./saved_db/faiss_diffusion_embeddings.index"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 모델 로드 (임베딩 벡터를 만들기 위함)
    model, tokenizer = load_diffusion_model(MODEL_PATH, device)
    
    # 2. 임베더 초기화
    embedder = DiffusionEmbeddings(model, tokenizer, device, t=499)

    # 3. Vector DB 로드
    print(f"\n>> Loading VectorDB from {VECTOR_DB_PATH}...")
    try:
        vector_store = FAISS.load_local(
            folder_path=VECTOR_DB_PATH,
            embeddings=embedder,
            allow_dangerous_deserialization=True
        )
        print(">> VectorDB loaded successfully.")
    except Exception as e:
        print(f">> Error loading DB: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 테스트 시나리오: 벡터 직접 주입하여 검색하기
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" TEST: Searching by Input Vector")
    print("="*50)

    # A. 검색하고 싶은 문장 정의
    query_text = "Once upon a time, a little girl lived in a forest."
    print(f">> Query Text: '{query_text}'")

    # B. 텍스트를 벡터(List[float])로 변환
    # (실제로는 외부에서 가져온 np.array 등 어떤 숫자 배열이든 상관없음)
    query_vector = embedder.embed_query(query_text)
    
    # [검증] 벡터 차원 확인
    print(f">> Generated Vector Dimension: {len(query_vector)}")

    # C. (옵션) 벡터 조작해보기 - "임의의 벡터" 테스트
    # 예: 벡터에 약간의 노이즈를 섞어서 원본과 조금 다른 벡터로 검색 시도
    np_vector = np.array(query_vector)
    noise = np.random.normal(0, 0.5, np_vector.shape) # 노이즈 추가
    noisy_vector = (np_vector + noise).tolist()       # 다시 리스트로 변환
    
    print(">> Applied random noise to the vector to simulate 'arbitrary' input.")

    # D. **핵심**: 벡터로 검색 수행 (similarity_search_by_vector)
    # k=3: 상위 3개 검색
    results = vector_store.similarity_search_with_score_by_vector(
        embedding=noisy_vector, 
        k=3
    )

    # E. 결과 출력
    print("\n>> Search Results (by Vector):")
    for i, (doc, score) in enumerate(results):
        print(f"\n   [Result {i+1}] (L2 Distance: {score:.4f})")
        print(f"   Content: {textwrap.shorten(doc.page_content, width=80, placeholder='...')}")