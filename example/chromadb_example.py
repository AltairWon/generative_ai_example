import chromadb
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient()
answers = client.create_collection(
    name="sample_answers"
)

filename = '/Users/whj121/Desktop/sample.xlsx'

df = pd.read_excel(filename)
df.sample(5)

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

ids = []
metadatas = []
embeddings = []

for row in tqdm(df.iterrows()):
    index = row[0]
    query = row[1].user
    answer = row[1].answer
    
    metadata = {
        "query": query,
        "answer": answer
    }
    
    embedding = model.encode(query, normalize_embeddings=True)
    
    ids.append(str(index))
    metadatas.append(metadata)
    embeddings.append(embedding)
    
chunk_size = 1024  # 한 번에 처리할 chunk 크기 설정
total_chunks = len(embeddings) // chunk_size + 1  # 전체 데이터를 chunk 단위로 나눈 횟수
embeddings = [ e.tolist() for e in tqdm(embeddings)]  

for chunk_idx in tqdm(range(total_chunks)):
    start_idx = chunk_idx * chunk_size
    end_idx = (chunk_idx + 1) * chunk_size
    
    # chunk 단위로 데이터 자르기
    chunk_embeddings = embeddings[start_idx:end_idx]
    chunk_ids = ids[start_idx:end_idx]
    chunk_metadatas = metadatas[start_idx:end_idx]
    
    # chunk를 answers에 추가
    answers.add(embeddings=chunk_embeddings, ids=chunk_ids, metadatas=chunk_metadatas)

    result = answers.query(
    query_embeddings=model.encode("한총TF 최신 파일을 어디있어?", normalize_embeddings=True).tolist(),
    n_results=3
)


print(result)