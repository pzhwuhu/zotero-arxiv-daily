import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

def rerank_paper(candidate:list[ArxivPaper],corpus:list[dict],model:str='avsolatorio/GIST-small-Embedding-v0') -> list[ArxivPaper]:
    encoder = SentenceTransformer(model)
    #sort corpus by date, from newest to oldest
    corpus = sorted(corpus,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in corpus])
    candidate_feature = encoder.encode([paper.summary for paper in candidate])
    sim = encoder.similarity(candidate_feature,corpus_feature) # [n_candidate, n_corpus]
    scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
    for s,c in zip(scores,candidate):
        c.score = s.item()
    candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    return candidate

def prefilter_paper(candidate:list[ArxivPaper], corpus:list[dict], keep_num:int) -> list[ArxivPaper]:
    if keep_num <= 0 or len(candidate) <= keep_num or len(corpus) == 0:
        return candidate

    corpus = sorted(corpus, key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()

    candidate_text = [f"{paper.title}\n{paper.summary}" for paper in candidate]
    corpus_text = [f"{paper['data']['title']}\n{paper['data']['abstractNote']}" for paper in corpus]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    features = vectorizer.fit_transform(candidate_text + corpus_text)
    candidate_feature = features[:len(candidate)]
    corpus_feature = features[len(candidate):]

    sim = candidate_feature @ corpus_feature.T
    scores = np.asarray(sim @ time_decay_weight).reshape(-1)
    top_indices = np.argsort(-scores)[:keep_num]

    return [candidate[i] for i in top_indices]
