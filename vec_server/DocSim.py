import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from log_conf import logger
import time


class DocSim(object):
    def __init__(self, model_path, stopwords=[]):
        start = time.time()
        self.w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')
        end = time.time()
        logger.info("[DocSim] Load word2vec model cost: {} s,".format(round(end - start, 3)))

        self.stopwords = stopwords

    def word_vector(self, word):
        try:
            word = word.lower()
            vec = self.w2v_model[word]
            return list(vec)
        except KeyError:
            # pass
            return []

    def most_similar(self, word):
        result = []
        try:
            word = word.lower()
            result = self.w2v_model.most_similar(word)
        except KeyError:
            pass

        return result

    def vectorize(self, doc):
        """Identify the vector values for each word in the given document"""
        doc = doc.lower()
        words = [w for w in doc.split(" ") if w not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                # w, t = word.split('/')
                vec = self.w2v_model[word]
                # print(word+','+','.join(map(lambda x: str(round(x, 4)), list(vec))))
                # vec =  vec * float(t)
                # print(vec)

                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass
        if len(word_vecs) == 0:
            return []

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        vector = np.mean(word_vecs, axis=0)
        # print(vector)
        return vector

    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def _l2_distance(self, vecA, vecB):
        """Find the L2 distance between two vectors."""
        dist = np.linalg.norm(vecA - vecB)
        return dist

    def calculate_similarity(self, source_doc, target_docs=[], threshold=0):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        if isinstance(target_docs, str):
            target_docs = [target_docs]

        source_vec = self.vectorize(source_doc)
        results = []
        for doc in target_docs:
            # target_vec = self.vectorize(doc)
            target_vec = self.vectorize(doc['text'])
            sim_score = self._cosine_sim(source_vec, target_vec)
            # sim_score = self._l2_distance(source_vec, target_vec)
            if sim_score > threshold:
                results.append({
                    'score': sim_score,
                    # 'doc' : doc
                    'id': doc['id']
                })
            # Sort results by score in desc order
            results.sort(key=lambda k: k['score'], reverse=True)

        return results
