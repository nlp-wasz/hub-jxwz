from typing import Optional, List, Union, Any, Dict,Callable
import os

import faiss
import redis
from torch.nn.functional import embedding

from 作业答案.nlp_study.chenwenyu.rag.backend.rag_test_qwen import questions


class SemanticRouter:
    def __init__(
            self,
            name:str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl:int=3600*24,
            redis_url:str="localhost",
            redis_port:int=6379,
            redis_password:str=None,
            distance_threshold=0.1
    ):
        self.name=name
        self.redis=redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )

        self.ttl=ttl
        self.distance_threshold=distance_threshold
        self.embedding_method=embedding_method
        self.current_question_num=0
        self.index_to_target: Dict[int, str] = {}

        if os.path.exists(f"{self.name}.index"):
            self.index = faiss.read_index(f"{self.name}.index")
        else:
            self.index = None


    def add_route(self,questions: List[str], target: str):
         if not questions:
             raise ValueError("questions 不能为空！")

         embedding=self.embedding_method(questions)

         if self.index is None:
             self.index=faiss.IndexFlatL2(embedding.shape[1])
         self.index.add(embedding)
         faiss.write_index(self.index,f"{self.name}.index")

         start_idx=self.current_question_num
         self.current_question_num+=len(questions)

         for i in range(len(questions)):
             self.index_to_target[start_idx+i]=target





    def route(self, question: str):
        if self.index is None:
            return None
        embedding=self.embedding_method(question)

        dis, ind = self.index.search(embedding, k=100)

        if dis[0][0] > self.distance_threshold:#如果最相似的都超过了阈值，那么就返回None
            return None

        filtered_ind = [i for i, d in enumerate(dis[0]) if d < self.distance_threshold]
        target=[self.index_to_target[i] for i in filtered_ind]
        target=set(target)

        return target


if __name__ == "__main__":
    router = SemanticRouter()
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货"],
        target="refund"
    )

    router("Hi, good morning")