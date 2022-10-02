from datasets import load_dataset
from tqdm.auto import tqdm  # progress bar
import pandas as pd
import pinecone
from sentence_transformers import SentenceTransformer  # sentence embedding
from transformers import BartTokenizer, BartForConditionalGeneration
from pprint import pprint


class QAsystem():
    def __init__(self, number_doc) -> None:
        # load bart tokenizer and model from huggingface
        self.number_doc = number_doc
        self.tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
        self.generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa')
        # connect to pinecone environment
        pinecone.init(
            api_key="442d1d81-beba-41f7-8510-f732c8fff44c",  # replace to your api key
            environment="us-west1-gcp"
        )

        index_name = "abstractive-question-answering"
        # check if the abstractive-question-answering index exists
        if index_name not in pinecone.list_indexes():
            # create the index if it does not exist
            pinecone.create_index(
                index_name,
                dimension=768,
                metric="cosine"
            )

        # connect to abstractive-question-answering index we created
        self.index = pinecone.Index(index_name)
        # load the retriever model from huggingface model hub
        self.retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")
        self.load_wiki_dataset()
        self.embedding()

    def load_data(self):

        file1 = open("1.txt", "r")
        passage = []
        lines = file1.readlines()
        for line in lines:
            passage.append(line.rstrip('\n'))

        _total_doc_count = self.number_doc
        _counter = 0
        _docs = []
        # iterate through the dataset and apply our filter
        for d in range(1, _total_doc_count):
            # extract the fields we need
            doc = {
                "passage_text": passage
            }
            # add the dict containing fields we need to docs list
            _docs.append(doc)

            # stop iteration once we reach 50k
            if _counter == _total_doc_count:
                break

            # increase the counter on every iteration
            _counter += 1
        # create a pandas dataframe with the documents we extracted
        self.df = pd.DataFrame(_docs)

    def load_wiki_dataset(self) -> None:
        # load the dataset from huggingface in streaming mode and shuffle it
        _wiki_data = load_dataset(
            'vblagoje/wikipedia_snippets_streamed',
            split='train',
            streaming=True
        ).shuffle(seed=960)

        # filter only documents with History as section_title
        _history = _wiki_data.filter(
            lambda d: d['section_title'].startswith('History')
        )

        _total_doc_count = 50000
        _counter = 0
        _docs = []
        # iterate through the dataset and apply our filter
        for d in tqdm(_history, total=_total_doc_count):
            # extract the fields we need
            doc = {
                "article_title": d["article_title"],
                "section_title": d["section_title"],
                "passage_text": d["passage_text"]
            }
            # add the dict containing fields we need to docs list
            _docs.append(doc)

            # stop iteration once we reach 50k
            if _counter == _total_doc_count:
                break

            # increase the counter on every iteration
            _counter += 1
        # create a pandas dataframe with the documents we extracted
        self.df = pd.DataFrame(_docs)

    def embedding(self) -> None:
        # we will use batches of 64
        batch_size = 64

        for i in tqdm(range(0, len(self.df), batch_size)):
            # find end of batch
            i_end = min(i + batch_size, len(self.df))
            # extract batch
            batch = self.df.iloc[i:i_end]
            # generate embeddings for batch
            emb = self.retriever.encode(batch["passage_text"].tolist()).tolist()
            # get metadata
            meta = batch.to_dict(orient="records")
            # create unique IDs
            ids = [f"{idx}" for idx in range(i, i_end)]
            # add all to upsert list
            to_upsert = list(zip(ids, emb, meta))
            # upsert/insert these records to pinecone
            _ = self.index.upsert(vectors=to_upsert)

        # check that we have all vectors in index
        self.index.describe_index_stats()

    def _query_pinecone(self, query: str, top_k: int):
        # generate embeddings for the query
        xq = self.retriever.encode([query]).tolist()
        # search pinecone index for context passage with the answer
        xc = self.index.query(xq, top_k=top_k, include_metadata=True)
        return xc

    def _format_query(self, query: str, context: int):
        # extract passage_text from Pinecone search result and add the <P> tag
        context = [f"<P> {m['metadata']['passage_text']}" for m in context]
        # concatinate all context passages
        context = " ".join(context)
        # contcatinate the query and context passages
        query = f"question: {query} context: {context}"
        return query

    def generate_answer(self, query: str) -> str:
        result = self._query_pinecone(query, top_k=1)
        query = self._format_query(query, result["matches"])

        for doc in result["matches"]:
            print(doc["metadata"]["passage_text"], end='\n---\n')

        # tokenize the query to get input_ids
        inputs = self.tokenizer([query], max_length=1024, return_tensors="pt", truncation=True)
        # use generator to predict output ids
        ids = self.generator.generate(inputs["input_ids"], num_beams=2, min_length=20, max_length=40)
        # use tokenizer to decode the output ids
        answer = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return answer


def main():
    query = "What is Covid-19"
    system = QAsystem()
    answer = system.generate_answer(query)
    pprint(answer)


if __name__ == "__main__":
    main()
