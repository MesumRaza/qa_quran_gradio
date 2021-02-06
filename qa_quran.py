from haystack import Finder
from haystack.preprocessor.utils import convert_files_to_dicts
from haystack.reader.farm import FARMReader
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
from os import path
import gradio

def read_corpus():
    document_store = InMemoryDocumentStore()
    doc_dir = "Quran"
    dicts = convert_files_to_dicts(dir_path=doc_dir, split_paragraphs=True)
    document_store.write_documents(dicts)
    return document_store

def retriever():
    document_store = read_corpus()
    retriever = TfidfRetriever(document_store=document_store)
    return retriever

retriever = retriever()

if not(path.exists('data/mlm-temp')):
    reader = FARMReader(model_name_or_path="deepset/minilm-uncased-squad2", use_gpu=False)
    reader.save(directory='data/mlm-temp')
else:
    reader = FARMReader(model_name_or_path="data/mlm-temp", use_gpu=False)
    
finder = Finder(reader, retriever)

def ask_question(inp):
    prediction = finder.get_answers(question=inp, top_k_retriever=10, top_k_reader=5)
    return prediction

gradio.Interface(
    fn=ask_question
    ,inputs="textbox"
    , outputs="json"
    ,layout="vertical"
    ,title="ASK A QUESTION FROM HOLY QURAN IN ENGLISH"
    ,examples=[['Who is Muhammad?'],['Who are Gog and Magog?'],['Who is Satan?']]
).launch()
