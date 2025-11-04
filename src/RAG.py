import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class E5Embedding(HuggingFaceEmbedding):
    def _get_query_embedding(self, query: str):
        return super()._get_query_embedding(f"query: {query}")

    def _get_text_embedding(self, text: str):
        return super()._get_text_embedding(f"passage: {text}")


def load_or_create_index(PERSIST_DIR):
    embed_model = E5Embedding(model_name="intfloat/multilingual-e5-base")
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("🔄 Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        print("✨ Creating new index...")
        with open("dataset/transformed_for_llamaindex.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            documents = [Document(text=block["text"], metadata=block.get("metadata", {})) for block in data]
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
        )
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("index created")
    return index


def format_prompt(query, context_str, tokenizer):
    system_msg = '''Jsi nápomocný chatbot Masarykovy univerzity. Tvým úkolem je pomáhat uživatelům orientovat se v Informačním systému (IS MU) a poskytovat rady, jak provést požadované akce v systému.
    Níže jsou oficiální dokumenty nápovědy IS MU, které mohou obsahovat užitečné informace. Pokud informace ve zdrojích nejsou dostatečné, řekni to upřímně.
    Začátek nalezených dokumentů:
    {context_str}
    Konec nalezených dokumentů.
    Moje otázka:'''
    
    messages = [
        {"role": "system", "content": system_msg.format(context_str=context_str)},
        {"role": "user", "content": query}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def query_augment_prompt(query, tokenizer):
    prompt = "Vygeneruj 3 různé varianty následujícího dotazu, které mají stejný význam, ale jinou formulaci. Dotaz pravděpodobně souvisí s universitním informačním systémem, studiem nebo universitou. Ten, kdo se ptá, může být student i učitel. Výsledek vrať výhradně jako JSON seznam řetězců, bez jakéhokoliv formátování kódu (nepoužívej ``` ani žádné značky).\nDotaz: {query}"
    messages = [
        {"role": "user", "content": prompt.format(query=query)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def augment_query(query, tokenizer, pipeline):
    augmented_query = pipeline(
        query_augment_prompt(query, tokenizer),
        max_new_tokens=400,
        do_sample=False,
        top_k=None,
        top_p=None,
        return_full_text=False
    )[0]["generated_text"]
    augmented_query_list = json.loads(augmented_query)
    augmented_query_list.append(query)
    return augmented_query_list


def retrieve_documents(index, list_of_queries):
    unique_retrieved_docs_ids = set()
    unique_retrieverd_nodes = []
    retriever = index.as_retriever(similarity_top_k=3)
    for query in list_of_queries:
        retrieved_nodes = retriever.retrieve(query)
        for node in retrieved_nodes:
            if 'id' in node.metadata and node.metadata["id"] not in unique_retrieved_docs_ids:
                unique_retrieved_docs_ids.add(node.metadata["id"])
                unique_retrieverd_nodes.append(node)
    return unique_retrieverd_nodes


def query_is_muni(query, index, tokenizer, pipeline):
    list_of_queries = augment_query(query, tokenizer, pipeline)
    retrieved_nodes = retrieve_documents(index, list_of_queries)
    context_str = ("\n\n" + "-"*69 + "\n\n").join([n.node.get_content() for n in retrieved_nodes])
    formatted_prompt = format_prompt(query, context_str, tokenizer)
    response = pipeline(
        formatted_prompt,
        max_new_tokens=1024,
        do_sample=False,
        top_k=None,
        top_p=None,
        return_full_text=False
    )[0]["generated_text"]
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_session(timestamp, query, response, context_str, list_of_queries)
    form_link = create_link(timestamp)
    return response, form_link


def create_link(timestamp):
    link = f"https://docs.google.com/forms/d/e/1FAIpQLSfeUp4MTpqv7q_Ww7Z_TvWWMApf_FAFYafQjMId_U5uoV8IUw/viewform?usp=pp_url&entry.1665816165={timestamp}"
    return f'<a href={link} target="_blank"><button>Zpětná vazba k téhle odpovědi</button></a>'


def save_session(timestamp, query, response, context_str, list_of_queries):
    delimiter = "\n\n" + "-" * 69 + "\n\n"
    base = Path(__file__).parent.parent
    session_logs_dir = f"{base}/session_logs"
    Path(session_logs_dir).mkdir(exist_ok=True)
    if len(os.listdir(session_logs_dir)) > 1000:
        return
    file = f"{session_logs_dir}/{timestamp}"
    with open(file, "w") as f:
        f.write(f"Query:\n{query}{delimiter}")
        f.write(f"Response:\n\n{response}{delimiter}")
        queries_string = "\n".join(list_of_queries)
        f.write(f"Augmented questions:\n\n{queries_string}{delimiter}")
        f.write(f"Retrieved documents:{context_str}")


def main(use_cli):
    def _query_is_muni(query):
        return query_is_muni(query, index, tokenizer, my_pipeline)

    PERSIST_DIR = "./dataset/index"
    model_name = "google/gemma-3-4b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    my_pipeline = pipeline("text-generation", model=model_name)
    index = load_or_create_index(PERSIST_DIR)

    if use_cli:
        while True:
            query = input("Zadejte dotaz nebo 'q' pro ukončení: ")
            if query.lower() == 'q':
                break
            response = query_is_muni(query, index, tokenizer, my_pipeline)
            print(response)
    else:
        server = gr.Interface(fn=_query_is_muni,
                              inputs=gr.Textbox(label="Otázka"),
                              outputs=[gr.Textbox(label="ISbot"), gr.HTML()],
                              flagging_mode='never')
        server.launch(server_name='0.0.0.0', server_port=1337, share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', action='store_true', help='uses CLI instead of browser front end')
    args = parser.parse_args()

    main(use_cli=args.cli)
