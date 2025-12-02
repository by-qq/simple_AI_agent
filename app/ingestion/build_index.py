from app.deps import get_vs
from app.ingestion.loader import split_docs, load_docs


def main():
    docs = split_docs(load_docs("./data/docs"))
    vs = get_vs()
    vs.add_documents(docs)
    # try:
    #     # 持久化，理论上是要报错的chroma没有attribute 属性 persist
    #     # 新版本默认持久化
    #     vs.persist()
    # except Exception as e:
    #     pass

    print(f"Indexed {len(docs)} chunks into Chroma")


if __name__ == "__main__":
    main()


