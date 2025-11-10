
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class SimpleBM25Retriever:
    """
    A simple wrapper class for LangChain's BM25Retriever.

    This class allows you to easily initialize a retriever from a list of
    LangChain Document objects and search for relevant documents using a query.
    """
    def __init__(self, docs: List[str]):
        """
        Initializes the SimpleBM25Retriever.

        Args:
            docs (List[Document]): A list of LangChain Document objects to build
                                   the retriever from.
        """

        langchain_docs = [Document(page_content=x) for x in docs]
        if not langchain_docs or not all(isinstance(doc, Document) for doc in langchain_docs):
            raise ValueError("The 'docs' argument must be a non-empty list of Document objects.")

        print("Building the BM25 retriever from the provided documents...")
        self.retriever = BM25Retriever.from_documents(langchain_docs)
        print("Retriever built successfully.")

    def search(self, query: str, top_k: int = 1) -> List[Document]:
        """
        Searches for documents relevant to the given query.

        Args:
            query (str): The search query string.
            top_k (int, optional): The number of top documents to return.
                                   Defaults to 5.

        Returns:
            List[Document]: A list of the most relevant Document objects.
        """
        if not query:
            return []
        
        # print(f"\nSearching for top {top_k} documents with query: '{query}'")
        self.retriever.k = top_k
        relevant_docs = self.retriever.invoke(query)
        return relevant_docs

def main():
    """
    An example of how to use the SimpleBM25Retriever class.
    """
    # 1. Create a list of sample documents
    sample_docs = [
        'hello',
        'what the fuck?',
        'heyhey',
        'pleaseeeeee',
        'the sky is blue'
    ]

    # 2. Initialize the retriever with the documents
    try:
        bm25_search = SimpleBM25Retriever(docs=sample_docs)
    except ValueError as e:
        print(f"Error initializing retriever: {e}")
        return

    # 3. Perform a search
    query1 = "what color is the sky?"
    results1 = bm25_search.search(query1, top_k=2)

    print("\n--- Search Results ---")
    for i, doc in enumerate(results1):
        print(f"Result {i+1}:")
        print(f"  Content: {doc.page_content}")
        print(f"  Metadata: {doc.metadata}")

    # 4. Perform another search with a different query and top_k
    query2 = "information about programming"
    results2 = bm25_search.search(query2, top_k=1)

    print("\n--- Search Results ---")
    for i, doc in enumerate(results2):
        print(f"Result {i+1}:")
        print(f"  Content: {doc.page_content}")
        print(f"  Metadata: {doc.metadata}")

if __name__ == "__main__":
    main()