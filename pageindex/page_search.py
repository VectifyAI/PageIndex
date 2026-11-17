"""Page Search: the pages, and the documents, that help answer a question."""
import time
from typing import Any, Optional, Union

from .client import PageIndexClient, _cloud_client
from .errors import PageIndexAPIError


class PageSearchClient:
    """
    Page Search over your PageIndex cloud documents. Cloud-only.

        client = PageSearchClient()   # reads PAGEINDEX_API_KEY
        doc = client.upload("report.pdf")
        client.search("What was revenue in 2024?", document=doc["doc_id"])

    ``upload()`` stores a document's pages and builds no outline index. The
    documents are the ones ``PageIndexClient`` and ``PageExtractClient`` see
    with the same key; ``client.pageindex`` is that ``PageIndexClient``, for
    listing and deleting them, collections and metadata.

    Args:
        api_key (str, optional): Your PageIndex API key. Defaults to the
            PAGEINDEX_API_KEY environment variable, then to
            PAGEINDEX_SEARCH_API_KEY, where Page Search's SDK read its key.
        client (PageIndexClient, optional): A cloud ``PageIndexClient`` to
            work through, in place of ``api_key``.
    """

    def __init__(self, api_key: Optional[str] = None, *,
                 client: Optional[PageIndexClient] = None):
        self.pageindex = _cloud_client(api_key, client, "PageSearchClient",
                                       "PAGEINDEX_SEARCH_API_KEY")

    def upload(
        self,
        file_path: str,
        *,
        collection: Optional[str] = None,
        metadata: Optional[dict] = None,
        wait: bool = True,
        timeout: float = 1800.0,
    ) -> dict[str, Any]:
        """
        Upload a PDF to search. Its pages are stored and no outline index is
        built, so the upload is charged less than one that builds it;
        ``PageIndexClient.build_outline_index()`` builds one later. Uploading
        the same file to the same place again returns the stored document.

        Args:
            file_path (str): Path to the PDF file.
            collection (str, optional): The ID of the collection to store it
                in.
            metadata (dict, optional): Your own tags for the document, as
                ``PageIndexClient.submit_document()`` takes them.
            wait (bool): Return once its pages can be searched (default), up
                to ``timeout`` seconds; raises PageIndexAPIError if its
                processing failed or is not done in time. False returns at
                once.

        Returns:
            dict: {'doc_id': ..., 'name': ...}.
        """
        return self.pageindex._upload(
            file_path, wait, timeout, metadata=metadata,
            collection_id=collection, index=False)

    def search(
        self,
        questions: Union[str, list[str]],
        *,
        document: Optional[str] = None,
        documents: Optional[list[str]] = None,
        collection: Optional[str] = None,
        folder: Optional[str] = None,
        recursive: bool = False,
        filter: Optional[dict[str, Any]] = None,
        max_documents: Optional[int] = None,
        document_top_k: Optional[int] = None,
        top_k: Optional[int] = None,
        include_text: bool = False,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Ask up to 10 questions. For each, the pages that help answer it,
        under the documents they are in.

            client.search(questions, document=doc_id)         # one document's pages
            client.search(questions, collection=collection_id)  # across documents
            client.search(questions)                          # across all of yours

        Across documents, the most relevant documents for each question are
        found first, as ``search_documents()`` finds them, and only their
        pages are searched: the documents likely to answer, at most
        ``max_documents``, or exactly ``document_top_k``. Every page searched
        is charged, and across documents every document judged too.

        Args:
            questions (str | list[str]): One question, or up to 10.
            document (str, optional): The document whose pages to search.
                Raises PageIndexAPIError (HTTP 409) while it is still
                processing.
            documents (list[str], optional): Search among these documents.
            collection (str, optional): Search among a collection's documents,
                by its ID.
            folder (str, optional): Search among a folder's documents, by its
                ID; "root" is the library root.
            recursive (bool): With ``folder``, the documents of its
                descendant folders too.
            filter (dict, optional): Search among the documents whose metadata
                meets it, e.g. ``{"year": {"$gte": 2023}}``.
            max_documents (int, optional): Across documents, the most a
                question's pages are searched in, 1 to 10. Defaults to 3.
            document_top_k (int, optional): Across documents, search exactly
                this many for each question, 1 to 10, in place of
                ``max_documents``.
            top_k (int, optional): List exactly this many pages of each
                document. Without it, each lists the pages likely to help,
                possibly none, and the next most relevant under 'more_pages'.
            include_text (bool): Give each listed page's text.
            start_page (int, optional): With ``document``, the first page to
                search, from 1.
            end_page (int, optional): With ``document``, the last page to
                search, included.

        Returns:
            dict: {'id', 'results', 'failed_documents',
            'processing_documents', 'usage', 'seconds'}. 'results' has one
            {'question', 'documents'} for each question, in the order asked.
            Each document is {'id', 'name', 'pageNum', 'folderId',
            'collectionId', 'metadata', 'relevance', 'pages', 'more_pages',
            'failed_pages'}, most relevant first; 'relevance' is 0 to 1, and
            None for the one you named with ``document``. Each page is
            {'page', 'relevance', 'read'} ('read' is "text" or "image": how
            the page was read), with 'text' under ``include_text``.
            'failed_pages' are the pages that could not be judged, and
            'failed_documents' the documents that could not be judged or
            searched: none of these is charged. 'processing_documents' were
            not ready, and not searched. 'usage' is {'documents',
            'billable_documents', 'pages', 'billable_pages', 'questions',
            'credits', 'balance'}: what the search cost and what is left.
        """
        api = self.pageindex._api
        questions = [questions] if isinstance(questions, str) else list(questions)
        across = recursive or any(value is not None for value in (
            documents, collection, folder, filter, max_documents, document_top_k))

        def listed(item: dict) -> dict:
            return {key: item[key] for key in ("pages", "more_pages") if key in item}

        if document is not None:
            if across:
                raise PageIndexAPIError(
                    "search takes document alone, or documents, collection, "
                    "folder, recursive, filter, max_documents and "
                    "document_top_k.")
            info = api.get_document(document)
            entry = {key: info.get(key) for key in (
                "id", "name", "pageNum", "folderId", "collectionId", "metadata")}
            found = api.search_pages(
                document, questions, top_k=top_k, include_text=include_text,
                start_page=start_page, end_page=end_page)
            return {
                "id": found["id"],
                "results": [
                    {"question": item["question"], "documents": [
                        {**entry, "relevance": None, **listed(item),
                         "failed_pages": found["failed_pages"]}]}
                    for item in found["results"]],
                "failed_documents": [],
                "processing_documents": [],
                "usage": {"documents": 0, "billable_documents": 0, **found["usage"]},
                "seconds": found["seconds"],
            }
        if start_page is not None or end_page is not None:
            raise PageIndexAPIError(
                "start_page and end_page are for one document: pass document.")
        if max_documents is not None and document_top_k is not None:
            raise PageIndexAPIError(
                "Pass max_documents or document_top_k, not both.")
        most = next((n for n in (document_top_k, max_documents) if n is not None), 3)
        if not 1 <= most <= 10:
            raise PageIndexAPIError(
                "max_documents and document_top_k are from 1 to 10.")

        started = time.monotonic()
        judged = api.search_documents(
            questions, document_ids=documents, collection_id=collection,
            folder_id=folder, recursive=recursive, metadata_filter=filter,
            top_k=document_top_k)
        chosen = [item["documents"][:most] for item in judged["results"]]
        asked: dict[str, list[int]] = {}
        for k, found in enumerate(chosen):
            for doc in found:
                asked.setdefault(doc["id"], []).append(k)

        def search_pages(doc_id: str):
            try:
                return api.search_pages(
                    doc_id, [questions[k] for k in asked[doc_id]],
                    top_k=top_k, include_text=include_text)
            except PageIndexAPIError as exc:
                if exc.status_code == 402:
                    raise
                return exc

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as pool:
            opened = dict(zip(asked, pool.map(search_pages, asked)))
        done = [s for s in opened.values() if isinstance(s, dict)]
        errors = [s for s in opened.values() if isinstance(s, PageIndexAPIError)]
        if errors and not done:
            raise errors[0]
        results = []
        for k, question in enumerate(questions):
            shown = []
            for doc in chosen[k]:
                searched = opened[doc["id"]]
                if isinstance(searched, dict):
                    item = searched["results"][asked[doc["id"]].index(k)]
                    shown.append({**doc, **listed(item),
                                  "failed_pages": searched["failed_pages"]})
            results.append({"question": question, "documents": shown})
        charged = [judged["usage"], *(s["usage"] for s in done)]
        balances = [u["balance"] for u in charged if u.get("balance") is not None]
        return {
            "id": judged["id"],
            "results": results,
            "failed_documents": judged["failed_documents"] + [
                doc_id for doc_id, s in opened.items() if not isinstance(s, dict)],
            "processing_documents": judged["processing_documents"],
            "usage": {
                "documents": judged["usage"]["documents"],
                "billable_documents": judged["usage"]["billable_documents"],
                "pages": sum(s["usage"]["pages"] for s in done),
                "billable_pages": sum(s["usage"]["billable_pages"] for s in done),
                "questions": len(questions),
                "credits": round(sum(u["credits"] for u in charged), 6),
                "balance": min(balances) if balances else None,
            },
            "seconds": round(time.monotonic() - started, 2),
        }

    def search_documents(
        self,
        questions: Union[str, list[str]],
        *,
        document_ids: Optional[list[str]] = None,
        collection: Optional[str] = None,
        folder: Optional[str] = None,
        recursive: bool = False,
        filter: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Ask up to 10 questions of your documents. For each, the documents
        likely to hold the answer, most relevant first; ``search()`` then
        finds the pages. Every document judged is charged.

        Args:
            questions (str | list[str]): One question, or up to 10.
            document_ids (list[str], optional): Judge only these documents.
                Without it, every document of yours in scope, up to 1,000;
                more raises PageIndexAPIError (HTTP 400).
            collection (str, optional): Judge a collection's documents, by
                its ID.
            folder (str, optional): Judge a folder's documents, by its ID;
                "root" is the library root.
            recursive (bool): With ``folder``, the documents of its
                descendant folders too.
            filter (dict, optional): Judge the documents whose metadata
                meets it, e.g. ``{"year": {"$gte": 2023}}``.
            top_k (int, optional): List exactly this many documents for each
                question. Without it, each lists the documents likely to
                help, possibly none, and the next most relevant under
                'more_documents'.

        Returns:
            dict: {'id', 'results', 'failed_documents',
            'processing_documents', 'usage', 'seconds'}. 'results' has one
            {'question', 'documents', 'more_documents'} for each question, in
            the order asked; each document is {'id', 'name', 'pageNum',
            'folderId', 'collectionId', 'metadata', 'relevance'}, with
            'relevance' from 0 to 1. 'failed_documents' could not be judged
            and are not charged; 'processing_documents' were not ready, and
            not judged. 'usage' is {'documents', 'questions',
            'billable_documents', 'credits', 'balance'}.
        """
        return self.pageindex._api.search_documents(
            [questions] if isinstance(questions, str) else list(questions),
            document_ids=document_ids, collection_id=collection,
            folder_id=folder, recursive=recursive, metadata_filter=filter,
            top_k=top_k)
