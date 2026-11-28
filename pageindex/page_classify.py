"""Page Classify: the type of a PDF, or of each of its pages, among the classes
you name."""
import os
import time
from typing import Any, Iterable, Mapping, Optional, Union

from .client import PageIndexClient, _cloud_client
from .errors import PageIndexAPIError
from .page_extract import PageExtractClient

COLLECTION = "Page Classify"

Classes = Union[Mapping[str, Optional[str]], Iterable[Union[str, Mapping[str, Any]]]]


def _classes(classes: Classes) -> list[dict]:
    """The API's classes: [{"name", "description"}]."""
    if isinstance(classes, str):
        classes = [classes]
    if isinstance(classes, Mapping):
        return [{"name": name, **({"description": about} if about else {})}
                for name, about in classes.items()]
    return [dict(c) if isinstance(c, Mapping) else {"name": c} for c in classes]


class PageClassifyClient:
    """
    Page Classify over your PageIndex cloud documents. Cloud-only.

        client = PageClassifyClient()   # reads PAGEINDEX_API_KEY
        job = client.classify("invoice.pdf", {
            "invoice": "A bill asking for payment",
            "receipt": "Proof of a purchase already paid",
        })
        job["class"]

    A PDF given by its path is uploaded to the "Page Classify" collection,
    with no outline index built. The documents are the ones
    ``PageIndexClient``, ``PageSearchClient`` and ``PageExtractClient`` see
    with the same key; ``client.documents`` is that ``PageIndexClient``, for
    listing and deleting them.

    Args:
        api_key (str, optional): Your PageIndex API key. Defaults to the
            PAGEINDEX_API_KEY environment variable.
        client (PageIndexClient, optional): A cloud ``PageIndexClient`` to
            work through, in place of ``api_key``.
    """

    def __init__(self, api_key: Optional[str] = None, *,
                 client: Optional[PageIndexClient] = None):
        self.documents = _cloud_client(api_key, client, "PageClassifyClient", None)
        self._collections: dict[str, str] = {}

    # Finds or makes the collection a file is kept in, as Page Extract does.
    _collection = PageExtractClient._collection

    def classify(
        self,
        document: Union[str, os.PathLike],
        classes: Classes,
        *,
        granularity: str = "document",
        pages: Union[str, int, tuple[int, int], None] = None,
        wait: bool = True,
        timeout: float = 1800.0,
        collection: Optional[str] = COLLECTION,
    ) -> dict[str, Any]:
        """
        Classify a PDF, or each of its pages, among ``classes``.

            job = client.classify("packet.pdf", classes, granularity="page")
            [(p["page"], p["class"]) for p in job["pages"]]

        Args:
            document (str): A PDF's path: the file is uploaded to
                ``collection`` with the job, with no outline index built, and
                kept as a document from then on. Only the classification is
                charged, once the job completes; a job that fails is not
                charged, and leaves a file it uploaded as a failed document,
                with its name and file but no pages, until you delete it.
                Uploading the same file again uses the stored document. Or
                the ``doc_id`` of an uploaded document: one still processing
                is waited for first, and with ``wait=False`` raises
                PageIndexAPIError (HTTP 409).
            classes (dict | list | str): The classes, by name: {name:
                description} (a description may be None), a list of names,
                or one name. 1 to 50; names and descriptions hold at most
                8,000 characters, all told.
            granularity (str): "document" (default): one class for the
                document, from its first 5 pages unless ``pages`` names
                others. "page": a class for each page, every page unless
                ``pages`` names fewer.
            pages (str | int | tuple, optional): Read these pages only:
                "3-9", (3, 9) or 3.
            wait (bool): Return the job once it is completed (default), up to
                ``timeout`` seconds; raises PageIndexAPIError if it failed,
                could not read some of its pages, or is not done in time (it
                goes on running). Read such a job with ``get_classify()``.
                False returns it pending once it starts: ``wait_classify()``
                waits for it later.
            timeout (float): Seconds to wait, the document's upload and
                processing included. Defaults to 30 minutes.
            collection (str, optional): Where a file is uploaded: a
                collection's name (any case) or ID, "Page Classify" by
                default. A name none of your collections has makes one.
                None stores the document outside any collection.

        Returns:
            dict: The job: {'id', 'status', 'doc_id', 'configuration',
            'created_at', 'updated_at'}. 'status' is "pending", "running",
            "completed" or "failed". Completed, it also has 'class' and
            'confidence' ("document"), or 'pages', each {'page', 'class',
            'confidence'} ("page"); a class is None for a document or page of
            none of the classes. And 'failed_pages' and 'usage' ({'pages',
            'credits'}); failed, 'error_message'. Classifying costs $0.001 a
            page read.
        """
        api = self.documents._api
        configuration: dict[str, Any] = {
            "classes": _classes(classes), "granularity": granularity}
        if isinstance(pages, tuple):
            pages = f"{pages[0]}-{pages[1]}"
        if pages is not None:
            configuration["target_pages"] = str(pages)
        document = os.fspath(document)
        deadline = time.monotonic() + timeout
        if os.path.exists(document):
            for last in (False, True):
                job = self._send(document, collection, configuration)
                if "id" in job:
                    return self._done(job, deadline, timeout) if wait else job
                # The file is stored already, and its document still processing.
                try:
                    self.documents._wait_until_ready(
                        job["doc_id"], deadline - time.monotonic())
                    break
                except PageIndexAPIError as exc:
                    # It failed or was deleted meanwhile: the file is sent again.
                    if last or not (
                            exc.status_code == 404
                            or self.documents.get_document(
                                job["doc_id"]).get("status") == "failed"):
                        raise
            document = job["doc_id"]
        elif document.lower().endswith(".pdf"):
            raise FileNotFoundError(document)
        try:
            job = api.create_classify(configuration, doc_id=document)
        except PageIndexAPIError as exc:
            if not wait or exc.status_code != 409:
                raise
            self.documents._wait_until_ready(document, deadline - time.monotonic())
            job = api.create_classify(configuration, doc_id=document)
        return self._done(job, deadline, timeout) if wait else job

    def _send(self, file_path: str, collection: Optional[str],
              configuration: dict) -> dict:
        def send():
            return self.documents._api.create_classify(
                configuration, file_path=file_path,
                collection_id=self._collection(collection))

        try:
            return send()
        except PageIndexAPIError as exc:
            # The collection found earlier is gone: find or make it again.
            if exc.status_code != 404 or not self._collections.pop(collection, None):
                raise
            return send()

    def _done(self, job: dict, deadline: float, timeout: float) -> dict:
        import requests
        api = self.documents._api
        while job["status"] in ("pending", "running"):
            if time.monotonic() > deadline:
                raise PageIndexAPIError(
                    f"Classify job {job['id']} is still {job['status']} after "
                    f"{int(timeout)}s. It goes on running in the cloud — read "
                    "it with get_classify(job_id).")
            time.sleep(1)
            try:
                job = api.get_classify(job["id"])
            except (PageIndexAPIError, requests.RequestException) as exc:
                raise PageIndexAPIError(
                    f"Could not read classify job {job['id']}: {exc}. It may "
                    "still be running in the cloud — read it with "
                    "get_classify(job_id).",
                    status_code=getattr(exc, "status_code", None)) from exc
        if job["status"] != "completed":
            raise PageIndexAPIError(
                f"Classify job {job['id']} {job['status']}"
                + (f": {job['error_message']}" if job.get("error_message") else "."))
        if job.get("failed_pages"):
            raise PageIndexAPIError(
                f"Classify job {job['id']} has pages it could not read: "
                f"{', '.join(map(str, job['failed_pages']))}. Its result leaves "
                "them out: read it with get_classify(job_id).")
        return job

    def wait_classify(self, job_id: str, timeout: float = 1800.0) -> dict[str, Any]:
        """
        Wait for a classify job, and return it once it is completed: the dict
        ``classify()`` returns. Raises PageIndexAPIError if it failed, could
        not read some of its pages, or is not done within ``timeout`` seconds
        (it goes on running).
        """
        return self._done(
            self.documents._api.get_classify(job_id),
            time.monotonic() + timeout, timeout)

    def get_classify(self, job_id: str) -> dict[str, Any]:
        """
        Get a classify job, with its result once it is completed: the dict
        ``classify()`` returns.
        """
        return self.documents._api.get_classify(job_id=job_id)
