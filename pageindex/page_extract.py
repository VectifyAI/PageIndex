"""Page Extract: JSON that fits a schema, extracted from a PDF."""
import os
import time
from typing import Any, Optional, Union

from .client import PageIndexClient, _cloud_client, _pageindex_alias
from .errors import PageIndexAPIError

COLLECTION = "Page Extract"


class PageExtractClient:
    """
    Page Extract over your PageIndex cloud documents. Cloud-only.

        client = PageExtractClient()   # reads PAGEINDEX_API_KEY
        job = client.extract("invoice.pdf", schema)
        job["extract_result"]

    A PDF given by its path is uploaded to the "Page Extract" collection,
    with no outline index built. The documents are the ones
    ``PageIndexClient`` and ``PageSearchClient`` see with the same key;
    ``client.documents`` is that ``PageIndexClient``, for listing and
    deleting them.

    Args:
        api_key (str, optional): Your PageIndex API key. Defaults to the
            PAGEINDEX_API_KEY environment variable, then to
            PAGEINDEX_EXTRACT_API_KEY, where Page Extract's SDK read its key.
        client (PageIndexClient, optional): A cloud ``PageIndexClient`` to
            work through, in place of ``api_key``.
    """

    def __init__(self, api_key: Optional[str] = None, *,
                 client: Optional[PageIndexClient] = None):
        self.documents = _cloud_client(api_key, client, "PageExtractClient",
                                       "PAGEINDEX_EXTRACT_API_KEY")
        self._collections: dict[str, str] = {}

    pageindex = _pageindex_alias("PageExtractClient")

    def extract(
        self,
        document: Union[str, os.PathLike],
        schema: Any,
        *,
        tier: str = "standard",
        pages: Union[str, int, tuple[int, int], None] = None,
        cite_sources: bool = False,
        system_prompt: Optional[str] = None,
        wait: bool = True,
        timeout: float = 3600.0,
        collection: Optional[str] = COLLECTION,
    ) -> dict[str, Any]:
        """
        Extract JSON that fits ``schema`` from a PDF.

            job = client.extract("invoice.pdf", schema)
            job["extract_result"]

        Args:
            document (str): A PDF's path: the file is uploaded to
                ``collection`` with the job, with no outline index built.
                Only the extraction is charged, once the job completes, and
                the file is kept as a document from then on; a job that fails
                or is cancelled is not charged, and leaves a file it uploaded
                as a failed document, with its name and file but no pages,
                until you delete it. Uploading the same file again uses the
                stored document, unless it failed: then the file is uploaded
                as a new document, usually under a suffixed name such as
                ``name_1.pdf``. Or
                the ``doc_id`` of an uploaded document:
                one still processing is waited for first, and with
                ``wait=False`` raises PageIndexAPIError (HTTP 409).
            schema (dict | type): A JSON Schema whose top is an object or an
                array, or a Pydantic model class.
            tier (str): "standard" (default, $0.03 a page), "pro" ($0.05)
                or "max" ($0.10, the most accurate).
            pages (str | int | tuple, optional): Extract from these pages
                only: "3-9", (3, 9) or 3. The extraction is charged for them
                alone.
            cite_sources (bool): Give the page each list item is on, under
                'extract_metadata'.
            system_prompt (str, optional): Your own instructions for the
                extraction.
            wait (bool): Return the job once it is completed (default), up to
                ``timeout`` seconds; raises PageIndexAPIError if it failed or
                was cancelled, or is not done in time (it goes on running:
                read it with ``get_extract()``). False returns it pending
                once it starts: ``wait_extract()`` waits for it later.
            timeout (float): Seconds to wait, the document's upload and
                processing included. Defaults to an hour.
            collection (str, optional): Where a file is uploaded: a
                collection's name (any case) or ID, "Page Extract" by
                default. A name none of your collections has makes one.
                None stores the document outside any collection.

        Returns:
            dict: The job: {'id', 'status', 'doc_id', 'configuration',
            'created_at', 'updated_at'}. 'status' is "pending", "running",
            "completed", "failed" or "cancelled". Completed, it also has
            'extract_result' (the JSON that fits the schema),
            'extract_metadata' ({'field_metadata': {...}}, with
            ``cite_sources`` the page of each list item by its JSON Pointer,
            as ``{"/rows/0": {"page": 3}}``) and 'usage' ({'pages',
            'credits'}); failed, 'error_message'.
        """
        api = self.documents._api
        if isinstance(schema, dict):
            data_schema = schema
        elif hasattr(schema, "model_json_schema"):
            data_schema = schema.model_json_schema()
        else:
            raise PageIndexAPIError(
                "schema must be a JSON Schema (a dict) or a Pydantic model class.")
        configuration: dict[str, Any] = {"data_schema": data_schema, "tier": tier}
        if isinstance(pages, tuple):
            pages = f"{pages[0]}-{pages[1]}"
        if pages is not None:
            configuration["target_pages"] = str(pages)
        if cite_sources:
            configuration["cite_sources"] = True
        if system_prompt:
            configuration["system_prompt"] = system_prompt
        document = os.fspath(document)
        deadline = time.monotonic() + timeout
        if os.path.exists(document):
            for last in (False, True):
                stored = self._store(document, collection, configuration)
                if "extract_job_id" in stored:
                    job = api.get_extract(stored["extract_job_id"])
                    return self._done(job, deadline, timeout) if wait else job
                # No job started: the file is stored already and not ready, or
                # the server is older than extraction on upload.
                try:
                    self.documents._wait_until_ready(
                        stored["doc_id"], deadline - time.monotonic())
                    break
                except PageIndexAPIError as exc:
                    # The job it was stored with did not complete, and failed
                    # it, or it was deleted: the file is uploaded again.
                    if last or not (
                            exc.status_code == 404
                            or self.documents.get_document(
                                stored["doc_id"]).get("status") == "failed"):
                        raise
            document = stored["doc_id"]
        elif document.lower().endswith(".pdf"):
            raise FileNotFoundError(document)
        try:
            job = api.create_extract(configuration, doc_id=document)
        except PageIndexAPIError as exc:
            if not wait or exc.status_code != 409:
                raise
            self.documents._wait_until_ready(document, deadline - time.monotonic())
            job = api.create_extract(configuration, doc_id=document)
        return self._done(job, deadline, timeout) if wait else job

    def _store(self, file_path: str, collection: Optional[str],
               configuration: dict) -> dict:
        def upload():
            return self.documents._upload(
                file_path, False, collection_id=self._collection(collection),
                index=False, extract=configuration)

        try:
            return upload()
        except PageIndexAPIError as exc:
            # The collection found earlier is gone: find or make it again.
            if exc.status_code != 404 or not self._collections.pop(collection, None):
                raise
            return upload()

    def _collection(self, collection: Optional[str]) -> Optional[str]:
        if collection is None or collection in self._collections:
            return self._collections.get(collection)
        api = self.documents._api

        name = collection.strip().casefold()

        def stored():
            return next(
                (c["id"] for c in api.list_collections()["collections"]
                 if c["id"] == collection or c["name"].casefold() == name), None)

        found = stored()
        if found is None:
            try:
                found = api.create_collection(name=collection)["collection"]["id"]
            except PageIndexAPIError as exc:
                if exc.status_code != 409:
                    raise
                found = stored()
                if found is None:
                    raise PageIndexAPIError(
                        f"A collection named like '{collection}' exists, but "
                        "not by that spelling: pass its exact name or ID.",
                        status_code=409) from exc
        self._collections[collection] = found
        return found

    def _done(self, job: dict, deadline: float, timeout: float) -> dict:
        import requests
        api = self.documents._api
        while job["status"] in ("pending", "running"):
            if time.monotonic() > deadline:
                raise PageIndexAPIError(
                    f"Extract job {job['id']} is still {job['status']} after "
                    f"{int(timeout)}s. It goes on running in the cloud — read "
                    "it with get_extract(job_id), or stop it with "
                    "cancel_extract(job_id).")
            time.sleep(3)
            try:
                job = api.get_extract(job["id"])
            except (PageIndexAPIError, requests.RequestException) as exc:
                raise PageIndexAPIError(
                    f"Could not read extract job {job['id']}: {exc}. It may "
                    "still be running in the cloud — read it with "
                    "get_extract(job_id), or stop it with "
                    "cancel_extract(job_id).",
                    status_code=getattr(exc, "status_code", None)) from exc
        if job["status"] != "completed":
            raise PageIndexAPIError(
                f"Extract job {job['id']} {job['status']}"
                + (f": {job['error_message']}" if job.get("error_message") else "."))
        return job

    def wait_extract(self, job_id: str, timeout: float = 3600.0) -> dict[str, Any]:
        """
        Wait for an extract job, and return it once it is completed: the
        dict ``extract()`` returns. Raises PageIndexAPIError if it failed or
        was cancelled, or is not done within ``timeout`` seconds (it goes on
        running).
        """
        return self._done(
            self.documents._api.get_extract(job_id),
            time.monotonic() + timeout, timeout)

    def get_extract(self, job_id: str) -> dict[str, Any]:
        """
        Get an extract job, with its result once it is completed: the dict
        ``extract()`` returns.
        """
        return self.documents._api.get_extract(job_id=job_id)

    def list_extracts(self, limit: int = 20,
                      after: Optional[str] = None) -> dict[str, Any]:
        """
        List your extract jobs, newest first, without their results.

        Args:
            limit (int): Maximum jobs to return (1-100).
            after (str, optional): The last page's 'next', for the page after
                it.

        Returns:
            dict: {'jobs': [...], 'next'}; 'next' is the id to pass as
            ``after``, or None on the last page.
        """
        return self.documents._api.list_extracts(limit=limit, after=after)

    def cancel_extract(self, job_id: str) -> dict[str, Any]:
        """
        Stop an extract job that is pending or running: nothing is charged,
        and a file uploaded with it stays as a failed document, with its name
        and file but no pages, until you delete it. One that has already ended
        raises PageIndexAPIError (HTTP 409).

        Returns:
            dict: The job, cancelled.
        """
        return self.documents._api.cancel_extract(job_id=job_id)

    def delete_extract(self, job_id: str) -> dict[str, Any]:
        """
        Delete an extract job and its result. The document it read stays;
        one uploaded with a job deleted before it completes stays failed.

        Returns:
            dict: {'id', 'deleted': True}.
        """
        return self.documents._api.delete_extract(job_id=job_id)
