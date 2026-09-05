"""PageIndex SDK client: the 0.2.x cloud surface, now with a local mode."""
from __future__ import annotations

import math
import os
import re
import threading
import time
import warnings
from numbers import Real
from typing import (TYPE_CHECKING, Any, Callable, Iterator, Literal, Mapping,
                    Optional, Union, cast, overload)

from .chat_stream import ChatStream
from .errors import PageIndexAPIError


_litellm_preload_started = False


def _preload_litellm() -> None:
    """Start litellm's multi-second import in the background, once per
    process — a per-client thread would churn under per-request clients."""
    global _litellm_preload_started
    if _litellm_preload_started:
        return
    _litellm_preload_started = True

    def _import() -> None:
        try:
            import litellm  # noqa: F401
        except Exception:
            pass

    threading.Thread(target=_import, daemon=True).start()


def _parse_pages(pages: str) -> list[int]:
    from .agent_tools import _PageSpecError, _expand_pages
    if isinstance(pages, str):
        # 0.2.10 tolerated whitespace on this surface; the tool layer stays
        # on the strict contract pattern.
        pages = re.sub(r"\s*([,-])\s*", r"\1", pages.strip())
    try:
        return _expand_pages(pages)
    except _PageSpecError as exc:
        raise PageIndexAPIError(str(exc)) from exc


def _agents_sdk_model_name(model: str) -> str:
    """Preserve supported Agents SDK prefixes and route other provider paths via LiteLLM."""
    passthrough_prefixes = ("litellm/", "openai/")
    if not model or "/" not in model:
        return model
    if model.startswith(passthrough_prefixes):
        return model
    return f"litellm/{model}"


_LOCAL_INDEX_KEYS = ("model", "summary_model", "backend", "storage_path")

# Near-synonyms of "cloud" that would otherwise parse as model names —
# a silent wrong mode. They error, pointing at the real word.
_RESERVED_MODE_WORDS = {"hosted", "managed"}


def _env_cloud_key(spelling: str, inline: str = "api_key=...") -> str:
    # .env support lives in utils' import-time load_dotenv(): load it
    # before the read, or a key in .env is visible only by import order.
    from . import utils  # noqa: F401
    key = os.environ.get("PAGEINDEX_API_KEY")
    if not key:
        raise PageIndexAPIError(
            f"{spelling} reads the PageIndex API key from the "
            "PAGEINDEX_API_KEY environment variable, which is not set — "
            f"export it, or pass the key inline ({inline}).")
    return key


# One argument vocabulary regardless of spelling: these values are shape-
# checked in the constructor, so a wrong type or an empty value refuses
# there as a PageIndexAPIError — never later, never silently.
_ARG_TYPES: "dict[str, tuple[type, ...]]" = {
    "model": (str,), "index_model": (str,), "summary_model": (str,),
    "chat_model": (str,), "retrieve_model": (str,),
    "storage_path": (str, os.PathLike), "index_backend": (dict,),
    "chat_backend": (dict,)}


def _declared_mode(value, side: str):
    if isinstance(value, str):
        value = value.strip().lower()
    if value not in (None, "cloud", "local"):
        raise PageIndexAPIError(
            f'{side} "mode" must be "cloud" or "local", not {value!r}.')
    return value


_CloudKey = Union[str, Callable[[], str], None]


def _resolve_index_slot(index) -> "tuple[_CloudKey, dict[str, Any]]":
    """The ``index=`` slot as (cloud api_key, local overrides). A dict
    declares its side by its keys; an optional "mode" states it and must
    agree. Keyless cloud spellings ("cloud" / "pageindex-cloud",
    {"mode": "cloud"}) return the environment read as a thunk, so the
    caller's mode cross-check runs before the environment is touched."""
    from .types import PAGEINDEX_CLOUD
    if isinstance(index, str):
        # Normalized compare: a case/whitespace variant of a mode word
        # must never fall through and silently become a model name.
        word = index.strip().lower()
        if word in (PAGEINDEX_CLOUD, "cloud"):
            return lambda: _env_cloud_key(f'index="{index.strip()}"',
                                          'index={"api_key": ...}'), {}
        if word == "local":
            return None, {}
        if word in _RESERVED_MODE_WORDS:
            raise PageIndexAPIError(
                f'index="{index}" is not a mode word — the cloud spelling '
                'is index="cloud" (key from PAGEINDEX_API_KEY) or '
                'index={"api_key": ...}.')
        if index.strip():
            return None, {"index_model": index}
        raise PageIndexAPIError(
            "index is an empty string — pass a local index model name, "
            'or "cloud".')
    if isinstance(index, Mapping):
        # None-valued keys mean "absent", exactly like the flat arguments.
        conf = {name: value for name, value in index.items()
                if value is not None}
        declared = _declared_mode(conf.pop("mode", None), "index")
        if not conf:
            if declared == "cloud":
                return lambda: _env_cloud_key('index={"mode": "cloud"}',
                                              'index={"api_key": ...}'), {}
            if declared == "local":
                return None, {}
            raise PageIndexAPIError(
                "index is an empty dict — its keys pick the side: "
                '{"api_key": ...} for cloud documents, or '
                f"{', '.join(_LOCAL_INDEX_KEYS)} for the local store.")
        unknown = set(conf) - {"api_key"} - set(_LOCAL_INDEX_KEYS)
        if unknown:
            raise PageIndexAPIError(
                f"Unknown index keys ({', '.join(sorted(unknown))}) — "
                'cloud takes "api_key"; local takes '
                f"{', '.join(_LOCAL_INDEX_KEYS)}.")
        if "api_key" in conf:
            if declared == "local":
                raise PageIndexAPIError(
                    'index declares mode "local" but carries api_key — '
                    "an API key means cloud documents. Drop one of them.")
            if len(conf) > 1:
                raise PageIndexAPIError(
                    "index mixes cloud and local keys — cloud documents "
                    'take {"api_key": ...} alone; the cloud pipeline does '
                    "its own indexing.")
            key = conf["api_key"]
            if not key or not isinstance(key, str):
                raise PageIndexAPIError(
                    'index["api_key"] must be a non-empty string.')
            return key, {}
        if declared == "cloud":
            raise PageIndexAPIError(
                'index declares mode "cloud" but carries local keys '
                f"({', '.join(sorted(conf))}) — the cloud pipeline does "
                'its own indexing; cloud takes "api_key" only.')
        mapped = {"index_model": conf.get("model"),
                  "summary_model": conf.get("summary_model"),
                  "index_backend": conf.get("backend"),
                  "storage_path": conf.get("storage_path")}
        return None, {name: value for name, value in mapped.items()
                      if value is not None}
    raise PageIndexAPIError("index must be a string or a dict.")


def _resolve_chat_slot(chat) -> "tuple[Optional[str], dict[str, Any]]":
    """The ``chat=`` slot as (mode, own-model overrides) — mode is
    "managed", "own", or None (nothing declared beyond the overrides)."""
    from .types import PAGEINDEX_CLOUD
    if isinstance(chat, str):
        word = chat.strip().lower()
        if word in (PAGEINDEX_CLOUD, "cloud"):
            return "managed", {}
        if word == "local":
            return "own", {}
        if word in _RESERVED_MODE_WORDS:
            raise PageIndexAPIError(
                f'chat="{chat}" is not a mode word — the managed chat is '
                'chat="cloud".')
        if chat.strip():
            return "own", {"chat_model": chat}
        raise PageIndexAPIError(
            "chat is an empty string — pass a model name, or "
            '"cloud" for the managed chat.')
    if isinstance(chat, Mapping):
        # None-valued keys mean "absent", exactly like the flat arguments.
        conf = {name: value for name, value in chat.items()
                if value is not None}
        declared = _declared_mode(conf.pop("mode", None), "chat")
        unknown = set(conf) - {"model", "backend"}
        if (not conf and declared is None) or unknown:
            raise PageIndexAPIError(
                ("chat is an empty dict" if not conf else
                 f"Unknown chat keys ({', '.join(sorted(unknown))})")
                + ' — chat takes "model" and "backend" (your own model), '
                'or {"mode": "cloud"} / "cloud" for the managed chat.')
        if declared == "cloud":
            if conf:
                raise PageIndexAPIError(
                    'chat declares mode "cloud" but carries '
                    f"({', '.join(sorted(conf))}) — the managed chat "
                    "selects its own model. Drop the mode, or the keys.")
            return "managed", {}
        mapped = {"chat_model": conf.get("model"),
                  "chat_backend": conf.get("backend")}
        return "own", {name: value for name, value in mapped.items()
                       if value is not None}
    raise PageIndexAPIError("chat must be a string or a dict.")


class PageIndexClient:
    """
    Python SDK client for PageIndex.

    Two independent sides, each locally run or cloud-managed:

    - **index** — where documents live. With an ``api_key`` they live in
      your PageIndex cloud account, indexed by the managed pipeline,
      exactly like the 0.2.x SDK. Without one they are indexed on your
      machine by the open-source pipeline (your own LLM provider key,
      e.g. ``OPENAI_API_KEY``) and stored under ``storage_path``.
    - **chat** — who answers. With a chat model configured
      (``chat_model=`` / ``chat=``), the document-QA agent runs in your
      process against your own model and credentials — in both index
      modes. On a cloud client with no chat model, the managed cloud
      chat answers.

    ``api_key`` moves your documents, never your model: ``chat_model``
    always means your own model on your own keys. The fourth combination
    (local documents + managed chat) cannot be expressed — the managed
    chat cannot read your disk.

    Usage:
        client = PageIndexClient()                  # local docs + your model
        client = PageIndexClient(api_key="...")     # cloud docs + managed chat
        client = PageIndexClient(api_key="...",     # cloud docs + your model
                                 chat_model="openai/gpt-5.2")

    ``index=`` / ``chat=`` are the grouped spelling of the same flat
    arguments — a string as shorthand, a dict for the full config; each
    side picks one spelling per client. ``index="cloud"`` (or the label
    ``"pageindex-cloud"``) is the keyless cloud spelling (the key comes
    from the ``PAGEINDEX_API_KEY`` environment variable — which is read
    only when the code explicitly says cloud; a bare ``PageIndexClient()``
    stays local regardless of the environment).

    Args:
        api_key (str, optional): PageIndex cloud API key
            (https://developer.pageindex.ai/api-keys). Omit for local mode.
        index (str | dict, optional): The index side, grouped —
            ``"cloud"`` / ``"pageindex-cloud"`` (cloud, key from the
            environment), ``"local"``, a local index model name, or a
            dict: ``{"api_key": ...}`` for cloud, ``{"model",
            "summary_model", "backend", "storage_path"}`` for local. An
            optional ``"mode"`` key (``"cloud"`` / ``"local"``) states
            the side and must agree with the other keys; ``{"mode":
            "cloud"}`` alone reads the key from the environment. Not
            combinable with this side's flat arguments.
        chat (str | dict, optional): The chat side, grouped — a model
            name (your own model), ``"cloud"`` / ``"pageindex-cloud"``
            (managed chat, cloud clients only), ``"local"`` (your own
            model, the default one), or ``{"model", "backend"}``. An
            optional ``"mode"`` key states the side: ``{"mode":
            "cloud"}`` alone is the managed chat, ``"local"`` is your
            own model and must agree with the other keys. Not
            combinable with this side's flat arguments.
        mode (str, optional): Client-level declaration of where the
            documents live — ``"cloud"`` or ``"local"`` — checked
            against the other arguments (``mode="local"`` with an
            api_key errors). ``mode="cloud"`` alone reads the key from
            the ``PAGEINDEX_API_KEY`` environment variable. Always
            optional: the arguments themselves already carry the mode.
        index_model (str, optional): Local mode only — LLM used to index
            documents (structure and summaries). Defaults to the SDK
            default (fast and cheap).
        chat_model (str, optional): Your own model for the chat surfaces
            (``chat``, ``chat_completions``), exposed as
            ``client.chat_model`` — on a cloud client, setting it runs
            the document-QA agent in your process over the cloud
            documents (page content then flows through your process to
            your model provider). Chat names route through LiteLLM and
            mean what LiteLLM says they mean; bare names are
            OpenAI-compatible shorthand, and ``openai/Qwen/...`` is the
            form for an OpenAI-compatible server that itself serves
            slashed model ids (vLLM, TGI). Defaults to the SDK default
            (strong); reads ``None`` on a cloud client where the managed
            chat answers.
        model (str, optional): Local mode only — one model for both roles:
            sets the default for ``index_model`` and ``chat_model`` at
            once. The role-specific arguments win over it. (Also the
            0.2.8-era name for the indexing model — old configs keep
            working unchanged.)
        summary_model (str, optional): Local mode only — legacy: overrides
            the model used for node summaries and document descriptions;
            ``index_model`` covers this.
        retrieve_model (str, optional): Legacy name for ``chat_model`` —
            same meaning everywhere, cloud clients included.
        storage_path (str or os.PathLike, optional): Local mode only —
            directory where indexed documents are stored. Defaults to
            ``./.pageindex``.
        index_backend (dict, optional): Local mode only — connection
            overrides for the indexing lane's LLM calls. Keys are
            LiteLLM's own connection params — ``api_key``, ``api_base``,
            ``api_version``, ``aws_*``, … — passed through verbatim.
        chat_backend (dict, optional): Default connection overrides for
            the chat surfaces — a chat-side argument like ``chat_model``,
            so on a cloud client it selects own-model chat. A call's own
            ``backend`` keys win over it. The dict reaches whichever
            door runs, in that door's vocabulary (see each method) —
            ``api_key`` / ``base_url`` mean the same thing on all three.

    PageIndexCloudClient / PageIndexLocalClient pin the index side at
    construction instead of inferring it from api_key.

    Local mode differences (all documented per method): indexing is
    synchronous, only PDFs are supported, and folders / ``beta_headers`` /
    the deprecated retrieval API (``submit_query``, ``get_retrieval``) are
    cloud-only.
    """

    BASE_URL = "https://api.pageindex.ai"

    _pin: Optional[str] = None  # the pinned subclasses' index side

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        index: Optional[Union[Mapping[str, Any], str]] = None,
        chat: Optional[Union[Mapping[str, Any], str]] = None,
        mode: Optional[str] = None,
        index_model: Optional[str] = None,
        chat_model: Optional[str] = None,
        model: Optional[str] = None,
        summary_model: Optional[str] = None,
        retrieve_model: Optional[str] = None,
        storage_path: Optional[Union[str, os.PathLike[str]]] = None,
        index_backend: Optional[dict[str, Any]] = None,
        chat_backend: Optional[dict[str, Any]] = None,
    ):
        if api_key == "":
            raise PageIndexAPIError(
                "api_key is an empty string. Pass a real PageIndex API key for "
                "cloud mode, or omit api_key entirely for local mode."
            )
        # Each side picks one spelling — its slot, or the flat arguments.
        # ``model`` sets every role, so it claims both sides.
        index_flat: dict[str, Any] = {
            name: value for name, value in
            (("api_key", api_key),
             ("index_model", index_model),
             ("summary_model", summary_model),
             ("index_backend", index_backend),
             ("storage_path", storage_path), ("model", model))
            if value is not None}
        chat_flat: dict[str, Any] = {
            name: value for name, value in
            (("chat_model", chat_model),
             ("retrieve_model", retrieve_model),
             ("chat_backend", chat_backend), ("model", model))
            if value is not None}
        if model is not None and (index is not None or chat is not None):
            raise PageIndexAPIError(
                "model= sets both roles at once, so no slot can absorb "
                'it — name the model inside the slot ({"model": ...}) '
                "and use index_model= / chat_model= for a side written "
                "flat.")
        if index is not None and index_flat:
            raise PageIndexAPIError(
                "index= and the flat index-side arguments "
                f"({', '.join(sorted(index_flat))}) are two spellings of "
                "the same thing — use one or the other.")
        if chat is not None and chat_flat:
            raise PageIndexAPIError(
                "chat= and the flat chat-side arguments "
                f"({', '.join(sorted(chat_flat))}) are two spellings of "
                "the same thing — use one or the other.")
        # ``mode=`` is a cross-check, not a spelling: it combines with
        # either spelling of the index side and must agree with it. The
        # pinned classes declare the side by class; their errors name the
        # class, never a mode= the user did not write.
        declared = _declared_mode(mode, "client") or self._pin
        pinned = type(self).__name__ if self._pin else None
        if index is not None:
            cloud_key, index_conf = _resolve_index_slot(index)
            if declared == "local" and cloud_key is not None:
                raise PageIndexAPIError(
                    f"{pinned} pins local documents — that index= selects "
                    "cloud documents. Drop it, or use PageIndexCloudClient."
                    if pinned else
                    'mode="local" disagrees with index= — that index '
                    "selects cloud documents. Drop one of them.")
            if declared == "cloud" and cloud_key is None:
                raise PageIndexAPIError(
                    f"{pinned} pins cloud documents — that index= "
                    "configures the local store. Drop it, or use "
                    "PageIndexLocalClient."
                    if pinned else
                    'mode="cloud" disagrees with index= — that index '
                    "configures the local store. Drop one of them.")
            if callable(cloud_key):
                cloud_key = cloud_key()
        else:
            cloud_key = api_key
            if declared == "local" and api_key is not None:
                raise PageIndexAPIError(
                    'mode="local" conflicts with api_key — an API key '
                    "means cloud documents. Drop one of them.")
            if declared == "cloud" and cloud_key is None:
                cloud_key = _env_cloud_key('mode="cloud"')
            index_conf = {name: value for name, value in index_flat.items()
                          if name != "api_key"}
        if chat is not None:
            chat_mode, chat_conf = _resolve_chat_slot(chat)
        else:
            chat_mode = "own" if chat_flat else None
            chat_conf = chat_flat
        # Every spelling lands here: strings are stripped, wrong types and
        # empty values refuse loudly — an empty chat-side value must never
        # silently select own-model chat.
        for side, slot, conf in (("index", index, index_conf),
                                 ("chat", chat, chat_conf)):
            for name, value in conf.items():
                # Slot keys are the flat names with the side prefix off.
                shown = (f'{side}["{name.removeprefix(side + "_")}"]'
                         if slot is not None else name)
                if not isinstance(value, _ARG_TYPES[name]):
                    raise PageIndexAPIError(
                        f"{shown} must be a {_ARG_TYPES[name][0].__name__}, "
                        f"got {type(value).__name__}.")
                if isinstance(value, str):
                    value = conf[name] = value.strip()
                if not value:
                    raise PageIndexAPIError(
                        f"{shown} is empty — it configures nothing. Pass a "
                        "real value, or drop the argument.")

        if cloud_key is not None:
            if index_conf:
                raise PageIndexAPIError(
                    "Cloud documents are indexed by the PageIndex pipeline "
                    "— the index-side arguments "
                    f"({', '.join(sorted(index_conf))}) have nothing to "
                    "configure there; remove them. (chat_model= / chat= "
                    "stay yours: they run the chat agent in your process "
                    "with your own model.)")
            self.api_key = cloud_key
            from .cloud_api import CloudAPI
            self._api = CloudAPI(self)
            if chat_mode == "own":
                from .utils import ConfigLoader
                overrides = {name: value for name, value in chat_conf.items()
                             if name in ("chat_model", "retrieve_model")
                             and value}
                opt = ConfigLoader().load(overrides or None)
                self.chat_model = opt.chat_model
                self.chat_backend = chat_conf.get("chat_backend")
                _preload_litellm()
            else:
                # Managed chat: the endpoint selects its own model.
                self.chat_model = None
                self.chat_backend = None
        else:
            if chat_mode == "managed":
                if pinned:
                    exits = (f"{pinned} pins local documents: use "
                             "PageIndexCloudClient (or PageIndexClient("
                             "api_key=...))")
                else:
                    exits = ('Go cloud (api_key=... or index="cloud")'
                             + (' and drop mode="local"' if mode is not None
                                else ""))
                raise PageIndexAPIError(
                    "The managed chat needs cloud documents — it cannot "
                    f"read the local store. {exits}, or set your own chat "
                    "model instead.")
            from .utils import ConfigLoader
            overrides = {name: value for name, value in
                         {**index_conf, **chat_conf}.items()
                         if name in ("model", "index_model", "summary_model",
                                     "chat_model", "retrieve_model")
                         and value}
            opt = ConfigLoader().load(overrides or None)
            self.model = opt.model
            self.index_model = opt.index_model
            self.summary_model = opt.summary_model
            self.chat_model = opt.chat_model
            self.chat_backend = chat_conf.get("chat_backend")
            self.storage_path = index_conf.get("storage_path") or ".pageindex"
            from .local_api import LocalAPI
            self._api = LocalAPI(
                storage_path=self.storage_path,
                model=self.model,
                summary_model=self.summary_model,
                index_backend=index_conf.get("index_backend"),
            )
            # LiteLLM's multi-second import would otherwise land on the
            # first chat call; failures resurface there with real context.
            _preload_litellm()

    @property
    def _local_chat(self) -> bool:
        # Derived, never stored: own-model chat is exactly "a chat model
        # is configured" (None on a managed-chat client). Blank configures
        # nothing — the constructor refuses it, and assignment must agree.
        model = getattr(self, "chat_model", None)
        if isinstance(model, str):
            return bool(model.strip())
        return model is not None

    def _require_own_chat(self, lane: str) -> None:
        # The one refusal for chat(protocol=...), the doors behind it, and
        # instructions: shared, so the doors cannot drift from chat().
        if self._local_chat:
            return
        if not getattr(self, "api_key", None):
            raise PageIndexAPIError(
                "chat_model is empty — it configures nothing, and a local "
                "client has no managed chat to fall back to. Set "
                "chat_model=... to run the agent with your own model.")
        raise PageIndexAPIError(
            f"{lane} drives your own chat model — construct the client "
            "with chat_model=... (or a chat= model); the managed cloud chat "
            "serves the answer lane and chat_completions() only.")

    if not TYPE_CHECKING:
        # The protocol doors live behind chat(protocol=...); their old
        # names are the vendor SDKs' own, so an agent-written
        # client.messages(...) fails here with the way in. Runtime-only:
        # a __getattr__ the type checker can see would silence every
        # attribute typo on the client.
        def __getattr__(self, name):
            if name in ("responses", "messages"):
                raise AttributeError(
                    f"{name}() moved: call chat(protocol={name!r}, ...) "
                    "— the same protocol, engine, and envelope. Pass the "
                    "rest by keyword; its sampling and thinking fields "
                    "ride extra_body under their wire names.")
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}")

    @property
    def retrieve_model(self):
        """Legacy name for ``chat_model``."""
        return self.chat_model

    @retrieve_model.setter
    def retrieve_model(self, value):
        # 0.2.9 allowed assignment; keep the write path working too.
        self.chat_model = value

    # ---------- DOCUMENT SUBMISSION ----------

    def submit_document(
        self,
        file_path: str,
        mode: Optional[str] = None,
        beta_headers: Optional[list[str]] = None,
        folder_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        """
        Submit a PDF document for processing. Returns {'doc_id': ..., 'name': ...}.

        Cloud: uploads the file; processing is asynchronous. Pass
        ``wait=True`` to block until the document is ready, or poll
        ``get_document(doc_id)['status']`` yourself.

        Local: indexes the document in this call and stores it under
        ``storage_path``. Defaults to Flash indexing: layout-based extraction,
        refined for retrieval (a deterministic merge, then an LLM expansion
        pass); node summaries, the expansion pass, and the document
        description use ``summary_model``. Pass ``mode="standard"`` for a
        full LLM-built tree (slower). ``beta_headers`` and ``folder_id`` are
        cloud-only.

        Args:
            file_path (str): Path to the PDF file.
            mode (str, optional): Processing mode. Local defaults to "flash";
                pass "standard" for a full LLM-built tree. Cloud modes are
                passed through (e.g. "mcp").
            beta_headers (list[str], optional): Cloud-only beta feature headers.
            folder_id (str, optional): Cloud-only folder (workspace) ID.
            metadata (dict, optional): Your own JSON-serializable tags for the
                document; returned in get_tree/get_ocr responses and
                list_documents entries (both modes).
            wait (bool): Return only once the document is ready for use.
                Cloud: polls status until "completed" (raises on "failed" or
                after 30 minutes). Local: indexing is synchronous already, so
                this changes nothing. Leave False to submit many documents
                concurrently and poll afterwards.

        Returns:
            dict: {'doc_id': ..., 'name': ...}. 'name' is the stored document
                name: a taken name gains a numeric suffix (name_1..name_99)
                and a UserWarning is emitted. Older cloud servers omit 'name'.
        """
        result = self._api.submit_document(
            file_path=file_path, mode=mode,
            beta_headers=beta_headers, folder_id=folder_id, metadata=metadata,
        )
        stored = result.get("name")
        if stored and stored != os.path.basename(file_path):
            warnings.warn(
                f'Document "{os.path.basename(file_path)}" was stored as '
                f'"{stored}".',
                stacklevel=2,
            )
        if wait:
            self.wait_until_completed(result["doc_id"])
        return result

    def wait_until_completed(
        self,
        doc_id: str,
        timeout: float = 1800.0,
        poll_interval: float = 2.0,
    ) -> dict[str, Any]:
        """Wait until an existing document is ready and return its metadata.

        ``poll_interval`` is the initial delay. Repeated polls back off by 1.5x,
        capped at 15 seconds (or the initial interval when it is larger).
        """
        import requests

        timings = {"timeout": timeout, "poll_interval": poll_interval}
        validated_timings: dict[str, float] = {}
        for name, value in timings.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite number greater than 0")
            try:
                seconds = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(
                    f"{name} must be a finite number greater than 0"
                ) from exc
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"{name} must be a finite number greater than 0")
            validated_timings[name] = seconds

        timeout_seconds = validated_timings["timeout"]
        interval = validated_timings["poll_interval"]
        max_interval = max(15.0, interval)
        deadline = time.monotonic() + timeout_seconds
        poll_failures = 0
        last_status = None

        def timeout_error() -> PageIndexAPIError:
            return PageIndexAPIError(
                f"Timed out after {timeout_seconds:g}s waiting for document "
                f"processing (doc_id: {doc_id}, last status: {last_status}). "
                "Processing continues in the cloud — poll "
                "get_document(doc_id) for status."
            )

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise timeout_error()
            status = None
            try:
                bounded_get_document = getattr(
                    self._api, "_get_document_with_timeout", None
                )
                if bounded_get_document is None:
                    document = self.get_document(doc_id)
                else:
                    document = bounded_get_document(
                        doc_id, timeout=min(30.0, remaining)
                    )
            except (PageIndexAPIError, requests.RequestException) as exc:
                if time.monotonic() >= deadline:
                    raise timeout_error() from exc
                if getattr(exc, "status_code", None) in (401, 403, 404):
                    raise  # a definite answer, not a poll failure
                # Tolerate transient poll failures; a 30-minute wait should
                # not die on one 502 or dropped connection.
                poll_failures += 1
                if poll_failures >= 3:
                    raise PageIndexAPIError(
                        f"Could not poll document status (doc_id: {doc_id}): "
                        f"{exc}. Processing continues in the cloud — poll "
                        "get_document(doc_id) for status."
                    ) from exc
            else:
                if time.monotonic() >= deadline:
                    raise timeout_error()
                status = document.get("status")
                last_status = status
                poll_failures = 0

            if status == "completed":
                return document
            if status == "failed":
                raise PageIndexAPIError(
                    f"Document processing failed (doc_id: {doc_id})."
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise timeout_error()
            time.sleep(min(interval, remaining))
            interval = min(interval * 1.5, max_interval)

    # ---------- OCR FUNCTIONALITY ----------

    def get_ocr(self, doc_id: str, format: str = "page") -> dict[str, Any]:
        """
        Get OCR status and results.

        Args:
            doc_id (str): Document ID.
            format (str): 'page' for page-based results, 'node' for node-based
                results, or 'raw' for concatenated markdown.

        Returns:
            dict: {'doc_id', 'status', 'retrieval_ready', 'result', ...}.
            With 'page', result entries are {'page_index', 'markdown', ...}.

        Local: the "OCR" result is the text extracted from the PDF while
        indexing (no OCR model runs locally, so scanned/image-only PDFs have
        no local text).
        """
        return self._api.get_ocr(doc_id=doc_id, format=format)

    def get_page_content(self, doc_id: str, pages: str) -> list[dict[str, Any]]:
        """
        Get text content of specific pages.

        Args:
            doc_id (str): Document ID.
            pages (str): Page specifier — '5-7', '3,8', or '12'.

        Returns:
            list: Matching entries from get_ocr (format='page').
        """
        wanted = set(_parse_pages(pages))
        result = self.get_ocr(doc_id, format="page")
        all_pages = result["result"]
        if all_pages is None:
            raise PageIndexAPIError(
                f"Document '{doc_id}' is not ready "
                f"(status: {result.get('status', 'unknown')})"
            )
        return [p for p in all_pages if p["page_index"] in wanted]

    # ---------- TREE GENERATION ----------

    def get_tree(self, doc_id: str, node_summary: bool = False,
                 include_text: bool = True) -> dict[str, Any]:
        """
        Get tree generation status and results.

        Args:
            doc_id (str): Document ID.
            node_summary (bool): Include node summaries in the tree.
            include_text (bool): Include node text (default True).
                False is useful for structure-only views (saves tokens).

        Returns:
            dict: {'doc_id', 'status', 'retrieval_ready', 'result', ...} where
            result nodes are {'title', 'node_id', 'page_index', ('summary' /
            'prefix_summary',) ('text',) 'nodes'}.
        """
        tree = self._api.get_tree(doc_id=doc_id, node_summary=node_summary,
                                  include_text=include_text)
        if not include_text and tree.get("result"):
            from .utils import remove_fields
            tree["result"] = remove_fields(tree["result"], fields=["text"])
        return tree

    def get_document_structure(self, doc_id: str) -> list[dict[str, Any]]:
        """
        Get the document's tree structure without text — summaries included.

        Returns:
            list: Tree nodes with titles, page ranges, and summaries.
        """
        return self.get_tree(doc_id, node_summary=True, include_text=False)["result"]

    def is_retrieval_ready(self, doc_id: str) -> bool:
        """
        Check if a document is ready for retrieval. API errors (including a
        missing document) are reported as False; transport errors (connection
        failures, timeouts) propagate.
        """
        try:
            result = self.get_tree(doc_id)
            return result.get("retrieval_ready", False)
        except PageIndexAPIError:
            return False

    # ---------- RETRIEVAL (cloud-only, deprecated) ----------

    def submit_query(self, doc_id: str, query: str, thinking: bool = False) -> dict[str, Any]:
        """
        Submit a retrieval query for a document. Returns {'retrieval_id': ...}.

        Cloud-only: the cloud API marks this endpoint deprecated in favor of
        chat completions, so local mode does not implement it — raises
        PageIndexAPIError. Use ``chat_completions`` instead.
        """
        return self._require_cloud(
            "submit_query is cloud-only — the retrieval API is deprecated in "
            "favor of chat completions; use chat_completions instead."
        ).submit_query(doc_id=doc_id, query=query, thinking=thinking)

    def get_retrieval(self, retrieval_id: str) -> dict[str, Any]:
        """
        Get retrieval status and results for a submitted query.

        Cloud-only: the cloud API marks this endpoint deprecated in favor of
        chat completions, so local mode does not implement it — raises
        PageIndexAPIError. Use ``chat_completions`` instead.
        """
        return self._require_cloud(
            "get_retrieval is cloud-only — the retrieval API is deprecated in "
            "favor of chat completions; use chat_completions instead."
        ).get_retrieval(retrieval_id=retrieval_id)

    # ---------- CHAT ----------

    # stream and protocol pick the return type: the docstring's `.events`
    # usage and the protocol envelopes must type-check for py.typed
    # consumers
    @overload
    def chat(
        self,
        messages: Union[str, list[dict[str, Any]]],
        doc_id: Optional[Union[str, list[str]]] = None,
        stream: Literal[False] = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        show_process: Union[bool, Mapping[str, Any], None] = None,
        *,
        protocol: None = None,
        instructions: Optional[Union[str, list[dict[str, Any]]]] = None,
        max_turns: Optional[int] = None,
        backend: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> str: ...

    @overload
    def chat(
        self,
        messages: Union[str, list[dict[str, Any]]],
        doc_id: Optional[Union[str, list[str]]] = None,
        *,
        stream: Literal[True],
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        show_process: Union[bool, Mapping[str, Any], None] = None,
        protocol: None = None,
        instructions: Optional[Union[str, list[dict[str, Any]]]] = None,
        max_turns: Optional[int] = None,
        backend: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> ChatStream: ...

    @overload
    def chat(
        self,
        messages: Union[str, list[dict[str, Any]]],
        doc_id: Optional[Union[str, list[str]]] = None,
        stream: Literal[False] = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        show_process: Union[bool, Mapping[str, Any], None] = None,
        *,
        protocol: Literal["responses", "messages"],
        instructions: Optional[Union[str, list[dict[str, Any]]]] = None,
        max_turns: Optional[int] = None,
        backend: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]: ...

    @overload
    def chat(
        self,
        messages: Union[str, list[dict[str, Any]]],
        doc_id: Optional[Union[str, list[str]]] = None,
        *,
        stream: Literal[True],
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        show_process: Union[bool, Mapping[str, Any], None] = None,
        protocol: Literal["responses"],
        instructions: Optional[Union[str, list[dict[str, Any]]]] = None,
        max_turns: Optional[int] = None,
        backend: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> Iterator[dict[str, Any]]: ...

    @overload
    def chat(
        self,
        messages: Union[str, list[dict[str, Any]]],
        doc_id: Optional[Union[str, list[str]]] = None,
        *,
        stream: Literal[True],
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        show_process: Union[bool, Mapping[str, Any], None] = None,
        protocol: Literal["messages"],
        instructions: Optional[Union[str, list[dict[str, Any]]]] = None,
        max_turns: Optional[int] = None,
        backend: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> Iterator[Any]: ...

    @overload
    def chat(
        self,
        messages: Union[str, list[dict[str, Any]]],
        doc_id: Optional[Union[str, list[str]]] = None,
        stream: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        show_process: Union[bool, Mapping[str, Any], None] = None,
        *,
        protocol: None = None,
        instructions: Optional[Union[str, list[dict[str, Any]]]] = None,
        max_turns: Optional[int] = None,
        backend: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> Union[str, ChatStream]: ...

    @overload
    def chat(
        self,
        messages: Union[str, list[dict[str, Any]]],
        doc_id: Optional[Union[str, list[str]]] = None,
        stream: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        show_process: Union[bool, Mapping[str, Any], None] = None,
        *,
        protocol: Optional[str] = None,
        instructions: Optional[Union[str, list[dict[str, Any]]]] = None,
        max_turns: Optional[int] = None,
        backend: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> Union[str, ChatStream, dict[str, Any], Iterator[Any]]: ...

    def chat(
        self,
        messages: Union[str, list[dict[str, Any]]],
        doc_id: Optional[Union[str, list[str]]] = None,
        stream: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        show_process: Union[bool, Mapping[str, Any], None] = None,
        *,
        protocol: Optional[str] = None,
        instructions: Optional[Union[str, list[dict[str, Any]]]] = None,
        max_turns: Optional[int] = None,
        backend: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> Union[str, ChatStream, dict[str, Any], Iterator[Any]]:
        """
        Ask a question about your documents.

        The answer lane (no ``protocol``): thin sugar over the same engine
        as ``chat_completions()`` in every mode — same wire, minus the
        envelope. Returns the answer string (a ``ChatStream`` when
        streaming). Multi-turn: keep your own role/content list of the
        visible conversation (append each answer as an assistant message;
        join a stream into one only with ``show_process=False``) and pass
        it back.

        The protocol lanes (``protocol="responses"`` / ``"messages"``):
        own-model chat driven natively over the OpenAI Responses API or
        Anthropic's Messages API. Input and output are that protocol's own
        shapes — the history may carry its transcript (Responses items, or
        Messages content blocks with prior tool_use/tool_result
        round-trips), and the return is its response envelope, streaming
        its native events. A round-tripped transcript continues the
        agent's memory and the provider's cached prefix: follow-ups re-read
        the run instead of redoing the tool work. The protocol is declared,
        never inferred from the model name. Keep ``doc_id`` and
        ``protocol`` constant across a conversation's calls.

        Args:
            messages: A question string, or the conversation history —
                role/content messages on every lane. ``system`` rows join
                the managed prompt on the answer lane only, wherever they
                sit; the protocol lanes pass rows to the wire as they are
                (use ``instructions`` for persona there). With a protocol,
                also that protocol's transcript items or content blocks.
            doc_id: Document ID or list of IDs to scope the conversation.
                Keep it identical across a conversation's calls. Local
                documents: also enforced at the tool layer, not just
                prompted. Cloud documents: the managed chat scopes
                server-side; own-model chat targets at the prompt level.
            stream: Answer lane: return a ``ChatStream`` — iterate it for
                the answer as text chunks as they are produced
                (``show_process`` is on by default, so the run's process
                arrives woven in; ``show_process=False`` gives the bare
                answer), or read its ``.events`` property instead for the
                run as typed event dicts — thinking/answer deltas, each
                tool call and its full result (own-model chat only; never
                clipped). One run serves one view. Protocol lanes: the
                protocol's own event stream.
            model: Own-model chat only — backend model name (defaults
                to ``chat_model``). ``protocol="messages"`` needs it named
                — a Claude model; there is no cross-vendor default.
            reasoning_effort: Own-model chat only — how hard the model
                thinks (``"low"`` / ``"medium"`` / ``"high"``; what a
                backend accepts is its own). Each lane sends its native
                spelling: LiteLLM's ``reasoning_effort``, Responses
                ``reasoning.effort``, Messages ``output_config.effort``.
                Unset sends nothing — the model's default applies.
            show_process: Answer lane, streamed chat — weave the run into
                the text stream for display: thinking flows as
                "[thinking] " sections, each tool call as a "[tool_call]
                name arguments" line with its "[tool_result]" line, and
                the answer unlabeled. **On by default**, weaving what the mode
                serves: the in-process agent's full run; on a managed
                client, the tool calls the endpoint streams (its wire
                carries no thinking and no tool results). Pass ``False``
                for the bare answer stream — do that before appending a
                streamed answer to the conversation history.
                ``True`` shows everything; a dict (typed as
                ``pageindex.ChatProcessOptions``) selects the parts —
                ``thinking`` / ``tool_call`` / ``tool_result``, bools
                defaulting on — and sets ``max_chars``, the per-line
                summary cap in characters (default 200). Omitted keys
                keep their defaults, so ``{"thinking": False}`` hides
                only thinking and ``{}`` equals ``True``.
                Thinking appears when the backend streams it
                (e.g. Claude models with ``reasoning_effort``; OpenAI
                models expose none on the chat protocol). The labels are
                not a parse format, and a process stream must not be
                appended back as conversation history — for the
                machine-readable process use ``.events``, or a protocol
                lane's transcript. A protocol lane returns that
                transcript itself, so ``show_process`` is an error there.
            protocol: ``None`` for the answer lane, or ``"responses"`` /
                ``"messages"`` — the wire protocol, engine, and
                input/output shapes of this call. Own-model chat only.
            instructions: Own-model chat only — persona or extra guidance
                appended after the managed system prompt (which stays: it
                carries the tool guidance and the document context). A
                string on every lane; with ``protocol="messages"`` also
                a list of Messages system blocks. On the answer lane it
                precedes any ``system`` rows in the history.
            max_turns: Own-model chat only — cap on agent turns per call
                (default 10). The OpenAI lanes raise at the cap;
                ``protocol="messages"`` returns the truncated run
                (``stop_reason: "tool_use"``, its ``messages`` valid for
                continuation).
            backend: Own-model chat only — connection overrides for this
                call's backend, merged over the client's ``chat_backend``
                (per-call keys win): LiteLLM's connection params on the
                answer lane, the openai / anthropic SDK's client params
                on the protocol lanes. Passed through verbatim.
            extra_headers: Own-model chat only — extra HTTP headers
                merged into each backend request; caller headers win.
                LiteLLM's anthropic adapter owns ``anthropic-beta`` on the
                answer lane — Anthropic beta flags ride
                ``protocol="messages"``.
            extra_body: Own-model chat only — the provider's own request
                fields beyond this method's parameters, in the lane's
                wire names (Responses ``max_output_tokens``, Messages
                ``thinking`` / ``top_k``), merged last so they win.
                Answer lane: LiteLLM's own params, mapped or refused per
                provider (``response_format`` has no door on
                LiteLLM-routed models); protocol lanes: verbatim into
                the request body. The managed prompt, conversation and
                tools are not fields here (``system`` / ``instructions``
                / ``input`` / ``messages`` / ``tools`` are refused);
                extend the prompt with ``instructions=``. Credentials
                belong in ``backend``, never here.

        Returns:
            - answer lane, stream=False: the answer string
            - answer lane, stream=True: a ``ChatStream`` — iterating it
              yields text chunks (with show_process, on by default, the
              run's process woven in as labeled sections); ``.events``
              yields typed event dicts:
              ``{"type": "thinking"|"answer", "delta": ...}``,
              ``{"type": "tool_call", "call_id", "name", "arguments"}``,
              ``{"type": "tool_result", "call_id", "name", "output"}``
            - protocol lane, stream=False: the protocol's response
              envelope — Responses: ``output`` plus an ``items``
              transcript and cross-turn ``usage``; Messages: the final
              message with a ``messages`` turn sequence and aggregated
              ``usage``
            - protocol lane, stream=True: an iterator of the protocol's
              own stream events
        """
        if protocol not in (None, "responses", "messages"):
            raise PageIndexAPIError(
                "protocol selects the wire: \"responses\" (OpenAI Responses) "
                "or \"messages\" (Anthropic Messages), or leave it unset for "
                f"the answer lane — got {protocol!r}.")
        if (protocol is not None and show_process is not False
                and show_process is not None):
            raise PageIndexAPIError(
                "show_process weaves the answer lane's run; with "
                f"protocol={protocol!r} the run comes back as the protocol's "
                "own transcript and events — drop show_process, or drop "
                "protocol for the woven text stream.")
        if show_process is not False and show_process is not None:
            from .local_chat import _process_options
            _process_options(show_process)  # a bad value chokes first
            if not stream:
                raise PageIndexAPIError(
                    "show_process shows the run as it happens and requires "
                    "stream=True; only show_process=False (or None) means "
                    f"off — got {show_process!r}.")
        if isinstance(instructions, list) and protocol != "messages":
            raise PageIndexAPIError(
                "instructions blocks are the Messages protocol's shape — "
                "with protocol=\"messages\" they append after the managed "
                "system blocks; the other lanes take a string.")
        if protocol is not None:
            self._require_own_chat(f"chat(protocol={protocol!r})")
            if protocol == "responses":
                body = extra_body
                if reasoning_effort:
                    # OpenAI's own effort field, beside the caller's other
                    # reasoning keys — theirs still win.
                    given = extra_body or {}
                    body = {**given, "reasoning": {
                        "effort": reasoning_effort,
                        **given.get("reasoning", {})}}
                return self._responses(
                    messages, model=model, stream=stream, doc_id=doc_id,
                    instructions=cast(Optional[str], instructions),
                    max_turns=max_turns, extra_body=body,
                    extra_headers=extra_headers, backend=backend)
            if not model:
                raise PageIndexAPIError(
                    "protocol=\"messages\" drives Anthropic's Messages API "
                    "with the Anthropic SDK — name the Claude model with "
                    "model=... (there is no cross-vendor default to guess).")
            body = extra_body
            if reasoning_effort:
                # Anthropic's own effort field, beside the caller's other
                # output_config keys — theirs still win.
                given = extra_body or {}
                body = {**given, "output_config": {
                    "effort": reasoning_effort,
                    **given.get("output_config", {})}}
            return self._messages(
                messages, model=model, stream=stream, doc_id=doc_id,
                system=instructions, max_turns=max_turns, extra_body=body,
                extra_headers=extra_headers, backend=backend)
        if instructions:
            self._require_own_chat("instructions")
            if isinstance(messages, str):
                if not messages.strip():
                    raise PageIndexAPIError(
                        "messages must be a non-empty string or a list of "
                        "message dicts.")
                messages = [{"role": "user", "content": messages}]
            if isinstance(messages, list):
                # The first system text: managed prompt, then instructions,
                # then the history's own system rows.
                messages = [{"role": "system", "content": instructions},
                            *messages]
        if stream:
            # the default means "on where available"
            resolved = True if show_process is None else show_process
            if self._local_chat:
                from .local_chat import run_chat_stream
                return run_chat_stream(self, messages, doc_id=doc_id,
                                       model=model,
                                       reasoning_effort=reasoning_effort,
                                       show_process=resolved,
                                       max_turns=max_turns, backend=backend,
                                       extra_headers=extra_headers,
                                       extra_body=extra_body)
            from .local_chat import run_cloud_chat_stream
            chunks = self.chat_completions(messages, stream=True,
                                           stream_metadata=True,
                                           doc_id=doc_id, model=model,
                                           reasoning_effort=reasoning_effort,
                                           max_turns=max_turns,
                                           backend=backend,
                                           extra_headers=extra_headers,
                                           extra_body=extra_body)
            return run_cloud_chat_stream(
                cast(Iterator[dict[str, Any]], chunks), resolved)
        result = self.chat_completions(messages, doc_id=doc_id, model=model,
                                       reasoning_effort=reasoning_effort,
                                       max_turns=max_turns, backend=backend,
                                       extra_headers=extra_headers,
                                       extra_body=extra_body)
        envelope = cast(dict[str, Any], result)
        try:
            return envelope["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise PageIndexAPIError(
                "The chat response carries no answer: "
                f"{str(envelope)[:200]}") from exc

    def chat_completions(
        self,
        messages: Union[str, list[dict[str, str]]],
        stream: bool = False,
        doc_id: Optional[Union[str, list[str]]] = None,
        temperature: Optional[float] = None,
        stream_metadata: bool = False,
        enable_citations: bool = False,
        model: Optional[str] = None,
        max_turns: Optional[int] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        extra_body: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        backend: Optional[dict[str, Any]] = None,
    ) -> Union[dict[str, Any], Iterator[str], Iterator[dict[str, Any]]]:
        """
        PageIndex Chat Completions: document QA in one call.

        With no chat model configured (a plain cloud client): the managed
        hosted chat endpoint. With one — local mode, or a cloud client
        constructed with ``chat_model=``/``chat=`` (own-model chat) — a
        managed document-QA agent runs in your process over the mode's
        tools (local store, or the live cloud tool set) against your own
        LLM backend, routed through LiteLLM — model names mean what
        LiteLLM says they mean.
        Bare names are OpenAI-compatible shorthand (the OpenAI SDK's usual
        env config — OPENAI_API_KEY, OPENAI_BASE_URL — selects the
        backend, so any OpenAI-compatible server works; write
        ``openai/Qwen/...`` when the server itself serves slashed ids),
        provider-prefixed names — ``anthropic/…``, ``bedrock/…`` — reach
        that provider, and LiteLLM-routed Claude models get the managed
        prompt prefix cache-marked automatically. The non-stream
        response carries the final answer only; streaming yields the
        agent's visible text as it is produced, including narration before
        tool calls. ``finish_reason`` carries the final turn's native
        finish reason — "stop", or the backend's "length" /
        "content_filter" when the last turn was cut short. For
        the tool-use process and prompt-cache round-trip use
        ``chat(protocol="responses")`` or ``chat(protocol="messages")``.

        Args:
            messages: Conversation messages with 'role' and 'content' keys,
                or a bare query string (it becomes a single user message).
                Own-model chat also accepts system/developer messages —
                their content is appended to the managed system prompt —
                and takes text history only: tool-role turns are rejected
                (the managed endpoint forwards them verbatim), and message
                fields beyond role/content are dropped.
            stream: Enable streaming responses.
            doc_id: Document ID or list of IDs to scope the conversation.
                Keep it identical across a conversation's calls — the
                targeting block it adds is re-set each call and is part
                of the cached prompt prefix. Local documents: also
                enforced at the tool layer, not just prompted. Cloud
                documents: the managed chat scopes server-side;
                own-model chat targets at the prompt level.
            temperature: Sampling temperature, passed through to the model.
            stream_metadata: With stream=True, yield chunk dicts instead of
                text pieces.
            enable_citations: Managed chat only — own-model chat raises
                (the in-process engine has no citation machinery).
            model: Own-model chat only — backend model name (defaults to
                ``chat_model``). The managed endpoint selects its own.
            max_turns: Own-model chat only — cap on agent turns per call.
            top_p: Own-model chat only — nucleus sampling, passed
                through to the model.
            max_tokens: Own-model chat only — per-call output cap,
                passed through; it bounds each backend call in the agent
                loop (the way max_turns bounds the loop), not the whole
                run.
            reasoning_effort: Own-model chat only — passed through verbatim as
                LiteLLM's ``reasoning_effort``; each provider maps it to
                its own thinking control, and the values mean what the
                backend says they mean. Unset sends nothing (the
                backend's default applies).
            extra_body: Own-model chat only — extra request fields beyond this
                method's parameters, merged last so they win.
                OpenAI-compatible backends take them verbatim in the
                request body; LiteLLM-routed providers take them as
                LiteLLM's own params (mapped or refused per provider).
                The managed prompt, conversation and tools are not
                fields here (``system`` / ``instructions`` / ``input`` /
                ``messages`` / ``tools`` are refused). Credentials belong
                in ``backend``, never here.
            extra_headers: Own-model chat only — extra HTTP headers merged into
                each backend request; caller headers win. One exception:
                LiteLLM's anthropic adapter owns the ``anthropic-beta``
                header (your value is dropped there) — Anthropic beta
                flags ride ``chat(protocol="messages")``.
            backend: Own-model chat only — connection overrides for this call's
                backend, merged over the client's ``chat_backend``
                (per-call keys win). Keys are LiteLLM's own connection
                params — ``api_key``, ``base_url``, ``api_version``,
                ``aws_*``, … — passed through verbatim.

        Returns:
            - stream=False: complete response dict ({'id', 'object', 'created',
              'choices', 'usage'})
            - stream=True, stream_metadata=False: iterator of text chunks
            - stream=True, stream_metadata=True: iterator of chunk dicts
        """
        if isinstance(messages, str):
            if not messages.strip():
                raise PageIndexAPIError(
                    "messages must be a non-empty string or a list of "
                    "message dicts.")
            messages = [{"role": "user", "content": messages}]
        if self._local_chat:
            from .local_chat import run_chat_completions
            return run_chat_completions(
                self, messages, stream=stream, doc_id=doc_id,
                temperature=temperature, stream_metadata=stream_metadata,
                enable_citations=enable_citations, model=model,
                max_turns=max_turns, top_p=top_p, max_tokens=max_tokens,
                reasoning_effort=reasoning_effort, extra_body=extra_body,
                extra_headers=extra_headers, backend=backend,
            )
        if not getattr(self, "api_key", None):
            raise PageIndexAPIError(
                "chat_model is empty — it configures nothing, and a local "
                "client has no managed chat to fall back to. Set "
                "chat_model=... to run the agent with your own model.")
        if (model or max_turns is not None or top_p is not None
                or max_tokens is not None or reasoning_effort
                or extra_body or extra_headers or backend):
            raise PageIndexAPIError(
                "model, max_turns, top_p, max_tokens, reasoning_effort, "
                "extra_body, extra_headers and backend drive your own chat "
                "model, which this client does not configure — construct "
                "the client with chat_model=... (or a chat= model) to run the "
                "agent in your process, or drop them to use the managed "
                "chat endpoint, which selects its own model."
            )
        from .cloud_api import CloudAPI
        return cast(CloudAPI, self._api).chat_completions(
            messages=messages, stream=stream, doc_id=doc_id,
            temperature=temperature, stream_metadata=stream_metadata,
            enable_citations=enable_citations,
        )

    def _responses(
        self,
        input: Union[str, list[dict[str, Any]]],
        model: Optional[str] = None,
        stream: bool = False,
        doc_id: Optional[Union[str, list[str]]] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_turns: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[dict[str, Any]] = None,
        extra_body: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        backend: Optional[dict[str, Any]] = None,
    ) -> Union[dict[str, Any], Iterator[dict[str, Any]]]:
        """
        The engine behind ``chat(protocol="responses")``: document QA over
        the OpenAI Responses protocol.

        Own-model chat only — local mode, or a cloud client constructed
        with ``chat_model=``/``chat=``. Drives your backend's /responses
        end to end (no translation layer). The envelope is official
        Responses shape — ``output`` carries the model-produced items
        and parses with the openai SDK types — and the whole process
        transcript (including the tool outputs the SDK executed) rides
        in the extra ``items`` field.
        Append the returned ``items`` to your next call's ``input`` verbatim
        to keep provider prompt-cache prefix continuity and the agent's
        memory of what it already read.

        Requires a backend that supports the
        Responses API; backends that only speak chat.completions should use
        ``chat_completions()``. Provider-prefixed models (``anthropic/…``)
        route through LiteLLM's chat.completions adapter and are therefore
        refused here — use ``chat_completions()`` or
        ``chat(protocol="messages")`` for those.

        Args:
            input: A user message string, or a list of Responses input items
                (round-trip prior ``items`` here).
            model: Backend model name (defaults to ``chat_model``).
            stream: Yield Responses stream events as dicts — one logical
                response per call: per-turn backend lifecycle events are
                collapsed to one opening ``response.created`` and one
                final terminal event, sequence numbers are reassigned
                monotonically, and ``output_index`` is re-based onto the
                single logical ``output``. The final event is the
                terminal ``response.*`` for the run's status; its
                ``response`` carries the tool outputs in ``items``.
            doc_id: Document ID or list of IDs to scope the conversation.
                Keep it identical across a conversation's calls — the
                targeting block it adds is re-set each call and is part
                of the cached prompt prefix. Local documents: also
                enforced at the tool layer; cloud documents:
                prompt-level targeting only.
            instructions: Appended to the managed system prompt.
            temperature / top_p: Passed through to the model.
            max_turns: Cap on agent turns per call.
            max_output_tokens: Per-call output cap, passed through; it
                bounds each backend call in the agent loop (the way
                max_turns bounds the loop), not the whole run. Echoed in
                the envelope.
            reasoning: Responses reasoning options, forwarded verbatim
                (e.g. ``{"effort": "low", "summary": "auto"}``) — the
                values mean what the backend says they mean. Unset sends
                nothing (the backend's default applies).
            extra_body: Extra request fields beyond this method's
                parameters, merged verbatim into the request body (last,
                so they win). Credentials belong in ``backend``, never
                here.
            extra_headers: Extra HTTP headers merged into each request;
                caller headers win over defaults.
            backend: Connection overrides for this call's backend client,
                merged over the client's ``chat_backend`` (per-call keys
                win). Keys are the openai SDK's client params —
                ``api_key``, ``base_url``, ``organization``, … — passed
                verbatim; unknown keys raise.
        """
        self._require_own_chat("chat(protocol='responses')")
        from .local_chat import run_responses
        return run_responses(
            self, input, model=model, stream=stream, doc_id=doc_id,
            instructions=instructions, temperature=temperature, top_p=top_p,
            max_turns=max_turns, max_output_tokens=max_output_tokens,
            reasoning=reasoning, extra_body=extra_body,
            extra_headers=extra_headers, backend=backend,
        )

    def _messages(
        self,
        messages: Union[str, list[dict[str, Any]]],
        model: str,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        doc_id: Optional[Union[str, list[str]]] = None,
        system: Optional[Union[str, list[dict[str, Any]]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        max_turns: Optional[int] = None,
        thinking: Optional[dict[str, Any]] = None,
        extra_body: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        backend: Optional[dict[str, Any]] = None,
    ) -> Union[dict[str, Any], Iterator[Any]]:
        """
        The engine behind ``chat(protocol="messages")``: document QA over
        the Anthropic Messages protocol, Claude-native.

        Own-model chat only — local mode, or a cloud client constructed
        with ``chat_model=``/``chat=``. Drives Anthropic's /v1/messages
        via the Anthropic SDK's own tool runner (requires
        ``pageindex[anthropic]``; ANTHROPIC_API_KEY selects the
        backend). ``tool_use``/``tool_result`` round-trip is the
        format's native behavior: the response is the
        final message envelope with cross-turn aggregated ``usage`` plus a
        ``messages`` field — the full new turn sequence, valid for verbatim
        append to your history. The managed system prompt carries a
        ``cache_control`` breakpoint, and the request sets the top-level
        ``cache_control`` so each turn re-reads the growing conversation
        from cache — skipped when your own blocks already use the three
        remaining breakpoints (the managed prompt holds the fourth).

        Args:
            messages: Native Messages-format history (including prior
                tool_use/tool_result blocks on round-trip), or a bare query
                string (it becomes a single user message).
            model: Required — there is no cross-vendor default to guess.
            max_tokens: Per-turn output budget the Messages API requires on
                the wire; the default is resolved per model (8192, or 4096
                for the claude-3 generation whose ceiling is lower) so the
                simple call needs only a question, and rises to
                budget_tokens + 8192 when ``thinking`` is enabled (the wire
                requires max_tokens above the budget). Passed through.
            stream: Yield the Anthropic SDK's event stream across turns
                (its native event objects, including SDK-synthesized
                convenience events), one message sequence per turn.
            doc_id: Document ID or list of IDs to scope the conversation.
                Keep it identical across a conversation's calls — the
                targeting block it adds is re-set each call. Local
                documents: also enforced at the tool layer; cloud
                documents: prompt-level targeting only.
            system: Appended after the managed system blocks.
            temperature / top_p / top_k / stop_sequences: Passed through.
            max_turns: Cap on agent turns per call (default 10, like the
                OpenAI surfaces). A truncated run reports
                ``stop_reason: "tool_use"`` and its ``messages`` remain
                valid for continuation.
            thinking: Anthropic thinking configuration, forwarded verbatim
                (e.g. ``{"type": "adaptive"}``) — the values and their
                constraints are the backend's. Unset sends nothing.
            extra_body: Extra request fields beyond this method's
                parameters, merged verbatim into each request body.
                Credentials belong in ``backend``, never here.
            extra_headers: Extra HTTP headers merged into each request
                (e.g. ``anthropic-beta`` feature flags); caller headers
                win over defaults.
            backend: Connection overrides for this call's backend client,
                merged over the client's ``chat_backend`` (per-call keys
                win). Keys are the anthropic SDK's client params —
                ``api_key``, ``base_url``, ``auth_token``, … — passed
                verbatim; unknown keys raise.
        """
        self._require_own_chat("chat(protocol='messages')")
        from .local_chat import run_messages
        return run_messages(
            self, messages, model=model, max_tokens=max_tokens,
            stream=stream, doc_id=doc_id, system=system,
            temperature=temperature, top_p=top_p, top_k=top_k,
            stop_sequences=stop_sequences, max_turns=max_turns,
            thinking=thinking, extra_body=extra_body,
            extra_headers=extra_headers, backend=backend,
        )

    # ---------- DOCUMENT MANAGEMENT ----------

    def get_document(self, doc_id: str) -> dict[str, Any]:
        """
        Get document metadata: {'id', 'name', 'description', 'status',
        'createdAt', 'pageNum', 'folderId'}. Status is one of "queued",
        "processing", "completed", "failed" (local documents are
        always "completed"; local 'folderId' is always None).

        'createdAt' is UTC with no timezone marker, in both modes. To show
        it in the user's timezone::

            from datetime import datetime, timezone
            datetime.fromisoformat(doc["createdAt"]).replace(
                tzinfo=timezone.utc).astimezone()
        """
        return self._api.get_document(doc_id=doc_id)

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        """
        Delete a PageIndex document and all its associated data.

        Returns:
            dict: {'message': 'Document deleted successfully.'}, or an empty
            dict if the cloud API responds with no body.
        """
        return self._api.delete_document(doc_id=doc_id)

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        folder_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        List documents with pagination, newest first.

        Args:
            limit (int): Maximum documents to return (1-100).
            offset (int): Number of documents to skip.
            folder_id (str, optional): Cloud-only folder filter.

        Returns:
            dict: {'documents': [...], 'total', 'limit', 'offset'}.
        """
        return self._api.list_documents(limit=limit, offset=offset, folder_id=folder_id)

    # ---------- AGENT INTEGRATION ----------

    def agent_tools(
        self, include_management: bool = False,
        doc_id: Optional[Union[str, list[str]]] = None,
    ) -> list[Callable[..., str]]:
        """
        Plain functions for any agent framework (LangChain, PydanticAI, ...).
        For the OpenAI / Claude Agent SDKs, prefer ``as_openai_tools()`` /
        ``as_claude_mcp()``.

        Cloud: the full live read tool set, discovered from the PageIndex
        MCP server when this method is called — one function per tool,
        signature and docstring synthesized from the server's schemas, calls
        executed from your process over MCP. Raises PageIndexAPIError if the
        server cannot be reached. Local: the built-in tools over the local
        store (``browse_documents``, ``get_document``,
        ``get_document_structure``, ``get_page_content``).

        Each function takes JSON-serializable arguments, returns a JSON
        string, and reports failures inside that JSON instead of raising —
        except a cloud 401/403, which raises PageIndexAPIError.

        Args:
            include_management (bool): Also expose tools that modify the
                library. Local: adds ``remove_document``. Cloud: the URL
                is the gate — the default serves what the read-only
                endpoint (``?tools=read``) registers; True connects to
                the full ``/mcp`` list (upload, delete, ...).
            doc_id: Local only — restrict the tools to this document ID
                (or list of IDs), enforced at the tool layer: out-of-scope
                lookups return NOT_FOUND. Raises on cloud.
        """
        from .agent_tools import build_agent_tools
        return build_agent_tools(self, include_management, doc_ids=doc_id)

    def as_openai_tools(self, include_management: bool = False,
                        hosted: bool = False,
                        doc_id: Optional[Union[str, list[str]]] = None) -> list:
        """
        Tools for the OpenAI Agents SDK — pass to ``Agent(tools=...)``
        (or ``openai_agent_config()`` for all the Agent slots in one
        call).

        Cloud (default): the full live read tool set (search, folders,
        images — as enabled for your key) as plain function tools,
        discovered from the PageIndex MCP server and executed from your
        process — works with any model backend. Binary tool results
        (e.g. ``get_document_image``) arrive as text placeholder stubs
        on this in-process path. Pass ``hosted=True`` to
        hand the connection to OpenAI instead: one hosted MCP tool, tool
        calls executed server-side (lowest latency; requires an
        OpenAI-hosted model on the Responses API). The framework's own
        ``MCPServerStreamableHttp`` — ``params={"url":
        f"{BASE_URL}/mcp?tools=read", "headers": {"Authorization":
        "Bearer <your PageIndex API key>"}}`` (drop ``?tools=read`` for
        the full tool set) — is the async-native alternative for its
        ``mcp_servers=`` slot.

        Local: the in-process tools, any model backend; ``hosted`` does
        not apply.

        ``openai-agents`` is imported only when this method is called.

        Args:
            include_management (bool): Also expose tools that modify the
                library (delete, upload). Default off: on cloud the URL
                is the gate — in-process and ``hosted=True`` alike
                connect to the read-only endpoint (``/mcp?tools=read``);
                True switches to the full ``/mcp`` list.
            hosted (bool): Cloud only — hand the MCP connection to OpenAI
                for server-side tool execution (OpenAI models only).
            doc_id: Local only — restrict the tools to this document ID
                (or list of IDs), enforced at the tool layer: out-of-scope
                lookups return NOT_FOUND. Raises on cloud.
        """
        from .integrations.openai_agents import build_openai_tools
        return build_openai_tools(self, include_management, hosted,
                                  doc_ids=doc_id)

    def _local_doc_scope(self, doc_id):
        """doc_id for the tool layer: passed through locally (structural
        allowlist), dropped on cloud — its tools take no allowlist, so
        own-model chat and the config helpers target at the prompt level
        only."""
        from .agent_tools import _require_doc_selection
        _require_doc_selection(doc_id)
        if not getattr(self, "api_key", None):
            return doc_id
        return None

    def openai_agent_config(
        self,
        doc_id: Optional[Union[str, list[str]]] = None,
        include_management: bool = False,
        model: Optional[str] = None,
        model_settings: Optional[Any] = None,
        name: str = "PageIndex",
    ) -> dict[str, Any]:
        """
        Document QA ``Agent`` kwargs for the OpenAI Agents SDK in one
        call::

            agent = Agent(**client.openai_agent_config())

        Sugar over the explicit form — ``agent_instructions`` (with
        ``doc_id`` targeting) as the instructions and
        ``as_openai_tools`` as the tools; clients with a configured
        ``chat_model`` — local mode, or cloud with ``chat_model=`` —
        also carry it (a plain cloud client omits ``model`` so the
        framework default applies). To customize further, switch to
        those methods directly. You run this config in your own
        environment, so its model auth comes from there —
        ``chat_backend`` does not travel with it.

        Prompt caching: OpenAI models cache server-side on their own;
        LiteLLM-routed Claude (Anthropic, Bedrock, Vertex) gets its
        cache marks from the bundled ``model_settings``. Pass
        ``model_settings`` here to layer your own on top — your fields
        win and ``extra_args`` merge. Replacing the returned key
        wholesale drops the marks instead.

        Args:
            doc_id: Document ID or list of IDs to target, as in
                ``agent_instructions``. Local: also enforced at the tool
                layer, not just prompted. Cloud: prompt-level targeting.
            include_management (bool): Also expose tools that modify the
                library.
            model: Backend model name; overrides the local default. Same
                grammar as ``chat_model`` (LiteLLM names; bare names are
                OpenAI-compatible shorthand).
            model_settings: Your own ``ModelSettings``, merged on top of
                the bundled cache marks; included verbatim when no marks
                apply.
            name (str): Agent display name; in composition it also seeds
                the SDK-derived handoff and ``as_tool`` names.
        """
        from .agent_tools import build_agent_instructions
        scope = self._local_doc_scope(doc_id)
        config: dict[str, Any] = {
            "name": name,
            "instructions": build_agent_instructions(
                self, doc_id, scoped=scope is not None,
                include_management=include_management),
            "tools": self.as_openai_tools(include_management, doc_id=scope),
        }
        model = model or (self.chat_model if self._local_chat else None)
        if model:
            config["model"] = _agents_sdk_model_name(model)
            if config["model"].startswith("litellm/"):
                # The runner resolves this model through LiteLLM in the
                # caller's process, outside our completion helpers.
                from .utils import (_mute_litellm_bridge_usage_warning,
                                    _repair_litellm_types)
                _repair_litellm_types()
                _mute_litellm_bridge_usage_warning()
                # Marks follow this lane's routing: the SDK strips litellm/
                # and LiteLLM resolves the rest (bare claude-* → Anthropic);
                # names without the prefix ride the SDK's OpenAI provider.
                from .local_chat import _litellm_claude_marks
                extra_args = _litellm_claude_marks(
                    config["model"].removeprefix("litellm/"))
                if extra_args:
                    from agents import ModelSettings
                    config["model_settings"] = ModelSettings(
                        extra_args=extra_args)
        if model_settings is not None:
            marks = config.get("model_settings")
            config["model_settings"] = (marks.resolve(model_settings)
                                        if marks else model_settings)
        return config

    def as_anthropic_tools(self, include_management: bool = False,
                           asynchronous: bool = False,
                           doc_id: Optional[Union[str, list[str]]] = None,
                           ) -> list:
        """
        Runnable tools for the Anthropic SDK's tool runner — pass to
        ``client.beta.messages.tool_runner(tools=...)`` (or
        ``anthropic_runner_config()`` for the whole setup in one call).
        The default flavor is for the sync ``Anthropic`` client; pass
        ``asynchronous=True`` for ``AsyncAnthropic``. For a manual
        ``messages.create`` loop, serialize with
        ``[tool.to_dict() for tool in ...]``.

        Cloud: the full live read tool set (search, folders, images — as
        enabled for your key), discovered from the PageIndex MCP server
        and executed from your process; the server's input schemas pass
        through verbatim (MCP and the Messages API share the schema
        shape), and binary tool results (e.g. ``get_document_image``)
        arrive as text placeholder stubs on this in-process path. The
        server-side alternative is the Messages API's beta
        MCP connector — ``mcp_servers=[{"type": "url", "name":
        "pageindex", "url": f"{BASE_URL}/mcp?tools=read",
        "authorization_token": <your PageIndex API key>}]`` (drop
        ``?tools=read`` for the full tool set) — with no client-side
        tools involved. Local: the in-process tools — the same set
        ``chat(protocol="messages")`` runs internally.

        Requires ``anthropic>=0.108.0``
        (``pip install 'pageindex[anthropic]'``), imported only when this
        method is called.

        Args:
            include_management (bool): Also expose tools that modify the
                library. Local: adds ``remove_document``. Cloud: the URL
                is the gate — the default serves what the read-only
                endpoint (``?tools=read``) registers; True connects to
                the full ``/mcp`` list (upload, delete, ...).
            asynchronous (bool): Build ``beta_async_tool`` runnables for
                ``AsyncAnthropic`` (each tool call runs in a worker
                thread, keeping blocking I/O off your event loop). The
                sync and async runners each accept only their own flavor.
            doc_id: Local only — restrict the tools to this document ID
                (or list of IDs), enforced at the tool layer: out-of-scope
                lookups return NOT_FOUND. Raises on cloud.
        """
        from .integrations.anthropic_sdk import build_anthropic_tools
        return build_anthropic_tools(self, include_management, asynchronous,
                                     doc_ids=doc_id)

    def anthropic_runner_config(
        self,
        model: str,
        doc_id: Optional[Union[str, list[str]]] = None,
        include_management: bool = False,
        asynchronous: bool = False,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
        thinking: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Document QA ``tool_runner`` kwargs for the Anthropic SDK in one
        call — only your ``messages`` remain::

            runner = anthropic_client.beta.messages.tool_runner(
                **client.anthropic_runner_config(model="claude-sonnet-4-5"),
                messages=[{"role": "user", "content": "..."}],
            )

        Sugar over the explicit form — ``agent_instructions`` (with
        ``doc_id`` targeting) as the system prompt and
        ``as_anthropic_tools`` as the tools — plus the ``max_tokens``
        default and 10-turn ``max_iterations`` bound
        ``chat(protocol="messages")`` uses,
        and a top-level ``cache_control`` so each loop turn re-reads the
        growing prompt from cache (pop the key if you place your own
        breakpoints — the API allows four). Unlike the chat lane,
        ``system`` here is the bare instructions string, without the chat
        header or its block-level breakpoint. To customize further,
        switch to those methods directly.

        Args:
            model: Backend model name (also resolves the ``max_tokens``
                default).
            doc_id: Document ID or list of IDs to target, as in
                ``agent_instructions``. Local: also enforced at the tool
                layer, not just prompted. Cloud: prompt-level targeting.
            include_management (bool): Also expose tools that modify the
                library.
            asynchronous (bool): Build async runnables for
                ``AsyncAnthropic``.
            max_tokens: Per-turn output budget; default resolved per
                model.
            max_turns: Agent-loop bound; default 10.
            thinking: Anthropic ``thinking`` config, included in the
                kwargs; an enabled budget also lifts the ``max_tokens``
                default above it. Pass it here, not alongside the
                unpacked config, so the default stays valid.
        """
        from .agent_tools import build_agent_instructions
        from .local_chat import _default_max_tokens, _validate_max_turns
        _validate_max_turns(max_turns)
        scope = self._local_doc_scope(doc_id)
        return {
            "model": model,
            "max_tokens": (max_tokens if max_tokens is not None
                           else _default_max_tokens(model, thinking)),
            "system": build_agent_instructions(
                self, doc_id, scoped=scope is not None,
                include_management=include_management),
            "tools": self.as_anthropic_tools(include_management, asynchronous,
                                             doc_id=scope),
            "max_iterations": max_turns if max_turns is not None else 10,
            **({"thinking": thinking} if thinking is not None else {}),
            "cache_control": {"type": "ephemeral"},
        }

    def as_claude_mcp(self, include_management: bool = False,
                      doc_id: Optional[Union[str, list[str]]] = None,
                      server_name: str = "pageindex"):
        """
        ``mcp_servers`` entry for the Claude Agent SDK.

        Cloud: returns the remote PageIndex MCP config.
        ``include_management`` picks the endpoint, so the URL itself is
        the gate — the default connects to the read-only endpoint
        (``/mcp?tools=read``: the server registers only read-only tools),
        ``True`` connects to the full tool set. Local: returns an
        in-process SDK MCP server exposing the agent tools, gated the
        same way at registration (requires ``claude-agent-sdk``;
        ``pip install 'pageindex[claude]'``). ``doc_id`` (local only)
        restricts those tools to that document ID (or list), enforced at
        the tool layer; it raises on cloud.
        ``server_name`` names the in-process server — match it to the key
        you register the entry under (cloud entries carry no name).

        Cloud hosts that surface MCP server instructions receive the same
        guidance ``agent_instructions()`` returns natively — passing both
        duplicates the text (harmless). ``system_prompt`` stays the
        recommended channel: it is guaranteed delivery, carries ``doc_id``
        targeting, and is the only channel local mode has.

        Usage (or ``claude_agent_config()`` for all three slots in one
        call)::

            options = ClaudeAgentOptions(
                system_prompt=client.agent_instructions(),
                mcp_servers={"pageindex": client.as_claude_mcp()},
                # Pre-approval only — the server itself is already gated.
                allowed_tools=["mcp__pageindex"],
            )
        """
        from .integrations.claude_agent_sdk import build_claude_mcp
        return build_claude_mcp(self, include_management, doc_ids=doc_id,
                                server_name=server_name)

    def claude_agent_config(
        self,
        doc_id: Optional[Union[str, list[str]]] = None,
        include_management: bool = False,
        server_name: str = "pageindex",
    ) -> dict[str, Any]:
        """
        Document QA ``ClaudeAgentOptions`` kwargs in one call::

            options = ClaudeAgentOptions(**client.claude_agent_config())

        Sugar over the explicit form — the managed system prompt
        (``agent_instructions``) and the server entry (``as_claude_mcp``,
        itself the tool gate) with its ``allowed_tools`` pre-approval,
        one ``include_management`` and ``server_name`` applied
        everywhere. To customize (your own system prompt, extra
        servers), switch to those methods directly.

        Args:
            doc_id: Document ID or list of IDs to target, as in
                ``agent_instructions``. Local: also enforced at the tool
                layer, not just prompted. Cloud: prompt-level targeting.
            include_management (bool): Also allow tools that modify the
                library.
            server_name (str): Key the server is registered under;
                locally also the name the SDK server declares.
        """
        from .agent_tools import build_agent_instructions
        scope = self._local_doc_scope(doc_id)
        return {
            "system_prompt": build_agent_instructions(
                self, doc_id, scoped=scope is not None,
                include_management=include_management),
            "mcp_servers": {server_name: self.as_claude_mcp(
                include_management, doc_id=scope, server_name=server_name)},
            # Pre-approval only — the server itself is already gated (the
            # read-only endpoint on cloud, the registered set locally).
            "allowed_tools": [f"mcp__{server_name}"],
        }

    def agent_instructions(
        self, doc_id: Optional[Union[str, list[str]]] = None,
        include_management: bool = False,
    ) -> str:
        """
        Orchestration guidance for document QA agents — pass as the agent's
        system prompt (or append to your own).

        Cloud: the live instructions the PageIndex MCP server serves for
        your key's tool set, fetched over the same session as
        ``agent_tools()`` — server-side guidance updates arrive without an
        SDK release. Raises PageIndexAPIError if the server cannot be
        reached. Local: the built-in guidance for the in-process tools.

        With ``doc_id`` (str or list, same shape as ``chat_completions``),
        appends the target documents' names and metadata and directs the
        agent to work within them. Raises PageIndexAPIError if a doc_id
        does not exist, or if its name is shadowed by a newer same-name
        document — the name-addressed tools could not reach it (the
        ``*_agent_config`` bundles, whose tools carry the doc_id scope,
        relax this to duplicates within the targeted set).

        ``include_management``: fetch the guidance for the full tool set,
        matching tools built with ``include_management=True`` (cloud;
        local guidance is a single set).
        """
        from .agent_tools import build_agent_instructions
        return build_agent_instructions(
            self, doc_id, include_management=include_management)

    # ---------- FOLDER MANAGEMENT ----------

    def create_folder(
        self,
        name: str,
        description: Optional[str] = None,
        parent_folder_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create a folder (workspace). Cloud-only: local mode raises
        PageIndexAPIError.
        """
        return self._require_cloud(
            "create_folder is cloud-only — folders are not supported in local "
            "mode. Create the client with an api_key to use folders."
        ).create_folder(
            name=name, description=description, parent_folder_id=parent_folder_id,
        )

    def list_folders(self, parent_folder_id: Optional[str] = None) -> dict[str, Any]:
        """
        List folders. Cloud-only: local mode raises PageIndexAPIError.
        """
        return self._require_cloud(
            "list_folders is cloud-only — folders are not supported in local "
            "mode. Create the client with an api_key to use folders."
        ).list_folders(
            parent_folder_id=parent_folder_id,
        )

    def _require_cloud(self, message: str):
        from .cloud_api import CloudAPI
        if not isinstance(self._api, CloudAPI):
            raise PageIndexAPIError(message)
        return self._api


class PageIndexCloudClient(PageIndexClient):
    """Cloud mode — the class name says cloud, so the key may come from
    the environment: ``PageIndexCloudClient()`` reads PAGEINDEX_API_KEY.
    The shortest env-key cloud spelling."""

    _pin = "cloud"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        index: Optional[Union[Mapping[str, Any], str]] = None,
        chat: Optional[Union[Mapping[str, Any], str]] = None,
        chat_model: Optional[str] = None,
        retrieve_model: Optional[str] = None,
        chat_backend: Optional[dict[str, Any]] = None,
    ):
        if index is None:
            if api_key is None:
                # .env keys arrive via utils' import-time load_dotenv().
                from . import utils  # noqa: F401
                api_key = os.environ.get("PAGEINDEX_API_KEY")
            if not api_key:
                raise PageIndexAPIError(
                    "PageIndexCloudClient requires a PageIndex API key — "
                    "pass api_key=..., or export PAGEINDEX_API_KEY. Get one "
                    "at https://developer.pageindex.ai/api-keys."
                )
        super().__init__(api_key, index=index, chat=chat,
                         chat_model=chat_model, retrieve_model=retrieve_model,
                         chat_backend=chat_backend)


class PageIndexLocalClient(PageIndexClient):
    """Local mode — no api_key parameter, no cloud access."""

    _pin = "local"

    def __init__(
        self,
        *,
        index: Optional[Union[Mapping[str, Any], str]] = None,
        chat: Optional[Union[Mapping[str, Any], str]] = None,
        index_model: Optional[str] = None,
        chat_model: Optional[str] = None,
        model: Optional[str] = None,
        summary_model: Optional[str] = None,
        retrieve_model: Optional[str] = None,
        storage_path: Optional[Union[str, os.PathLike[str]]] = None,
        index_backend: Optional[dict[str, Any]] = None,
        chat_backend: Optional[dict[str, Any]] = None,
    ):
        super().__init__(None, index=index, chat=chat,
                         index_model=index_model, chat_model=chat_model,
                         model=model, summary_model=summary_model,
                         retrieve_model=retrieve_model, storage_path=storage_path,
                         index_backend=index_backend, chat_backend=chat_backend)
