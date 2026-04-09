import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PageIndexConfig:
    pdf_path: Optional[str] = None
    md_path: Optional[str] = None
    model: Optional[str] = None
    toc_check_pages: Optional[int] = None
    max_pages_per_node: Optional[int] = None
    max_tokens_per_node: Optional[int] = None
    add_node_id: Optional[bool] = None
    add_node_summary: Optional[bool] = None
    add_doc_description: Optional[bool] = None
    add_node_text: Optional[bool] = None
    if_thinning: bool = False
    thinning_threshold: int = 5000
    summary_token_threshold: int = 200
    output_dir: str = "./results"


class PageIndex:
    """Public API for generating structure from PDF or Markdown documents."""

    def __init__(self, config: PageIndexConfig):
        self.config = config
        self._doc_kind: Optional[str] = None
        self._doc_path: Optional[str] = None
        self._validate_and_resolve_input()

    def run(self) -> Dict[str, Any]:
        if self._is_pdf():
            return self._process_pdf()
        if self._is_markdown():
            return self._process_markdown()
        raise ValueError("Input file must be a PDF or Markdown document.")

    def run_and_save(self) -> str:
        result = self.run()
        output_file = self._build_output_file()
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
        return output_file

    def _process_pdf(self) -> Dict[str, Any]:
        from pageindex.page_index import page_index_main
        from pageindex.utils import ConfigLoader

        user_opt = {
            "model": self.config.model,
            "toc_check_page_num": self.config.toc_check_pages,
            "max_page_num_each_node": self.config.max_pages_per_node,
            "max_token_num_each_node": self.config.max_tokens_per_node,
            "if_add_node_id": self._to_yes_no(self.config.add_node_id),
            "if_add_node_summary": self._to_yes_no(self.config.add_node_summary),
            "if_add_doc_description": self._to_yes_no(self.config.add_doc_description),
            "if_add_node_text": self._to_yes_no(self.config.add_node_text),
        }
        opt = ConfigLoader().load({key: value for key, value in user_opt.items() if value is not None})
        return page_index_main(self._doc_path, opt)

    def _process_markdown(self) -> Dict[str, Any]:
        from pageindex.page_index_md import md_to_tree
        from pageindex.utils import ConfigLoader

        user_opt = {
            "model": self.config.model,
            "if_add_node_summary": self._to_yes_no(self.config.add_node_summary),
            "if_add_doc_description": self._to_yes_no(self.config.add_doc_description),
            "if_add_node_text": self._to_yes_no(self.config.add_node_text),
            "if_add_node_id": self._to_yes_no(self.config.add_node_id),
        }
        opt = ConfigLoader().load({key: value for key, value in user_opt.items() if value is not None})

        return asyncio.run(
            md_to_tree(
                md_path=self._doc_path,
                if_thinning=self.config.if_thinning,
                min_token_threshold=self.config.thinning_threshold,
                if_add_node_summary=opt.if_add_node_summary,
                summary_token_threshold=self.config.summary_token_threshold,
                model=opt.model,
                if_add_doc_description=opt.if_add_doc_description,
                if_add_node_text=opt.if_add_node_text,
                if_add_node_id=opt.if_add_node_id,
            )
        )

    def _build_output_file(self) -> str:
        base_name = os.path.splitext(os.path.basename(self._doc_path))[0]
        return os.path.join(self.config.output_dir, f"{base_name}_structure.json")

    def _validate_and_resolve_input(self) -> None:
        pdf_path = self.config.pdf_path
        md_path = self.config.md_path
        if not pdf_path and not md_path:
            raise ValueError("Either --pdf_path or --md_path must be specified")
        if pdf_path and md_path:
            raise ValueError("Only one of --pdf_path or --md_path can be specified")

        if pdf_path:
            self._validate_pdf(pdf_path)
            self._doc_kind = "pdf"
            self._doc_path = pdf_path
            return

        self._validate_markdown(md_path)
        self._doc_kind = "markdown"
        self._doc_path = md_path

    @staticmethod
    def _validate_pdf(path: str) -> None:
        if not path.lower().endswith(".pdf"):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(path):
            raise ValueError(f"PDF file not found: {path}")

    @staticmethod
    def _validate_markdown(path: str) -> None:
        if not path.lower().endswith((".md", ".markdown")):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(path):
            raise ValueError(f"Markdown file not found: {path}")

    def _is_pdf(self) -> bool:
        return self._doc_kind == "pdf"

    def _is_markdown(self) -> bool:
        return self._doc_kind == "markdown"

    @staticmethod
    def _to_yes_no(value: Optional[bool]) -> Optional[str]:
        if value is None:
            return None
        return "yes" if value else "no"
