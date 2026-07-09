import re
from pathlib import Path
from .protocol import ContentNode, ParsedDocument
from ..tokens import count_tokens


class MarkdownParser:
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        path = Path(file_path)
        model = kwargs.get("model")

        # utf-8-sig strips a leading BOM if present (common from Windows
        # editors/exporters) and is otherwise identical to plain utf-8. Without
        # it, a BOM-prefixed first line fails the header regex below (the BOM
        # isn't whitespace, so .strip() doesn't remove it), misclassifying the
        # document's first heading as unrecognized preamble text.
        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read()

        lines = content.split("\n")
        headers = self._extract_headers(lines)
        nodes = self._build_nodes(headers, lines, model, doc_title=path.stem)

        return ParsedDocument(doc_name=path.stem, nodes=nodes)

    def _extract_headers(self, lines: list[str]) -> list[dict]:
        header_pattern = r"^(#{1,6})\s+(.+)$"
        # CommonMark allows both backtick and tilde fences; only recognizing
        # backticks let a '#'-prefixed line inside a ~~~-fenced block (e.g. a
        # shell comment in a code sample) be misparsed as a real heading.
        code_block_pattern = r"^(?:```|~~~)"
        headers = []
        in_code_block = False

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if re.match(code_block_pattern, stripped):
                in_code_block = not in_code_block
                continue
            if not in_code_block and stripped:
                match = re.match(header_pattern, stripped)
                if match:
                    headers.append({
                        "title": match.group(2).strip(),
                        "level": len(match.group(1)),
                        "line_num": line_num,
                    })
        return headers

    def _build_nodes(self, headers: list[dict], lines: list[str], model: str | None,
                     doc_title: str = "Document") -> list[ContentNode]:
        nodes = []

        # A file with no headings at all still has content — index it as a
        # single node instead of producing zero nodes (which would push an
        # empty page list into the LLM pipeline).
        if not headers:
            text = "\n".join(lines).strip()
            if text:
                nodes.append(ContentNode(
                    content=text,
                    tokens=count_tokens(text, model=model),
                    title=doc_title,
                    index=1,
                    level=1,
                ))
            return nodes

        # Content before the first heading (abstract, preamble) would
        # otherwise be silently dropped and become unretrievable.
        preamble = "\n".join(lines[: headers[0]["line_num"] - 1]).strip()
        if preamble:
            nodes.append(ContentNode(
                content=preamble,
                tokens=count_tokens(preamble, model=model),
                title=doc_title,
                index=1,
                level=headers[0]["level"],
            ))

        for i, header in enumerate(headers):
            start = header["line_num"] - 1
            end = headers[i + 1]["line_num"] - 1 if i + 1 < len(headers) else len(lines)
            text = "\n".join(lines[start:end]).strip()
            tokens = count_tokens(text, model=model)
            nodes.append(ContentNode(
                content=text,
                tokens=tokens,
                title=header["title"],
                index=header["line_num"],
                level=header["level"],
            ))
        return nodes
