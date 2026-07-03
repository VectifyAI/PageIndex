from pydantic import BaseModel, Field, AliasChoices
from typing import Literal, Optional, List, Union

class TitleAppearance(BaseModel):
    thinking: str = Field(
        validation_alias=AliasChoices("thinking", "reason", "explanation", "rationale"),
        description="why do you think the section appears or starts in the page_text"
    )
    answer: Literal["yes", "no"] = Field(description="yes if the section appears or starts in the page_text, no otherwise")

class TitleAppearanceInStart(BaseModel):
    thinking: str = Field(
        validation_alias=AliasChoices("thinking", "reason", "explanation", "rationale"),
        description="why do you think the section appears or starts in the page_text"
    )
    start_begin: Literal["yes", "no"] = Field(description="yes if the section starts in the beginning of the page_text, no otherwise")

class TOCDetection(BaseModel):
    thinking: str = Field(
        validation_alias=AliasChoices("thinking", "reason", "explanation", "rationale"),
        description="why do you think there is or isn't a table of content in the given text"
    )
    toc_detected: Literal["yes", "no"] = Field(description="yes if a table of contents is detected, no otherwise")

class TOCCompletionCheck(BaseModel):
    thinking: str = Field(
        validation_alias=AliasChoices("thinking", "reason", "explanation", "rationale"),
        description="why do you think the table of contents is complete or not"
    )
    completed: Literal["yes", "no"] = Field(description="yes if the table of contents is complete, no otherwise")

class PageIndexDetection(BaseModel):
    thinking: str = Field(
        validation_alias=AliasChoices("thinking", "reason", "explanation", "rationale"),
        description="why do you think there are page numbers/indices given within the table of contents"
    )
    page_index_given_in_toc: Literal["yes", "no"] = Field(description="yes if page numbers/indices are detected within the table of contents, no otherwise")

class TOCItem(BaseModel):
    structure: Optional[str] = Field(
        default=None,
        description="Numeric hierarchy index e.g. '1', '1.1', '1.2', or None"
    )
    title: str = Field(description="Title of the section")
    page: Optional[Union[int, str]] = Field(description="Page number from the table of contents, or None")

class TOCTransformation(BaseModel):
    table_of_contents: List[TOCItem] = Field(description="List of all sections extracted from the table of contents")

class TOCIndexItem(BaseModel):
    structure: Optional[str] = Field(
        default=None,
        description="Numeric hierarchy index e.g. '1', '1.1', or None"
    )
    title: str = Field(description="Title of the section")
    physical_index: Optional[str] = Field(
        default=None,
        description="Physical index tag in format <physical_index_X>, or None if section not found in provided pages",
        pattern=r"^<physical_index_\d+>$"
    )

class PageNumberItem(BaseModel):
    structure: Optional[str] = Field(
        default=None,
        description="Numeric hierarchy index e.g. '1', '1.1', or None"
    )
    title: str = Field(description="Title of the section")
    start: Literal["yes", "no"] = Field(description="yes if the section starts in the current partial document, no otherwise")
    physical_index: Optional[str] = Field(
        default=None,
        description="Physical index tag in format <physical_index_X> if start is yes, None otherwise",
        pattern=r"^<physical_index_\d+>$"
    )

class PhysicalIndexResult(BaseModel):
    thinking: str = Field(
        validation_alias=AliasChoices("thinking", "reason", "explanation", "rationale"),
        description="explain which page, started and closed by <physical_index_X>, contains the start of this section"
    )
    physical_index: str = Field(description="The physical index tag of the start page, in format <physical_index_X>", pattern=r"^<physical_index_\d+>$")

class TOCStructureItem(BaseModel):
    structure: str = Field(
        default=None,
        description="Numeric hierarchy index e.g. '1', '1.1', '1.2', or None"
    )
    title: str = Field(description="Original title of the section, only fix space inconsistency")
    physical_index: str = Field(
        description="Physical index tag of the section start, in format <physical_index_X>",
        pattern=r"^<physical_index_\d+>$"
    )

class TOCStructureList(BaseModel):
    items: List[TOCStructureItem]

class TOCIndexList(BaseModel):
    items: List[TOCIndexItem]

class PageNumberList(BaseModel):
    items: List[PageNumberItem]