from pydantic import BaseModel, Field
from typing import Optional, List, Literal


# Response schemas for simple yes/no checks with thinking
class TitleAppearanceResponse(BaseModel):
    """Response for checking if a title appears in a page."""
    thinking: str = Field(description="Explanation of why the section appears or starts in the page_text")
    answer: Literal["yes", "no"] = Field(description="Whether the section appears or starts in the page_text")


class TitleAppearanceInStartResponse(BaseModel):
    """Response for checking if a title appears at the start of a page."""
    thinking: str = Field(description="Explanation of why the section starts at the beginning")
    start_begin: Literal["yes", "no"] = Field(description="Whether the section starts at the beginning of the page")


class TocDetectedResponse(BaseModel):
    """Response for detecting table of contents in text."""
    thinking: str = Field(description="Explanation of why there is or isn't a table of contents")
    toc_detected: Literal["yes", "no"] = Field(description="Whether a table of contents was detected")


class CompletedResponse(BaseModel):
    """Response for checking if extraction/transformation is complete."""
    thinking: str = Field(description="Explanation of why the operation is complete or not")
    completed: Literal["yes", "no"] = Field(description="Whether the operation is complete")


class PageIndexGivenResponse(BaseModel):
    """Response for detecting if page numbers are given in TOC."""
    thinking: str = Field(description="Explanation of whether page numbers are present")
    page_index_given_in_toc: Literal["yes", "no"] = Field(description="Whether page indices are given in the table of contents")


# TOC item schemas
class TocItem(BaseModel):
    """Table of contents item with physical index."""
    structure: Optional[str] = Field(None, description="Structure index like '1', '1.1', '1.2.1', etc.")
    title: str = Field(description="Title of the section")
    physical_index: Optional[str] = Field(None, description="Physical index in format '<physical_index_X>' or null")


class TocWithPage(BaseModel):
    """Table of contents item with page number."""
    structure: Optional[str] = Field(None, description="Structure index like '1', '1.1', '1.2.1', etc.")
    title: str = Field(description="Title of the section")
    page: Optional[int] = Field(None, description="Page number as integer or null")


class TableOfContents(BaseModel):
    """Root object for table of contents with page numbers."""
    table_of_contents: List[TocWithPage] = Field(description="List of table of contents items")


class TocItemWithStart(BaseModel):
    """Table of contents item with start indicator."""
    structure: Optional[str] = Field(None, description="Structure index like '1', '1.1', '1.2.1', etc.")
    title: str = Field(description="Title of the section")
    start: Literal["yes", "no"] = Field(description="Whether the section starts in the given document part")
    physical_index: Optional[str] = Field(None, description="Physical index in format '<physical_index_X>' or null")


class PhysicalIndexResponse(BaseModel):
    """Response with physical index location."""
    thinking: str = Field(description="Explanation of which page contains the start of the section")
    physical_index: str = Field(description="Physical index in format '<physical_index_X>'")


# List response types
class TocItemList(BaseModel):
    """List of table of contents items."""
    items: List[TocItem] = Field(description="List of TOC items")
    
    def __iter__(self):
        return iter(self.items)
    
    def __getitem__(self, index):
        return self.items[index]


class TocItemWithStartList(BaseModel):
    """List of table of contents items with start indicators."""
    items: List[TocItemWithStart] = Field(description="List of TOC items with start indicators")
    
    def __iter__(self):
        return iter(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
