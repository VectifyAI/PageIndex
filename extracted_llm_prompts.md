# PageIndex LLM Prompts and System Messages

This document contains all the prompts and system messages used for LLM interactions in the PageIndex repository.

## Table of Contents
1. [Table of Contents Detection & Extraction](#toc-detection--extraction)
2. [Structure Generation](#structure-generation)
3. [Page Number & Index Operations](#page-number--index-operations)
4. [Verification & Validation](#verification--validation)
5. [Error Correction](#error-correction)
6. [Summary & Description Generation](#summary--description-generation)

---

## TOC Detection & Extraction

### 1. TOC Detector (Single Page)
**Function:** `toc_detector_single_page()`
**File:** `pageindex/page_index.py:105`

```
Your job is to detect if there is a table of content provided in the given text.

Given text: {content}

return the following JSON format:
{
    "thinking": <why do you think there is a table of content in the given text>
    "toc_detected": "<yes or no>",
}

Directly return the final JSON structure. Do not output anything else.
Please note: abstract,summary, notation list, figure list, table list, etc. are not table of contents.
```

### 2. Extract TOC Content
**Function:** `extract_toc_content()`
**File:** `pageindex/page_index.py:161`

```
Your job is to extract the full table of contents from the given text, replace ... with :

Given text: {content}

Directly return the full table of contents content. Do not output anything else.
```

**Follow-up prompt** (if incomplete):
```
please continue the generation of table of contents , directly output the remaining part of the structure
```

### 3. Detect Page Index in TOC
**Function:** `detect_page_index()`
**File:** `pageindex/page_index.py:201`

```
You will be given a table of contents.

Your job is to detect if there are page numbers/indices given within the table of contents.

Given text: {toc_content}

Reply format:
{
    "thinking": <why do you think there are page numbers/indices given within the table of contents>
    "page_index_given_in_toc": "<yes or no>"
}
Directly return the final JSON structure. Do not output anything else.
```

---

## Structure Generation

### 4. TOC Transformer (Convert to JSON)
**Function:** `toc_transformer()`
**File:** `pageindex/page_index.py:272`

```
You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

The response should be in the following JSON format:
{
table_of_contents: [
    {
        "structure": <structure index, "x.x.x" or None> (string),
        "title": <title of the section>,
        "page": <page number or None>,
    },
    ...
    ],
}
You should transform the full table of contents in one go.
Directly return the final JSON structure, do not output anything else.
```

**Follow-up prompt** (if incomplete):
```
Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.
The response should be in the following JSON format:

The raw table of contents json structure is:
{toc_content}

The incomplete transformed table of contents json structure is:
{last_complete}

Please continue the json structure, directly output the remaining part of the json structure.
```

### 5. Generate TOC Init (From Raw Text)
**Function:** `generate_toc_init()`
**File:** `pageindex/page_index.py:536`

```
You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

For the title, you need to extract the original title from the text, only fix the space inconsistency.

The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X.

For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

The response should be in the following format.
    [
        {
            "structure": <structure index, "x.x.x"> (string),
            "title": <title of the section, keep the original title>,
            "physical_index": "<physical_index_X> (keep the format)"
        },

    ],


Directly return the final JSON structure. Do not output anything else.
```

### 6. Generate TOC Continue (Incremental)
**Function:** `generate_toc_continue()`
**File:** `pageindex/page_index.py:501`

```
You are an expert in extracting hierarchical tree structure.
You are given a tree structure of the previous part and the text of the current part.
Your task is to continue the tree structure from the previous part to include the current part.

The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

For the title, you need to extract the original title from the text, only fix the space inconsistency.

The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X.

For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

The response should be in the following format.
    [
        {
            "structure": <structure index, "x.x.x"> (string),
            "title": <title of the section, keep the original title>,
            "physical_index": "<physical_index_X> (keep the format)"
        },
        ...
    ]

Directly return the additional part of the final JSON structure. Do not output anything else.
```

---

## Page Number & Index Operations

### 7. TOC Index Extractor (Add Physical Index)
**Function:** `toc_index_extractor()`
**File:** `pageindex/page_index.py:242`

```
You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

The response should be in the following JSON format:
[
    {
        "structure": <structure index, "x.x.x" or None> (string),
        "title": <title of the section>,
        "physical_index": "<physical_index_X>" (keep the format)
    },
    ...
]

Only add the physical_index to the sections that are in the provided pages.
If the section is not in the provided pages, do not add the physical_index to it.
Directly return the final JSON structure. Do not output anything else.
```

### 8. Add Page Number to TOC
**Function:** `add_page_number_to_toc()`
**File:** `pageindex/page_index.py:454`

```
You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

The response should be in the following format.
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "start": "<yes or no>",
            "physical_index": "<physical_index_X> (keep the format)" or None
        },
        ...
    ]
The given structure contains the result of the previous part, you need to fill the result of the current part, do not change the previous result.
Directly return the final JSON structure. Do not output anything else.
```

---

## Verification & Validation

### 9. Check Title Appearance
**Function:** `check_title_appearance()`
**File:** `pageindex/page_index.py:23`

```
Your job is to check if the given section appears or starts in the given page_text.

Note: do fuzzy matching, ignore any space inconsistency in the page_text.

The given section title is {title}.
The given page_text is {page_text}.

Reply format:
{

    "thinking": <why do you think the section appears or starts in the page_text>
    "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
}
Directly return the final JSON structure. Do not output anything else.
```

### 10. Check Title Appearance in Start
**Function:** `check_title_appearance_in_start()`
**File:** `pageindex/page_index.py:49`

```
You will be given the current section title and the current page_text.
Your job is to check if the current section starts in the beginning of the given page_text.
If there are other contents before the current section title, then the current section does not start in the beginning of the given page_text.
If the current section title is the first content in the given page_text, then the current section starts in the beginning of the given page_text.

Note: do fuzzy matching, ignore any space inconsistency in the page_text.

The given section title is {title}.
The given page_text is {page_text}.

reply format:
{
    "thinking": <why do you think the section appears or starts in the page_text>
    "start_begin": "yes or no" (yes if the section starts in the beginning of the page_text, no otherwise)
}
Directly return the final JSON structure. Do not output anything else.
```

### 11. Check TOC Extraction Completeness
**Function:** `check_if_toc_extraction_is_complete()`
**File:** `pageindex/page_index.py:126`

```
You are given a partial document  and a  table of contents.
Your job is to check if the  table of contents is complete, which it contains all the main sections in the partial document.

Reply format:
{
    "thinking": <why do you think the table of contents is complete or not>
    "completed": "yes" or "no"
}
Directly return the final JSON structure. Do not output anything else.
```

### 12. Check TOC Transformation Completeness
**Function:** `check_if_toc_transformation_is_complete()`
**File:** `pageindex/page_index.py:144`

```
You are given a raw table of contents and a  table of contents.
Your job is to check if the  table of contents is complete.

Reply format:
{
    "thinking": <why do you think the cleaned table of contents is complete or not>
    "completed": "yes" or "no"
}
Directly return the final JSON structure. Do not output anything else.
```

---

## Error Correction

### 13. Single TOC Item Index Fixer
**Function:** `single_toc_item_index_fixer()`
**File:** `pageindex/page_index.py:733`

```
You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

Reply in a JSON format:
{
    "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
    "physical_index": "<physical_index_X>" (keep the format)
}
Directly return the final JSON structure. Do not output anything else.
```

---

## Summary & Description Generation

### 14. Generate Node Summary
**Function:** `generate_node_summary()`
**File:** `pageindex/utils.py:606`

```
You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

Partial Document Text: {node['text']}

Directly return the description, do not include any other text.
```

### 15. Generate Document Description
**Function:** `generate_doc_description()`
**File:** `pageindex/utils.py:650`

```
Your are an expert in generating descriptions for a document.
You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.

Document Structure: {structure}

Directly return the description, do not include any other text.
```

---

## Key Patterns & Instructions

### Common Instructions Across Prompts:
1. **JSON Output Format**: Most prompts require strict JSON format output
2. **Direct Response**: "Directly return the final JSON structure. Do not output anything else."
3. **Thinking Field**: Many prompts include a "thinking" field for chain-of-thought reasoning
4. **Physical Index Tags**: Use of `<physical_index_X>` tags to mark page boundaries
5. **Fuzzy Matching**: Instructions to ignore space inconsistencies
6. **Structure Index System**: Hierarchical numbering (1, 1.1, 1.2, 2, 2.1, etc.)

### Response Validation:
- Prompts often include completion checks
- Multiple retry mechanisms for incomplete responses
- Incremental continuation prompts for long outputs

### Temperature Setting:
All API calls use `temperature=0` for deterministic outputs (see `utils.py:43, 75, 98`)

---

## API Configuration

**Model Used:** `gpt-4o-2024-11-20` (default from `config.yaml`)
**Temperature:** `0` (for deterministic outputs)
**Max Retries:** `10` attempts with exponential backoff
**API Key:** Loaded from `.env` file (`CHATGPT_API_KEY`)

---

## Notes

1. All prompts are designed for **OpenAI GPT models**
2. The system uses **async processing** for parallel API calls
3. Prompts are optimized for **JSON structured outputs**
4. **Chain-of-thought prompting** is used extensively (thinking field)
5. The system has **self-correction mechanisms** (verification + retry loops)

---

*Extracted from PageIndex repository - A reasoning-based RAG system for long document analysis*
