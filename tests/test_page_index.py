import asyncio
import importlib
from types import SimpleNamespace

import pytest

pi = importlib.import_module("pageindex.page_index")


class Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def error(self, message):
        self.messages.append(message)


def test_toc_llm_wrappers(monkeypatch):
    monkeypatch.setattr(pi, "llm_completion", lambda **_: '{"toc_detected": "yes"}')
    assert pi.toc_detector_single_page("contents") == "yes"

    monkeypatch.setattr(pi, "llm_completion", lambda **_: '{"completed": "no"}')
    assert pi.check_if_toc_extraction_is_complete("doc", "toc") == "no"
    assert pi.check_if_toc_transformation_is_complete("raw", "clean") == "no"

    monkeypatch.setattr(pi, "llm_completion", lambda **_: '{"page_index_given_in_toc": "yes"}')
    assert pi.detect_page_index("toc") == "yes"
    assert pi.toc_index_extractor([], "content") == {"page_index_given_in_toc": "yes"}


def test_check_title_appearance_paths(monkeypatch):
    item = {"title": "A", "list_index": 0}
    assert asyncio.run(pi.check_title_appearance(item, [("text", 1)])) == {
        "list_index": 0,
        "answer": "no",
        "title": "A",
        "page_number": None,
    }

    async def fake_acompletion(**_kwargs):
        return '{"answer": "yes"}'

    monkeypatch.setattr(pi, "llm_acompletion", fake_acompletion)
    item = {"title": "A", "list_index": 1, "physical_index": 1}
    assert asyncio.run(pi.check_title_appearance(item, [("A text", 1)]))["answer"] == "yes"

    async def no_answer(**_kwargs):
        return "{}"

    monkeypatch.setattr(pi, "llm_acompletion", no_answer)
    assert asyncio.run(pi.check_title_appearance(item, [("A text", 1)]))["answer"] == "no"


def test_check_title_start_concurrent_handles_none_and_exceptions(monkeypatch):
    async def fake_acompletion(**_kwargs):
        return '{"start_begin": "yes"}'

    monkeypatch.setattr(pi, "llm_acompletion", fake_acompletion)
    assert asyncio.run(pi.check_title_appearance_in_start("A", "A body", logger=Logger())) == "yes"

    monkeypatch.setattr(pi, "llm_acompletion", lambda **_kwargs: asyncio.sleep(0, result="{}"))
    assert asyncio.run(pi.check_title_appearance_in_start("A", "body")) == "no"

    async def fake_start(title, page_text, model=None, logger=None):
        if title == "bad":
            raise RuntimeError("boom")
        return "yes"

    monkeypatch.setattr(pi, "check_title_appearance_in_start", fake_start)
    structure = [
        {"title": "none", "physical_index": None},
        {"title": "ok", "physical_index": 1},
        {"title": "bad", "physical_index": 1},
    ]
    result = asyncio.run(pi.check_title_appearance_in_start_concurrent(structure, [("text", 1)], logger=Logger()))
    assert [item["appear_start"] for item in result] == ["no", "yes", "no"]


def test_extract_toc_content_completion_paths(monkeypatch):
    monkeypatch.setattr(pi, "llm_completion", lambda **kwargs: ("done", "finished"))
    monkeypatch.setattr(pi, "check_if_toc_transformation_is_complete", lambda *_args, **_kwargs: "yes")
    assert pi.extract_toc_content("raw") == "done"

    calls = {"completion": 0, "complete": ["no", "yes"]}

    def fake_completion(**kwargs):
        calls["completion"] += 1
        if kwargs.get("return_finish_reason"):
            return (f"part{calls['completion']}", "finished")
        return "unused"

    monkeypatch.setattr(pi, "llm_completion", fake_completion)
    monkeypatch.setattr(pi, "check_if_toc_transformation_is_complete", lambda *_args, **_kwargs: calls["complete"].pop(0))
    assert pi.extract_toc_content("raw") == "part1part2"

    monkeypatch.setattr(pi, "llm_completion", lambda **kwargs: ("x", "length") if kwargs.get("return_finish_reason") else "x")
    monkeypatch.setattr(pi, "check_if_toc_transformation_is_complete", lambda *_args, **_kwargs: "no")
    with pytest.raises(Exception, match="maximum retries"):
        pi.extract_toc_content("raw")


def test_toc_extractor_transforms_dots(monkeypatch):
    monkeypatch.setattr(pi, "detect_page_index", lambda content, model=None: "yes")

    result = pi.toc_extractor([("A ..... 1\n", 1), ("B . . . . . . 2", 1)], [0, 1], "m")

    assert "A :  1" in result["toc_content"]
    assert result["page_index_given_in_toc"] == "yes"


def test_toc_transformer_success_and_retry(monkeypatch):
    monkeypatch.setattr(pi, "check_if_toc_transformation_is_complete", lambda *_: "yes")
    monkeypatch.setattr(pi, "llm_completion", lambda **kwargs: ('{"table_of_contents":[{"page":"4"}]}', "finished"))
    assert pi.toc_transformer("toc") == [{"page": 4}]

    responses = iter(
        [
            ('{"table_of_contents":[', "length"),
            ('```json\n{"page":"1"}]}\n```', "finished"),
        ]
    )
    complete = iter(["no", "yes"])
    monkeypatch.setattr(pi, "llm_completion", lambda **kwargs: next(responses))
    monkeypatch.setattr(pi, "check_if_toc_transformation_is_complete", lambda *_: next(complete))
    assert pi.toc_transformer("toc") == [{"page": 1}]

    monkeypatch.setattr(pi, "llm_completion", lambda **kwargs: ("{}", "length"))
    monkeypatch.setattr(pi, "check_if_toc_transformation_is_complete", lambda *_: "no")
    with pytest.raises(Exception, match="maximum retries"):
        pi.toc_transformer("toc")


def test_find_toc_pages_and_check_toc(monkeypatch):
    opt = SimpleNamespace(toc_check_page_num=3, model="m")
    original_find_toc_pages = pi.find_toc_pages
    detections = iter(["no", "yes", "yes", "no"])
    monkeypatch.setattr(pi, "toc_detector_single_page", lambda *_args, **_kwargs: next(detections))
    assert pi.find_toc_pages(0, [("p0", 1), ("p1", 1), ("p2", 1), ("p3", 1)], opt, Logger()) == [1, 2]

    monkeypatch.setattr(pi, "find_toc_pages", lambda **kwargs: [])
    assert pi.check_toc([("p", 1)], opt)["toc_content"] is None

    monkeypatch.setattr(pi, "find_toc_pages", lambda **kwargs: [0])
    monkeypatch.setattr(pi, "toc_extractor", lambda pages, toc_pages, model: {"toc_content": "toc", "page_index_given_in_toc": "yes"})
    assert pi.check_toc([("p", 1)], opt)["page_index_given_in_toc"] == "yes"

    calls = {"count": 0}

    def fake_find(**kwargs):
        calls["count"] += 1
        return [kwargs["start_page_index"]]

    def fake_extract(_pages, toc_pages, _model):
        return {"toc_content": f"toc{toc_pages[0]}", "page_index_given_in_toc": "yes" if toc_pages[0] else "no"}

    monkeypatch.setattr(pi, "find_toc_pages", fake_find)
    monkeypatch.setattr(pi, "toc_extractor", fake_extract)
    assert pi.check_toc([("p0", 1), ("p1", 1)], opt)["toc_content"] == "toc1"

    monkeypatch.setattr(pi, "toc_extractor", lambda *_args: {"toc_content": "toc", "page_index_given_in_toc": "no"})
    assert pi.check_toc([("p0", 1), ("p1", 1)], opt)["page_index_given_in_toc"] == "no"

    finds = iter([[0], []])
    monkeypatch.setattr(pi, "find_toc_pages", lambda **kwargs: next(finds))
    assert pi.check_toc([("p0", 1), ("p1", 1)], opt)["page_index_given_in_toc"] == "no"

    detections = iter(["no", "no"])
    monkeypatch.setattr(pi, "toc_detector_single_page", lambda *_args, **_kwargs: next(detections))
    assert original_find_toc_pages(0, [("p0", 1), ("p1", 1)], SimpleNamespace(toc_check_page_num=1, model="m"), Logger()) == []
    assert original_find_toc_pages(2, [("p0", 1)], SimpleNamespace(toc_check_page_num=1, model="m")) == []

    detections = iter(["yes", "no"])
    monkeypatch.setattr(pi, "toc_detector_single_page", lambda *_args, **_kwargs: next(detections))
    assert original_find_toc_pages(0, [("p0", 1), ("p1", 1)], SimpleNamespace(toc_check_page_num=3, model="m")) == [0]


def test_page_number_and_group_helpers(monkeypatch):
    data = [{"page_number": 1, "nodes": [{"page_number": 2}]}]
    assert pi.remove_page_number(data) == [{"nodes": [{}]}]
    assert pi.remove_page_number({"page_number": 1, "nodes": [{"page_number": 2}]}) == {"nodes": [{}]}

    pairs = pi.extract_matching_page_pairs(
        [{"title": "A", "page": 2}, {"title": "B", "page": 5}],
        [{"title": "A", "physical_index": 4}, {"title": "B", "physical_index": 8}, {"title": "C", "physical_index": None}],
        3,
    )
    assert pairs == [{"title": "A", "page": 2, "physical_index": 4}, {"title": "B", "page": 5, "physical_index": 8}]
    assert pi.extract_matching_page_pairs([{"title": "Z"}], [{"title": "A", "physical_index": 1}], 1) == []
    assert pi.calculate_page_offset(pairs) == 2
    assert pi.calculate_page_offset([{"physical_index": None}]) is None
    assert pi.add_page_offset_to_toc_json([{"page": 2}, {"page": "x"}], 3) == [
        {"physical_index": 5},
        {"page": "x"},
    ]

    assert pi.page_list_to_group_text(["a", "b"], [1, 1], max_tokens=5) == ["ab"]
    assert pi.page_list_to_group_text(["a", "b", "c"], [10, 10, 10], max_tokens=10, overlap_page=1)

    assert pi.remove_first_physical_index_section("<physical_index_1>a<physical_index_1>b") == "b"
    assert pi.remove_first_physical_index_section("none") == "none"


def test_generate_toc_and_add_page_number(monkeypatch):
    monkeypatch.setattr(pi, "llm_completion", lambda **kwargs: ('[{"structure":"1","title":"A","physical_index":"<physical_index_2>"}]', "finished"))
    assert pi.generate_toc_init("part")[0]["title"] == "A"
    assert pi.generate_toc_continue([], "part")[0]["title"] == "A"

    monkeypatch.setattr(pi, "llm_completion", lambda **kwargs: ("[]", "length"))
    with pytest.raises(Exception, match="finish reason"):
        pi.generate_toc_init("part")
    with pytest.raises(Exception, match="finish reason"):
        pi.generate_toc_continue([], "part")

    monkeypatch.setattr(pi, "llm_completion", lambda **kwargs: '[{"title":"A","start":"yes","physical_index":"<physical_index_1>"}]')
    assert pi.add_page_number_to_toc("part", [{"title": "A"}]) == [{"title": "A", "physical_index": "<physical_index_1>"}]
    monkeypatch.setattr(pi, "llm_completion", lambda **kwargs: '[{"title":"A","physical_index":"<physical_index_1>"}]')
    assert pi.add_page_number_to_toc("part", [{"title": "A"}]) == [{"title": "A", "physical_index": "<physical_index_1>"}]


def test_process_no_toc_and_toc_no_page_numbers(monkeypatch):
    logger = Logger()
    monkeypatch.setattr(pi, "count_tokens", lambda text, model=None: 1)
    monkeypatch.setattr(pi, "generate_toc_init", lambda part, model=None: [{"physical_index": "<physical_index_1>", "title": "A"}])
    monkeypatch.setattr(pi, "generate_toc_continue", lambda toc, part, model=None: [{"physical_index": "<physical_index_2>", "title": "B"}])
    monkeypatch.setattr(pi, "page_list_to_group_text", lambda contents, tokens: ["g1", "g2"])

    assert pi.process_no_toc([("a", 1), ("b", 1)], logger=logger) == [
        {"physical_index": 1, "title": "A"},
        {"physical_index": 2, "title": "B"},
    ]

    monkeypatch.setattr(pi, "toc_transformer", lambda toc, model=None: [{"title": "A", "page": 1}])
    monkeypatch.setattr(pi, "add_page_number_to_toc", lambda group, toc, model=None: [{"physical_index": "<physical_index_3>", "title": "A"}])
    assert pi.process_toc_no_page_numbers("toc", [], [("a", 1)], logger=logger) == [{"physical_index": 3, "title": "A"}]


def test_process_toc_with_page_numbers_and_none_pages(monkeypatch):
    logger = Logger()
    original_process_none = pi.process_none_page_numbers
    monkeypatch.setattr(pi, "toc_transformer", lambda toc, model=None: [{"title": "A", "page": 1}, {"title": "B"}])
    monkeypatch.setattr(pi, "toc_index_extractor", lambda toc, content, model=None: [{"title": "A", "physical_index": "<physical_index_3>"}])
    monkeypatch.setattr(pi, "process_none_page_numbers", lambda toc, pages, model=None: toc)

    result = pi.process_toc_with_page_numbers("toc", [0], [("toc", 1), ("main", 1), ("later", 1)], toc_check_page_num=2, logger=logger)
    assert result[0]["physical_index"] == 3

    monkeypatch.setattr(pi, "add_page_number_to_toc", lambda contents, item, model=None: [{"physical_index": "<physical_index_2>"}])
    toc = [{"title": "A", "physical_index": 1}, {"title": "B", "page": 2}, {"title": "C", "physical_index": 3}]
    assert original_process_none(toc, [("a", 1), ("b", 1), ("c", 1)])[1]["physical_index"] == 2

    toc = [{"title": "A", "physical_index": 1}, {"title": "B", "page": 2}, {"title": "C", "physical_index": 1}]
    monkeypatch.setattr(pi, "add_page_number_to_toc", lambda contents, item, model=None: [{"physical_index": None}])
    assert "physical_index" not in original_process_none(toc, [("a", 1)], start_index=5)[1]

    toc = [{"title": "B", "page": 2}]
    monkeypatch.setattr(pi, "add_page_number_to_toc", lambda contents, item, model=None: [{"physical_index": "bad"}])
    assert "physical_index" not in original_process_none(toc, [("a", 1), ("b", 1)])[0]


def test_verify_toc_and_fixers(monkeypatch):
    page_list = [("A", 1), ("B", 1), ("C", 1), ("D", 1)]
    assert asyncio.run(pi.verify_toc(page_list, [{"physical_index": None}])) == (0, [])

    async def fake_check(item, page_list, start_index=1, model=None):
        return {"list_index": item["list_index"], "answer": "yes" if item["title"] == "A" else "no", "title": item["title"], "page_number": item["physical_index"]}

    monkeypatch.setattr(pi, "check_title_appearance", fake_check)
    accuracy, bad = asyncio.run(pi.verify_toc(page_list, [{"title": "A", "physical_index": 3}, {"title": "B", "physical_index": 4}], N=None))
    assert accuracy == 0.5
    assert bad[0]["title"] == "B"
    monkeypatch.setattr(pi.random, "sample", lambda population, n: [0])
    accuracy, bad = asyncio.run(pi.verify_toc(page_list, [{"title": "A", "physical_index": 4}], N=10))
    assert accuracy == 1.0 and bad == []
    accuracy, bad = asyncio.run(pi.verify_toc(page_list, [{"title": "skip", "physical_index": None}, {"title": "A", "physical_index": 4}], N=None))
    assert accuracy == 1.0

    async def yes_check(item, page_list, start_index=1, model=None):
        return {"answer": "yes"}

    async def fake_fixer(title, content, model=None):
        return 1

    monkeypatch.setattr(pi, "single_toc_item_index_fixer", fake_fixer)
    monkeypatch.setattr(
        pi,
        "check_title_appearance",
        lambda *args, **kwargs: asyncio.sleep(0, result={"answer": "yes"}),
    )
    toc, invalid = asyncio.run(
        pi.fix_incorrect_toc(
            [{"title": "A", "physical_index": 1}, {"title": "B", "physical_index": 2}],
            page_list,
            [{"list_index": 1, "title": "B", "physical_index": 2}, {"list_index": 9, "title": "X"}],
            logger=Logger(),
        )
    )
    assert toc[1]["physical_index"] == 1
    assert invalid[0]["list_index"] == 9

    monkeypatch.setattr(pi, "check_title_appearance", yes_check)
    toc, invalid = asyncio.run(
        pi.fix_incorrect_toc(
            [
                {"title": "A", "physical_index": 1},
                {"title": "B", "physical_index": 2},
                {"title": "C", "physical_index": 3},
            ],
            page_list,
            [{"list_index": 1, "title": "B", "physical_index": 2}],
            logger=Logger(),
        )
    )
    assert invalid == []

    async def raising_fixer(title, content, model=None):
        raise RuntimeError("bad")

    monkeypatch.setattr(pi, "single_toc_item_index_fixer", raising_fixer)
    toc, invalid = asyncio.run(
        pi.fix_incorrect_toc(
            [{"title": "A", "physical_index": 1}],
            page_list,
            [{"list_index": 0, "title": "A", "physical_index": 1}],
            logger=Logger(),
        )
    )
    assert invalid == []

    async def fake_fix(current_toc, page_list, current_incorrect, start_index=1, model=None, logger=None):
        return current_toc, []

    monkeypatch.setattr(pi, "fix_incorrect_toc", fake_fix)
    assert asyncio.run(pi.fix_incorrect_toc_with_retries([], page_list, [{"list_index": 0}], logger=Logger())) == ([], [])

    async def still_bad(current_toc, page_list, current_incorrect, start_index=1, model=None, logger=None):
        return current_toc, current_incorrect

    monkeypatch.setattr(pi, "fix_incorrect_toc", still_bad)
    assert asyncio.run(pi.fix_incorrect_toc_with_retries([], page_list, [{"list_index": 0}], max_attempts=1, logger=Logger()))[1]


def test_single_toc_item_index_fixer_and_validate(monkeypatch):
    async def fake_acompletion(**_kwargs):
        return '{"physical_index": "<physical_index_4>"}'

    monkeypatch.setattr(pi, "llm_acompletion", fake_acompletion)
    assert asyncio.run(pi.single_toc_item_index_fixer("A", "content")) == 4

    logger = Logger()
    result = pi.validate_and_truncate_physical_indices(
        [
            {"title": "A", "physical_index": 1},
            {"title": "B", "physical_index": 99},
            {"title": "C", "physical_index": 2},
        ],
        page_list_length=5,
        logger=logger,
    )
    assert result == [
        {"title": "A", "physical_index": 1},
        {"title": "B", "physical_index": None},
        {"title": "C", "physical_index": 2},
    ]
    assert pi.validate_and_truncate_physical_indices([], 3) == []
    assert pi.validate_and_truncate_physical_indices([{"title": "A", "physical_index": 3}], 2, start_index=2)[0]["physical_index"] == 3
    assert pi.validate_and_truncate_physical_indices([{"title": "A"}], 2)[0] == {"title": "A"}


def test_meta_processor_branches(monkeypatch):
    logger = Logger()
    opt = SimpleNamespace(toc_check_page_num=2, model="m")
    page_list = [("A", 1), ("B", 1)]
    monkeypatch.setattr(pi, "validate_and_truncate_physical_indices", lambda toc, *_args, **_kwargs: toc)
    monkeypatch.setattr(pi, "process_toc_with_page_numbers", lambda *_args, **_kwargs: [{"title": "A", "physical_index": 1}])
    monkeypatch.setattr(pi, "verify_toc", lambda *_args, **_kwargs: asyncio.sleep(0, result=(1.0, [])))

    result = asyncio.run(pi.meta_processor(page_list, mode="process_toc_with_page_numbers", toc_content="toc", toc_page_list=[0], opt=opt, logger=logger))
    assert result == [{"title": "A", "physical_index": 1}]

    monkeypatch.setattr(pi, "process_toc_no_page_numbers", lambda *_args, **_kwargs: [{"title": "B", "physical_index": 1}])
    result = asyncio.run(pi.meta_processor(page_list, mode="process_toc_no_page_numbers", toc_content="toc", toc_page_list=[0], opt=opt, logger=logger))
    assert result[0]["title"] == "B"

    monkeypatch.setattr(pi, "process_no_toc", lambda *_args, **_kwargs: [{"title": "C", "physical_index": 1}])
    result = asyncio.run(pi.meta_processor(page_list, mode="process_no_toc", opt=opt, logger=logger))
    assert result[0]["title"] == "C"

    monkeypatch.setattr(pi, "verify_toc", lambda *_args, **_kwargs: asyncio.sleep(0, result=(0.7, [{"list_index": 0, "title": "C"}])))
    monkeypatch.setattr(pi, "fix_incorrect_toc_with_retries", lambda toc, *_args, **_kwargs: asyncio.sleep(0, result=(toc, [])))
    assert asyncio.run(pi.meta_processor(page_list, mode="process_no_toc", opt=opt, logger=logger))[0]["title"] == "C"

    outcomes = iter([(0.0, []), (1.0, [])])
    monkeypatch.setattr(pi, "verify_toc", lambda *_args, **_kwargs: asyncio.sleep(0, result=next(outcomes)))
    result = asyncio.run(pi.meta_processor(page_list, mode="process_toc_with_page_numbers", toc_content="toc", toc_page_list=[0], opt=opt, logger=logger))
    assert result[0]["title"] == "B"
    outcomes = iter([(0.0, []), (1.0, [])])
    result = asyncio.run(pi.meta_processor(page_list, mode="process_toc_no_page_numbers", toc_content="toc", toc_page_list=[0], opt=opt, logger=logger))
    assert result[0]["title"] == "C"
    monkeypatch.setattr(pi, "verify_toc", lambda *_args, **_kwargs: asyncio.sleep(0, result=(0.0, [])))
    with pytest.raises(Exception, match="Processing failed"):
        asyncio.run(pi.meta_processor(page_list, mode="process_no_toc", opt=opt, logger=logger))


def test_process_large_node_tree_parser_and_page_index(monkeypatch):
    logger = Logger()
    opt = SimpleNamespace(max_page_num_each_node=1, max_token_num_each_node=2, model="m")
    page_list = [("A", 2), ("B", 2), ("C", 2)]

    async def fake_meta(node_page_list, mode=None, start_index=1, opt=None, logger=None, **_kwargs):
        return [{"title": "Sub", "physical_index": start_index}, {"title": "Next", "physical_index": start_index + 1}]

    async def fake_start(structure, page_list, model=None, logger=None):
        for item in structure:
            item["appear_start"] = "yes"
        return structure

    monkeypatch.setattr(pi, "meta_processor", fake_meta)
    monkeypatch.setattr(pi, "check_title_appearance_in_start_concurrent", fake_start)
    node = {"title": "Root", "start_index": 1, "end_index": 3}
    result = asyncio.run(pi.process_large_node_recursively(node, page_list, opt, logger))
    assert result["nodes"]

    async def matching_meta(node_page_list, mode=None, start_index=1, opt=None, logger=None, **_kwargs):
        return [{"title": "Root", "physical_index": start_index}, {"title": "Child", "physical_index": start_index + 1}]

    monkeypatch.setattr(pi, "meta_processor", matching_meta)
    node = {"title": "Root", "start_index": 1, "end_index": 3}
    result = asyncio.run(pi.process_large_node_recursively(node, page_list, opt, logger))
    assert result["end_index"] == 2

    monkeypatch.setattr(pi, "check_toc", lambda page_list, opt: {"toc_content": "toc", "toc_page_list": [0], "page_index_given_in_toc": "yes"})
    monkeypatch.setattr(pi, "meta_processor", lambda *args, **kwargs: asyncio.sleep(0, result=[{"title": "A", "physical_index": 1}]))
    monkeypatch.setattr(pi, "check_title_appearance_in_start_concurrent", lambda toc, *_args, **_kwargs: asyncio.sleep(0, result=toc))
    monkeypatch.setattr(pi, "post_processing", lambda toc, end: [{"title": "tree", "start_index": 1, "end_index": end}])
    monkeypatch.setattr(pi, "add_node_text", lambda tree, pages: tree[0].update({"text": "body"}))
    monkeypatch.setattr(pi, "add_node_text_with_labels", lambda tree, pages: tree[0].update({"text": "labeled"}))
    monkeypatch.setattr(pi, "generate_summaries_for_structure", lambda tree, model=None: asyncio.sleep(0, result=tree))
    monkeypatch.setattr(pi, "generate_doc_description", lambda tree, model=None: "desc")
    monkeypatch.setattr(pi, "clean_structure_post", lambda tree: tree)
    monkeypatch.setattr(pi, "format_structure", lambda tree, order=None: tree)
    monkeypatch.setattr(pi, "write_node_id", lambda tree: None)
    monkeypatch.setattr(pi, "get_pdf_name", lambda doc: "doc.pdf")
    monkeypatch.setattr(pi, "get_page_tokens", lambda doc, model=None: page_list)

    opt2 = SimpleNamespace(
        toc_check_page_num=2,
        model="m",
        max_page_num_each_node=10,
        max_token_num_each_node=99,
        if_add_node_text="yes",
        if_add_node_id="yes",
        if_add_node_summary="yes",
        if_add_doc_description="yes",
    )
    tree = asyncio.run(pi.tree_parser(page_list, opt2, doc="doc.pdf", logger=logger))
    assert tree[0]["title"] == "tree"
    monkeypatch.setattr(pi, "check_toc", lambda page_list, opt: {"toc_content": None, "toc_page_list": [], "page_index_given_in_toc": "no"})
    assert asyncio.run(pi.tree_parser(page_list, opt2, doc="doc.pdf", logger=logger))[0]["title"] == "tree"
    monkeypatch.setattr(pi, "JsonLogger", lambda doc: logger)
    indexed = pi.page_index_main(pi.BytesIO(b"pdf"), opt2)
    assert indexed["doc_description"] == "desc"
    assert indexed["structure"][0]["text"] == "body"
    monkeypatch.setattr(pi, "ConfigLoader", lambda: SimpleNamespace(load=lambda user_opt: opt2))
    assert pi.page_index(pi.BytesIO(b"pdf"), model="m")["doc_name"] == "doc.pdf"

    with pytest.raises(ValueError, match="Unsupported input type"):
        pi.page_index_main("not.pdf", opt2)

    opt3 = SimpleNamespace(
        toc_check_page_num=2,
        model="m",
        max_page_num_each_node=10,
        max_token_num_each_node=99,
        if_add_node_text="no",
        if_add_node_id="no",
        if_add_node_summary="yes",
        if_add_doc_description="no",
    )
    indexed = pi.page_index_main(pi.BytesIO(b"pdf"), opt3)
    assert "doc_description" not in indexed

    opt4 = SimpleNamespace(
        toc_check_page_num=2,
        model="m",
        max_page_num_each_node=10,
        max_token_num_each_node=99,
        if_add_node_text="no",
        if_add_node_id="no",
        if_add_node_summary="no",
        if_add_doc_description="no",
    )
    indexed = pi.page_index_main(pi.BytesIO(b"pdf"), opt4)
    assert indexed["structure"][0]["title"] == "tree"
