import pytest
from types import SimpleNamespace
from pageindex.config import ConfigLoader, PageIndexConfig

def test_config_loader_default(tmp_path):
    # Mock config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text('model: "gpt-4-test"\nmax_page_num_each_node: 10', encoding="utf-8")
    
    loader = ConfigLoader(default_path=config_file)
    cfg = loader.load()
    
    assert isinstance(cfg, PageIndexConfig)
    assert cfg.model == "gpt-4-test"
    assert cfg.max_page_num_each_node == 10
    # Check default logic
    assert cfg.toc_check_page_num == 3

def test_config_loader_override():
    loader = ConfigLoader(default_path=None)
    override = {"model": "gpt-override", "if_add_node_id": False}
    
    cfg = loader.load(user_opt=override)
    assert cfg.model == "gpt-override"
    assert cfg.if_add_node_id is False

def test_config_validation_error():
    loader = ConfigLoader(default_path=None)
    # Pass invalid type for integer field
    override = {"max_page_num_each_node": "not-an-int"}
    
    with pytest.raises(ValueError, match="Configuration validation failed"):
        loader.load(user_opt=override)

def test_partial_override_object():
    args = SimpleNamespace(model="cmd-model", other_arg=None)
    loader = ConfigLoader(default_path=None)
    cfg = loader.load(user_opt=args)
    assert cfg.model == "cmd-model"
