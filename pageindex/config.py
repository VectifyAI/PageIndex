import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, ValidationError

class PageIndexConfig(BaseModel):
    """
    Configuration schema for PageIndex.
    """
    model: str = Field(default="gpt-4o", description="LLM model to use")
    
    # PDF Processing
    toc_check_page_num: int = Field(default=3, description="Number of pages to check for TOC")
    max_page_num_each_node: int = Field(default=5, description="Maximum pages per leaf node")
    max_token_num_each_node: int = Field(default=4000, description="Max tokens per node") # Approx
    
    # Enrichment
    if_add_node_id: bool = Field(default=True, description="Add unique ID to nodes")
    if_add_node_summary: bool = Field(default=True, description="Generate summary for nodes")
    if_add_doc_description: bool = Field(default=True, description="Generate doc-level description")
    if_add_node_text: bool = Field(default=True, description="Keep raw text in nodes")
    
    # Tree Optimization
    if_thinning: bool = Field(default=True, description="Merge small adjacent nodes")
    thinning_threshold: int = Field(default=500, description="Token threshold for merging")
    summary_token_threshold: int = Field(default=200, description="Min tokens required to trigger summary generation")
    
    # Additional
    api_key: Optional[str] = Field(default=None, description="OpenAI API Key (optional, prefers env var)")

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


class ConfigLoader:
    def __init__(self, default_path: Optional[Union[str, Path]] = None):
        if default_path is None:
            env_path = os.getenv("PAGEINDEX_CONFIG")
            if env_path:
                default_path = Path(env_path)
            else:
                cwd_path = Path.cwd() / "config.yaml"
                repo_path = Path(__file__).resolve().parents[1] / "config.yaml"
                default_path = cwd_path if cwd_path.exists() else repo_path
                
        self.default_path = default_path
        self._default_dict = self._load_yaml(default_path) if default_path else {}

    @staticmethod
    def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
        if not path or not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            return {}

    def load(self, user_opt: Optional[Union[Dict[str, Any], Any]] = None) -> PageIndexConfig:
        """
        Load configuration, merging defaults with user overrides and validating via Pydantic.
        
        Args:
            user_opt: Dictionary or object with overrides.
            
        Returns:
            PageIndexConfig: Validated configuration object.
        """
        user_dict: Dict[str, Any] = {}
        if user_opt is None:
            pass
        elif hasattr(user_opt, '__dict__'):
            # Handle SimpleNamespace or other objects
            user_dict = {k: v for k, v in vars(user_opt).items() if v is not None}
        elif isinstance(user_opt, dict):
            user_dict = {k: v for k, v in user_opt.items() if v is not None}
        else:
             raise TypeError(f"user_opt must be dict or object, got {type(user_opt)}")

        # Merge defaults and user overrides
        # Pydantic accepts kwargs, efficiently merging
        merged_data = {**self._default_dict, **user_dict}
        
        try:
            return PageIndexConfig(**merged_data)
        except ValidationError as e:
            # Re-raise nicely or log
            raise ValueError(f"Configuration validation failed: {e}")
