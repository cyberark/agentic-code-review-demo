from typing import Dict, List, Any, TypedDict, Literal, Optional

# Types for our state
class CompanyPolicy(TypedDict):
    style_guidelines: str
    security_requirements: str
    coding_standards: str

class RepoInfo(TypedDict):
    name: str
    description: str
    url: str
    last_checked: Optional[str]
    issues: List[Dict[str, Any]]

class CommitInfo(TypedDict):
    org_name: str
    repo_name: str
    commit_id: str
    files_changed: List[Dict[str, str]]  # List of {file_path: content}
    commit_message: str
    commit_url: str
    author: str
    timestamp: str

class Issue(TypedDict):
    file_path: str
    description: str
    severity: Literal["high", "medium", "low"]

class AgentState(TypedDict):
    company_policy: CompanyPolicy
    repositories: List[RepoInfo]
    selected_repo: Optional[str]
    selected_issues: List[Issue]  # The specific issues selected for fixing
    commit_info: Optional[CommitInfo]
    fixed_files: List[Dict[str, str]]
    messages: List[Any]
    final_commit_message: str
    continue_analysis: Optional[Dict[str, bool]]  # Track which repos need further analysis
    analysis_iterations: Optional[Dict[str, int]]  # Track how many times each repo has been analyzed