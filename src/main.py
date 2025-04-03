import os
import json
import yaml
import time
from typing import Dict, List, Any, Union, Optional
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
import requests
from datetime import datetime, timedelta

# Import types from structures.py
from structures import CompanyPolicy, RepoInfo, CommitInfo, Issue, AgentState

# Load configuration from YAML file
def load_config():
    config_path = Path(__file__).parents[1] / "config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
            # Ensure required sections exist
            if "git" not in config:
                raise ValueError("Missing 'git' section in config file")
                
            # Verify required git configuration values
            if "org" not in config["git"]:
                raise ValueError("Missing organization name in git config")
            
            # Check for required git API token (unless in local mode)
            if "general" in config and not config["general"].get("local_mode", False):
                if "api_token" not in config["git"] or not config["git"]["api_token"]:
                    raise ValueError("Missing or empty Git API token in config. Required when not in local mode.")
            
            # Ensure api_base_url has a default
            if "api_base_url" not in config["git"]:
                config["git"]["api_base_url"] = ""
                
            # Check for required OpenAI API token
            if "model" not in config:
                raise ValueError("Missing 'model' section in config file")
                
            if "api_token" not in config["model"] or not config["model"]["api_token"]:
                raise ValueError("Missing or empty OpenAI API token in config.yaml")
                    
            return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        # Don't recover from config errors, exit instead
        import sys
        sys.exit(1)

# Load configuration
config = load_config()

# Configuration
GIT_API_TOKEN = config["git"].get("api_token", "")

# Git server configuration
GIT_ORG = config["git"]["org"] 
GIT_API_BASE_URL = config["git"]["api_base_url"]

# Other configuration
POLLING_INTERVAL = config["general"]["polling_interval"]
DEBUG_MODE = config["general"]["debug_mode"]
LOCAL_MODE = config["general"]["local_mode"]
LOCAL_REPOS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["local"]["repos_path"])

# No backward compatibility needed

# Debug print function
def debug_print(message, data=None, important=False, always_print=False):
    """Print debug information when DEBUG_MODE is enabled or always_print is True"""
    if not DEBUG_MODE and not important and not always_print:
        return
        
    # Print with timestamp
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # Format based on importance
    if important:
        print(f"\n{'='*80}\n[{timestamp}] {message}\n{'='*80}")
    else:
        print(f"\n[{timestamp}] {message}")
        
    # Print optional data with indentation
    if data is not None:
        if isinstance(data, str):
            for line in data.split('\n'):
                print(f"    {line}")
        else:
            try:
                # Try to pretty print
                import json
                formatted = json.dumps(data, indent=4)
                for line in formatted.split('\n'):
                    print(f"    {line}")
            except:
                print(f"    {data}")

# Special function just for LLM logging - always prints regardless of debug mode
def log_llm(agent_name, is_input, content, elapsed_time=None):
    """Log LLM input/output, always printing regardless of debug mode"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    if is_input:
        header = f"🔄 LLM INPUT FROM {agent_name}"
        print(f"\n{'='*80}\n[{timestamp}] {header}\n{'='*80}")
        # Print just the first 1000 chars of the input for readability
        print(content[:1000] + "..." if len(content) > 100000000000 else content)
    else:
        time_info = f" (took {elapsed_time:.2f}s)" if elapsed_time is not None else ""
        header = f"🔄 LLM OUTPUT FROM {agent_name}{time_info}"
        print(f"\n{'='*80}\n[{timestamp}] {header}\n{'='*80}")
        # Print just the first 1000 chars of the output for readability
        print(content[:1000] + "..." if len(content) > 100000000000 else content)

# Function to log agent state transitions
def log_state_transition(from_agent, to_agent, state_summary=None):
    """Log a transition between agents, with optional state summary"""
    if not DEBUG_MODE:
        return
        
    transition_msg = f"STATE TRANSITION: {from_agent} → {to_agent}"
    debug_print(transition_msg, important=True)
    
    if state_summary:
        debug_print("State summary:", state_summary)

# Initialize our LLM
llm = ChatOpenAI(
    model=config["model"]["name"],
    api_key=config["model"].get("api_token", "")
)

if DEBUG_MODE:
    debug_print("Debug mode enabled - verbose logging active", important=True)

# Debug mode is enabled for basic logging

# Load company policies from local file
def load_company_policy(policy_file_path: str = "company_policy.json") -> CompanyPolicy:
    """Load company policies from a local JSON file"""
    try:
        with open(policy_file_path, 'r') as f:
            policy = json.load(f)
        return policy
    except FileNotFoundError:
        print(f"Policy file {policy_file_path} not found. Using default policies.")
        return {
            "style_guidelines": "Use 4 spaces for indentation. Maximum line length is 100 characters. Follow PEP 8 for Python code.",
            "security_requirements": "No hardcoded credentials. Use environment variables for secrets. Validate all user inputs.",
            "coding_standards": "Write unit tests for all functions. Use type hints in Python. Document public APIs."
        }

# Git API functions
def get_git_headers():
    """Get headers for Git API requests"""
    return {
        "Authorization": f"token {GIT_API_TOKEN}",
        "Accept": "application/json"
    }
        
# Helper function to create web URLs for repositories
def get_repo_web_url(org_name: str, repo_name: str) -> str:
    """Get the web URL for a repository"""
    if GIT_API_BASE_URL:
        # Extract domain from API base URL
        domain = GIT_API_BASE_URL.replace("http://", "").replace("https://", "").split("/")[0]
        return f"https://{domain}/{org_name}/{repo_name}"
    else:
        # Default to GitHub
        return f"https://github.com/{org_name}/{repo_name}"

def get_organization_repos(org_name: str) -> List[Dict[str, Any]]:
    """Get all repositories for an organization"""
    repos_info = []
    page = 1
    
    # Use custom API base URL if provided, otherwise use GitHub API
    api_base = GIT_API_BASE_URL if GIT_API_BASE_URL else "https://api.github.com"
    
    # Use the GitHub-style API endpoint for organization repositories
    url_template = f"{api_base}/orgs/{org_name}/repos?page={{page}}&per_page=100"
            
    # GitHub-style pagination
    while True:
        url = url_template.format(page=page)
        response = requests.get(url, headers=get_git_headers())
        
        if response.status_code != 200:
            print(f"Error fetching repositories: {response.status_code}")
            print(response.text)
            break
        
        page_repos = response.json()
        if not page_repos:
            break
            
        for repo in page_repos:
            repos_info.append({
                "name": repo["name"],
                "description": repo.get("description", ""),
                "url": repo.get("html_url", ""),  # GitHub uses html_url
                "last_checked": None,
                "issues": []
            })
        
        page += 1
    
    return repos_info

def get_repo_content(org_name: str, repo_name: str, path: str = "") -> List[Dict[str, Any]]:
    """Get content of a repository"""
    api_base = GIT_API_BASE_URL if GIT_API_BASE_URL else "https://api.github.com"
    url = f"{api_base}/repos/{org_name}/{repo_name}/contents/{path}"
    response = requests.get(url, headers=get_git_headers())
    
    
    if response.status_code != 200:
        print(f"Error fetching repo content: {response.status_code}")
        print(response.text)
        return []
    
    return response.json()

def get_file_content(file_url: str) -> Optional[str]:
    """Get content of a file from its URL"""
    try:
        response = requests.get(file_url, headers=get_git_headers())
        
        if response.status_code != 200:
            print(f"Error fetching file content: {response.status_code}")
            # Return empty string for 404 (file not found/empty) to avoid None errors
            if response.status_code == 404:
                print(f"File not found or empty (404). Returning empty string.")
                return ""
            return None
        
        return response.text
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error while fetching file content: {e}")
        print(f"Site might be unreachable. Returning empty string.")
        return ""
    except Exception as e:
        print(f"Unexpected error while fetching file content: {e}")
        return None

def get_recent_commits(org_name: str, repo_name: str, since_time: datetime) -> List[Dict]:
    """Get commits for a repository since a specific time"""
    since_iso = since_time.isoformat()
    api_base = GIT_API_BASE_URL if GIT_API_BASE_URL else "https://api.github.com"
    response = requests.get(
        f"{api_base}/repos/{org_name}/{repo_name}/commits?since={since_iso}",
        headers=get_git_headers()
    )
    
    if response.status_code != 200:
        print(f"Error fetching commits for {repo_name}: {response.status_code}")
        print(response.text)
        return []
    
    return response.json()

def get_commit_details(org_name: str, repo_name: str, commit_sha: str) -> Optional[CommitInfo]:
    """Get details of a specific commit"""
    # Get commit info
    api_base = GIT_API_BASE_URL if GIT_API_BASE_URL else "https://api.github.com"
    commit_response = requests.get(
        f"{api_base}/repos/{org_name}/{repo_name}/commits/{commit_sha}",
        headers=get_git_headers()
    )
    
    if commit_response.status_code != 200:
        print(f"Error fetching commit details: {commit_response.status_code}")
        print(commit_response.text)
        return None
    
    commit_data = commit_response.json()
    
    # Get changed files
    files_changed = []
    for file_change in commit_data["files"]:
        file_path = file_change["filename"]
        
        # Skip binary files, deleted files, or very large files
        if file_change.get("status") == "removed" or not file_change.get("raw_url"):
            continue
        
        # Get file content
        file_response = requests.get(file_change["raw_url"])
        if file_response.status_code == 200:
            files_changed.append({file_path: file_response.text})
    
    return {
        "org_name": org_name,
        "repo_name": repo_name,
        "commit_id": commit_sha,
        "files_changed": files_changed,
        "commit_message": commit_data["commit"]["message"],
        "commit_url": commit_data["html_url"],
        "author": commit_data["commit"]["author"]["name"],
        "timestamp": commit_data["commit"]["author"]["date"]
    }

def create_pull_request(org_name: str, repo_name: str, base_branch: str, head_branch: str, title: str, body: str) -> bool:
    """Create a pull request with the fixes using GitHub CLI"""
    import subprocess
    import tempfile
    
    debug_print(f"Creating pull request using git commands", important=True)
    
    # Check if GitHub CLI is installed
    gh_check = subprocess.run(["which", "gh"], capture_output=True, text=True)
    
    if gh_check.returncode != 0:
        debug_print("GitHub CLI (gh) not found. Falling back to GitHub API.", important=True)
        # Fall back to API method if gh not available
        pr_data = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch
        }
        
        api_base = GIT_API_BASE_URL if GIT_API_BASE_URL else "https://api.github.com"
        response = requests.post(
            f"{api_base}/repos/{org_name}/{repo_name}/pulls",
            headers=get_git_headers(),
            json=pr_data
        )
        
        if response.status_code in (201, 200):
            debug_print(f"Pull request created successfully: {response.json()['html_url']}", important=True)
            return True
        else:
            debug_print(f"Error creating pull request: {response.status_code}")
            debug_print(response.text)
            return False
            
    # GitHub CLI is available, use it
    debug_print(f"Using GitHub CLI to create pull request")
    
    # Write the PR body to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(body)
        body_file_path = temp_file.name
    
    try:
        # Create the pull request using gh CLI
        gh_command = [
            "gh", "pr", "create",
            "--repo", f"{org_name}/{repo_name}",
            "--base", base_branch,
            "--head", head_branch,
            "--title", title,
            "--body-file", body_file_path
        ]
        
        debug_print(f"Running command: {' '.join(gh_command)}")
        
        pr_result = subprocess.run(
            gh_command,
            capture_output=True,
            text=True
        )
        
        if pr_result.returncode == 0:
            debug_print(f"Pull request created successfully: {pr_result.stdout.strip()}", important=True)
            return True
        else:
            debug_print(f"Error creating pull request: {pr_result.stderr}", important=True)
            return False
            
    finally:
        # Clean up the temporary file
        import os
        if os.path.exists(body_file_path):
            os.unlink(body_file_path)

def create_branch_and_commit(org_name: str, repo_name: str, base_ref: str, branch_name: str, 
                            files_to_update: List[Dict[str, str]], commit_message: str) -> bool:
    """Create a new branch and commit the fixed files using git commands"""
    import subprocess
    import os
    import shutil
    
    debug_print(f"Creating branch and committing changes using git commands", important=True)
    
    # For local mode, find the repository in state repositories
    repo_path = None
    work_dir = os.path.join(os.getcwd(), "git_work_dir")
    
    # Create a directory for our work
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        
    repo_dir = os.path.join(work_dir, repo_name)
    
    # Remove existing repo directory if it exists
    if os.path.exists(repo_dir):
        debug_print(f"Removing existing repo directory: {repo_dir}")
        shutil.rmtree(repo_dir)
    
    # Look up the repository path in the global LOCAL_REPOS_PATH
    if LOCAL_MODE:
        potential_repo_path = os.path.join(LOCAL_REPOS_PATH, repo_name)
        if os.path.exists(potential_repo_path) and os.path.isdir(potential_repo_path):
            repo_path = potential_repo_path
            debug_print(f"Using local repository path: {repo_path}")
            
            # Create a copy of the repository in our work directory
            shutil.copytree(repo_path, repo_dir)
    else:
        # Get the repository URL with authentication
        if GIT_API_TOKEN:
            # Extract domain and protocol from API base URL if available, or use github.com
            domain = "github.com"
            protocol = "https"
            if GIT_API_BASE_URL:
                if GIT_API_BASE_URL.startswith("http://"):
                    protocol = "http"
                domain = GIT_API_BASE_URL.replace("http://", "").replace("https://", "").split("/")[0]
                
            # Use token for authentication
            repo_url = f"{protocol}://{GIT_API_TOKEN}@{domain}/{org_name}/{repo_name}.git"
        else:
            # Without token (will only work for public repositories)
            domain = "github.com"
            protocol = "https"
            if GIT_API_BASE_URL:
                if GIT_API_BASE_URL.startswith("http://"):
                    protocol = "http"
                domain = GIT_API_BASE_URL.replace("http://", "").replace("https://", "").split("/")[0]
                
            repo_url = f"{protocol}://{domain}/{org_name}/{repo_name}.git"
        
        # Clone the repository
        debug_print(f"Cloning repository: {repo_url}")
        clone_result = subprocess.run(
            ["git", "clone", repo_url, repo_dir],
            capture_output=True,
            text=True
        )
        
        if clone_result.returncode != 0:
            debug_print(f"Error cloning repository: {clone_result.stderr}", important=True)
            return False
    
    # Checkout the base branch using -C to specify the repository path
    debug_print(f"Checking out base branch: {base_ref}")
    checkout_result = subprocess.run(
        ["git", "-C", repo_dir, "checkout", base_ref],
        capture_output=True,
        text=True
    )
    
    if checkout_result.returncode != 0:
        debug_print(f"Error checking out base branch: {checkout_result.stderr}", important=True)
        # Try the default branch if specified branch fails
        default_branch_result = subprocess.run(
            ["git", "-C", repo_dir, "checkout", "main"],
            capture_output=True,
            text=True
        )
        if default_branch_result.returncode != 0:
            debug_print(f"Error checking out default branch: {default_branch_result.stderr}", important=True)
            return False
    
    # Create and checkout a new branch
    debug_print(f"Creating new branch: {branch_name}")
    branch_result = subprocess.run(
        f"git -C {repo_dir} checkout -b {branch_name}",
        capture_output=True,
        text=True,
        shell=True
    )
    
    if branch_result.returncode != 0:
        debug_print(f"Error creating new branch: {branch_result.stderr}", important=True)
        return False
    
    # Write updated files
    for file_dict in files_to_update:
        for file_path, content in file_dict.items():
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.join(repo_dir, file_path)), exist_ok=True)
            
            # Write the file content
            debug_print(f"Writing updated file: {file_path}")
            with open(os.path.join(repo_dir, file_path), 'w') as f:
                f.write(content)
    
    # Add all changed files
    debug_print("Adding files to git index")
    add_result = subprocess.run(
        ["git", "-C", repo_dir, "add", "."],
        capture_output=True,
        text=True
    )
    
    if add_result.returncode != 0:
        debug_print(f"Error adding files: {add_result.stderr}", important=True)
        return False
    
    # Commit the changes
    debug_print(f"Committing changes with message: {commit_message}")
    commit_result = subprocess.run(
        ["git", "-C", repo_dir, "commit", "-m", commit_message],
        capture_output=True,
        text=True
    )
    
    if commit_result.returncode != 0:
        debug_print(f"Error committing changes: {commit_result.stderr}", important=True)
        return False
    
    # Now merge directly to the base branch and push
    if not LOCAL_MODE:
        # Checkout the base branch
        debug_print(f"Checking out base branch {base_ref} to merge changes")
        checkout_base_result = subprocess.run(
            ["git", "-C", repo_dir, "checkout", base_ref],
            capture_output=True,
            text=True
        )
        
        if checkout_base_result.returncode != 0:
            debug_print(f"Error checking out base branch: {checkout_base_result.stderr}", important=True)
            
            # Try main if specified branch fails
            main_branch_result = subprocess.run(
                ["git", "-C", repo_dir, "checkout", "main"],
                capture_output=True,
                text=True
            )
            
            if main_branch_result.returncode != 0:
                debug_print(f"Error checking out main branch: {main_branch_result.stderr}", important=True)
                return False
            else:
                base_ref = "main"
        
        # Merge the changes from the feature branch
        debug_print(f"Merging changes from {branch_name} to {base_ref}")
        merge_result = subprocess.run(
            ["git", "-C", repo_dir, "merge", "--ff-only", branch_name],
            capture_output=True,
            text=True
        )
        
        if merge_result.returncode != 0:
            debug_print(f"Error merging changes: {merge_result.stderr}", important=True)
            # Try with --no-ff if fast-forward fails
            merge_no_ff_result = subprocess.run(
                ["git", "-C", repo_dir, "merge", "--no-ff", "-m", f"Merge {branch_name} into {base_ref}: {commit_message}", branch_name],
                capture_output=True,
                text=True
            )
            
            if merge_no_ff_result.returncode != 0:
                debug_print(f"Error merging with --no-ff: {merge_no_ff_result.stderr}", important=True)
                return False
        
        # Push the merged changes to the main branch
        debug_print(f"Pushing merged changes to {base_ref}")
        push_result = subprocess.run(
            ["git", "-C", repo_dir, "push", "origin", base_ref],
            capture_output=True,
            text=True
        )
        
        if push_result.returncode != 0:
            debug_print(f"Error pushing merged changes: {push_result.stderr}", important=True)
            return False
    else:
        debug_print("Skipping merge and push in local mode")
    
    debug_print("Successfully created branch, committed changes, and merged to main branch", important=True)
    return True

# Get and analyze a sample of files from a repository
def sample_repo_files(org_name: str, repo_name: str, max_files: int = 10) -> List[Dict[str, str]]:
    """Get a sample of files from a repository for analysis"""
    files_to_analyze = []
    
    # Get repository contents
    contents = get_repo_content(org_name, repo_name)
    
    # Process files and directories
    for item in contents:
        if len(files_to_analyze) >= max_files:
            break
            
        # Skip directories for now
        if item["type"] == "file":
            # Skip non-code files
            file_ext = os.path.splitext(item["name"])[1].lower()
            if file_ext not in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cs', '.go', '.rb', '.php']:
                continue
                
            # Get file content
            content = get_file_content(item["download_url"])
            if content:
                files_to_analyze.append({item["path"]: content})
    
    return files_to_analyze

# Agent definitions
def analysis_agent(state: AgentState) -> Dict[str, Union[str, List[Any]]]:
    """
    The analysis agent that selects which repository and specific issues to fix
    """
    messages = state["messages"]
    repositories = state["repositories"]
    company_policy = state["company_policy"]
    
    debug_print("======== STARTING ANALYSIS AGENT ========", important=True)
    
    # Log the repository analysis information
    repos_with_issues = []
    for repo in repositories:
        if repo.get("issues"):
            issues = repo.get("issues", [])
            high_issues = sum(1 for i in issues if i["severity"] == "high")
            medium_issues = sum(1 for i in issues if i["severity"] == "medium")
            low_issues = sum(1 for i in issues if i["severity"] == "low")
            
            repos_with_issues.append({
                "name": repo["name"], 
                "issues_count": len(issues),
                "severities": f"high: {high_issues}, medium: {medium_issues}, low: {low_issues}"
            })
    
    debug_print(f"Found {len(repos_with_issues)} repositories with issues:", important=True)
    for repo in repos_with_issues:
        debug_print(f"- {repo['name']}: {repo['issues_count']} issues ({repo['severities']})")
        
    debug_print("Preparing to select repository and issues to fix...", important=True)
    
    # Construct prompt for repository and issue selection
    prompt = f"""
    You are a code quality orchestrator agent. Your task is to select which repository and specific issues should be prioritized for fixes.
    
    Company Policies:
    Style Guidelines: {company_policy['style_guidelines']}
    Security Requirements: {company_policy['security_requirements']}
    Coding Standards: {company_policy['coding_standards']}
    
    The following repositories have been analyzed for policy violations. Please:
    
    1. Select ONE repository that should be prioritized for fixes based on:
       - Number of issues
       - Severity of issues (high, medium, low)
       - Types of issues (security issues should be prioritized over style issues)
    
    2. Select 1-3 specific issues from that repository to fix first, focusing on:
       - High severity issues
       - Security issues over style issues
       - Issues that are likely to have the biggest impact if fixed
    
    Your output should be in this specific format:
    SELECTED_REPO: [repository name]
    REASON: [brief explanation why this repository needs fixes most urgently]
    SELECTED_ISSUES:
    - File: [file path], Issue: [issue description], Severity: [severity]
    - [repeat for each selected issue]
    
    Note: Each issue description includes relevant code snippets highlighted with inline backticks (`code`).
    
    Repositories with issues:
    """
    
    repos_with_issues = []
    
    for repo in repositories:
        repo_name = repo["name"]
        issues = repo.get("issues", [])
        
        if not issues:
            continue
            
        repos_with_issues.append(repo)
        
        # Count issues by severity
        high_severity = sum(1 for issue in issues if issue["severity"] == "high")
        medium_severity = sum(1 for issue in issues if issue["severity"] == "medium")
        low_severity = sum(1 for issue in issues if issue["severity"] == "low")
        
        prompt += f"\n- {repo_name}: {repo['description']} (URL: {repo['url']})"
        prompt += f"\n  Issues: {len(issues)} total ({high_severity} high, {medium_severity} medium, {low_severity} low)"
        
        # List the actual issues
        prompt += "\n  Details:"
        for i, issue in enumerate(issues):
            prompt += f"\n    {i+1}. {issue['file_path']}: {issue['description']} (Severity: {issue['severity']})"
    
    # If no repositories have issues, return END
    if not repos_with_issues:
        print("No repositories have issues to fix. Ending workflow.")
        return {
            "next": END,
            "selected_repo": None,
            "selected_issues": [],
            "messages": messages,
            "repositories": repositories
        }
    
    # Add the prompt as a human message
    messages.append(HumanMessage(content=prompt))
    
    # Log LLM input
    log_llm("ANALYSIS_AGENT", True, messages[-1].content)
    
    # Get the LLM response with timing
    start_time = time.time()
    response = llm.invoke(messages)
    elapsed_time = time.time() - start_time
    
    # Log LLM output
    log_llm("ANALYSIS_AGENT", False, response.content, elapsed_time)
    
    messages.append(response)
    
    # Parse the selected repository and issues from the response
    selected_repo = None
    reason = "No specific reason provided"
    selected_issues = []
    response_text = response.content
    
    # Extract the selected repository
    if "SELECTED_REPO:" in response_text:
        repo_line = response_text.split("SELECTED_REPO:")[1].split("\n")[0].strip()
        selected_repo = repo_line
    
    # Extract the reason
    if "REASON:" in response_text:
        reason_section = response_text.split("REASON:")[1]
        if "SELECTED_ISSUES:" in reason_section:
            reason = reason_section.split("SELECTED_ISSUES:")[0].strip()
        else:
            reason = reason_section.strip()
    
    # Extract selected issues directly from the LLM output
    if "SELECTED_ISSUES:" in response_text:
        issues_section = response_text.split("SELECTED_ISSUES:")[1].strip()
        issue_lines = [line.strip() for line in issues_section.split("\n") if line.strip() and line.strip().startswith("-")]
        
        # Parse each selected issue line and create issue objects
        for issue_line in issue_lines:
            # Get the file path, description, and severity from the line
            if "File:" in issue_line and "Issue:" in issue_line and "Severity:" in issue_line:
                try:
                    file_path = issue_line.split("File:")[1].split(",")[0].strip()
                    issue_desc = issue_line.split("Issue:")[1].split(",")[0].strip()
                    severity_text = issue_line.split("Severity:")[1].strip().lower()
                    
                    # Normalize severity
                    if "high" in severity_text:
                        severity = "high"
                    elif "medium" in severity_text:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    # Create a new issue object directly from the LLM output
                    selected_issues.append({
                        "file_path": file_path,
                        "description": issue_desc,
                        "severity": severity
                    })
                    
                    debug_print(f"Added issue from LLM output: {file_path} ({severity})")
                except Exception as e:
                    debug_print(f"Error parsing issue line: {str(e)}")
                    continue
    
    print(f"LLM selected repository: {selected_repo}")
    print(f"Reason: {reason}")
    print(f"Selected {len(selected_issues)} issues to fix")

    
    # Determine next state and log the transition
    next_state = "developer_agent" if selected_repo and selected_issues else END
    
    if next_state == "developer_agent":
        log_state_transition("ANALYSIS AGENT", "DEVELOPER AGENT", {
            "selected_repo": selected_repo, 
            "issues_count": len(selected_issues)
        })
    else:
        log_state_transition("ANALYSIS AGENT", "END", {
            "reason": "No repository or issues selected for fixing"
        })
    
    return {
        "next": next_state,
        "selected_repo": selected_repo,
        "selected_issues": selected_issues,
        "messages": messages
    }

def code_review_agent(state: AgentState) -> Dict[str, Union[str, List[Any]]]:
    """
    Agent that analyzes all repositories for policy violations
    """
    messages = state["messages"]
    repositories = state["repositories"]
    company_policy = state["company_policy"]
    
    debug_print("======== STARTING CODE REVIEW AGENT ========", important=True)
    debug_print(f"Processing {len(repositories)} repositories for policy compliance")
    debug_print("Loading company policies:", important=True)
    debug_print(f"- Style Guidelines: {company_policy['style_guidelines'][:100]}...")
    debug_print(f"- Security Requirements: {company_policy['security_requirements'][:100]}...")
    debug_print(f"- Coding Standards: {company_policy['coding_standards'][:100]}...")
    debug_print("Beginning repository analysis...", important=True)
    
    # Log workflow entry point
    log_state_transition("WORKFLOW START", "CODE REVIEW AGENT", {
        "repositories_count": len(repositories),
        "policy_loaded": True
    })
    
    # Analyze all repositories
    
    for i, repo in enumerate(repositories):
        repo_name = repo["name"]
        
        # Skip if already analyzed
        # if repo.get("last_checked") and repo.get("issues") is not None:
        #     debug_print(f"Repository {repo_name} already analyzed, skipping...")
        #     continue
            
        debug_print(f"Analyzing repository: {repo_name} ({i+1}/{len(repositories)})")
        
        # Get sample files from the repository
        if LOCAL_MODE:
            if "path" in repo:
                files_to_analyze = get_local_repo_files(repo["path"])
            else:
                debug_print(f"Error: Local repository {repo_name} has no path specified")
                files_to_analyze = []
        else:
            files_to_analyze = sample_repo_files(GIT_ORG, repo_name)
        
        if not files_to_analyze:
            print(f"No suitable files found in repository {repo_name}")
            repositories[i]["issues"] = []
            repositories[i]["last_checked"] = datetime.now().isoformat()
            continue
        
        # Reset messages for each repository to avoid context overload
        repo_messages = []
        
        # Construct prompt for policy check with strict output format
        prompt = f"""
        You are a code review agent. Your task is to review code in a repository and ensure it complies with company policies.
        
        Company Policies:
        Style Guidelines: {company_policy['style_guidelines']}
        Security Requirements: {company_policy['security_requirements']}
        Coding Standards: {company_policy['coding_standards']}
        
        Repository: {repo_name}
        
        Please analyze the following files for compliance issues:
        """
        
        for file_dict in files_to_analyze:
            for file_path, content in file_dict.items():
                # Limit content length for large files
                content_preview = content[:3000] + "..." if len(content) > 3000 else content
                prompt += f"\nFile: {file_path}\n```\n{content_preview}\n```\n"
        
        prompt += """\nIdentify all issues with the code that violate company policies.

IMPORTANT: You MUST use the following structured format for EACH issue found:

ISSUE_START
FILE: [exact file path]
DESCRIPTION: [clear description of the issue, including the relevant code with inline backticks]
SEVERITY: [high/medium/low]
ISSUE_END

For the DESCRIPTION, always include the actual line or snippet of code containing the issue using inline backticks (`code`).
Do NOT use code blocks with triple backticks.

Example description format:
"Function missing error handling when calling `fetch_data(url)`"
"Variable `let user` not properly initialized, creating a potential null reference"

IMPORTANT DECISION POINT: At the end of your response, indicate ONLY if you are CERTAIN there are significant issues in the codebase that you couldn't identify in this pass. Be extremely selective and conservative - only request further analysis if you encountered clear signs of additional serious issues that you couldn't fully capture this time. Use the following format:

CONTINUE_ANALYSIS: [YES/NO] 
REASON: [Specific evidence of additional serious issues that weren't fully captured]

Note that answering YES will trigger another analysis pass, so only do this if absolutely necessary.

If multiple issues are found, list them one after another using the format above.
If no issues are found, respond with "NO_ISSUES_FOUND".
"""
        
        # Add the prompt as a human message
        repo_messages.append(HumanMessage(content=prompt))
        
        # Log LLM input
        log_llm(f"CODE_REVIEW_AGENT_{repo_name}", True, prompt)
        
        # Get the LLM response with timing
        start_time = time.time()
        response = llm.invoke(repo_messages)
        elapsed_time = time.time() - start_time
        
        # Log LLM output
        log_llm(f"CODE_REVIEW_AGENT_{repo_name}", False, response.content, elapsed_time)
        
        repo_messages.append(response)
        
        # Parse issues from the response
        issues_text = response.content
        
        # Parse issue detection with the improved structured format
        issues = []
        continue_analysis = False
        continue_reason = "No specific reason provided"
        
        # Check if any issues were found
        if "NO_ISSUES_FOUND" in issues_text:
            debug_print(f"No issues found in repository {repo_name}")
        elif "ISSUE_START" in issues_text:
            # Split by ISSUE_START to get each issue block
            issue_blocks = issues_text.split("ISSUE_START")
            
            for block in issue_blocks:
                if "ISSUE_END" not in block:
                    continue
                    
                # Extract the content between ISSUE_START and ISSUE_END
                issue_content = block.split("ISSUE_END")[0].strip()
                
                # Parse the issue fields
                try:
                    file_line = next((line for line in issue_content.split('\n') if line.startswith("FILE:")), None)
                    desc_line = next((line for line in issue_content.split('\n') if line.startswith("DESCRIPTION:")), None)
                    sev_line = next((line for line in issue_content.split('\n') if line.startswith("SEVERITY:")), None)
                    
                    if file_line and desc_line and sev_line:
                        file_path = file_line.replace("FILE:", "").strip()
                        description = desc_line.replace("DESCRIPTION:", "").strip()
                        severity_text = sev_line.replace("SEVERITY:", "").strip().lower()
                        
                        # Normalize severity to one of the three allowed values
                        if "high" in severity_text:
                            severity = "high"
                        elif "medium" in severity_text:
                            severity = "medium"
                        else:
                            severity = "low"
                        
                        issues.append({
                            "file_path": file_path,
                            "description": description,
                            "severity": severity
                        })
                        debug_print(f"Found issue in {file_path}: {description} ({severity})")
                except Exception as e:
                    debug_print(f"Error parsing issue block: {str(e)}")
                    continue
        
        # Parse whether to continue analysis
        if "CONTINUE_ANALYSIS:" in issues_text:
            continue_section = issues_text.split("CONTINUE_ANALYSIS:")[1].strip()
            continue_value = continue_section.split("\n")[0].strip() if "\n" in continue_section else continue_section
            
            # Check if we should continue
            if "yes" in continue_value.lower():
                continue_analysis = True
                
                # Try to extract the reason if provided
                if "REASON:" in issues_text:
                    reason_section = issues_text.split("REASON:")[1].strip()
                    continue_reason = reason_section.split("\n")[0].strip() if "\n" in reason_section else reason_section
                
                debug_print(f"LLM suggests continuing analysis of {repo_name}. Reason: {continue_reason}", important=True)
        
        print(f"Found {len(issues)} issues in repository {repo_name}")
        
        # Update the repository's issues in the state
        repositories[i]["issues"] = issues
        repositories[i]["last_checked"] = datetime.now().isoformat()
        
        # Store files for later use
        repositories[i]["files_to_analyze"] = files_to_analyze
        
        # Store the continue_analysis flag for this repository
        if "continue_analysis" not in state:
            state["continue_analysis"] = {}
        state["continue_analysis"][repo_name] = continue_analysis
        
        # Track the number of times this repository has been analyzed
        if "analysis_iterations" not in state:
            state["analysis_iterations"] = {}
        state["analysis_iterations"][repo_name] = state["analysis_iterations"].get(repo_name, 0) + 1
        
        # Log the iteration count
        iteration_count = state["analysis_iterations"].get(repo_name, 1)
        debug_print(f"Repository {repo_name} has been analyzed {iteration_count} times", important=True)
    
    # Check if any repositories need further analysis
    repos_needing_more_analysis = [repo["name"] for repo in repositories 
                                  if state.get("continue_analysis", {}).get(repo["name"], False)]

    if repos_needing_more_analysis:
        debug_print(f"These repositories need further analysis: {', '.join(repos_needing_more_analysis)}", important=True)
        
        # Let the LLM's response determine our next action
        # Simply continue analyzing if any repository needs more analysis
        debug_print("Continuing analysis of repositories based on LLM recommendation", important=True)
        next_state = "code_review_agent"  # Loop back to continue analysis
    else:
        # No repositories need further analysis, proceed to analysis agent
        debug_print("No further analysis needed, proceeding to fix identified issues", important=True)
        next_state = "analysis_agent"

    # Go to the next appropriate agent
    log_state_transition("CODE REVIEW AGENT", next_state.upper(), {
        "repositories_analyzed": len(repositories),
        "repositories_with_issues": sum(1 for r in repositories if r.get("issues")),
        "repositories_needing_more_analysis": len(repos_needing_more_analysis)
    })
    
    return {
        "next": next_state,
        "messages": messages,
        "repositories": repositories,
        "continue_analysis": state.get("continue_analysis", {})  # Pass along the continue_analysis state
    }

def developer_agent(state: AgentState) -> Dict[str, Union[str, List[Any]]]:
    """
    The developer agent that fixes specific issues selected by the main agent
    """
    messages = state["messages"]
    company_policy = state["company_policy"]
    selected_repo = state["selected_repo"]
    selected_issues = state["selected_issues"]
    repositories = state["repositories"]
    
    debug_print("======== STARTING DEVELOPER AGENT ========", important=True)
    debug_print(f"Working on repository: {selected_repo}")
    debug_print(f"Assigned to fix {len(selected_issues)} issues:", important=True)
    
    # Print detailed issue information
    for i, issue in enumerate(selected_issues):
        debug_print(f"Issue {i+1}:", important=True)
        debug_print(f"  File: {issue['file_path']}")
        debug_print(f"  Severity: {issue['severity']}")
        debug_print(f"  Description: {issue['description']}")
    
    debug_print("Preparing to fix code issues...", important=True)
    
    # Find the selected repository with its files
    repo_data = next((repo for repo in repositories if repo["name"] == selected_repo), None)
    
    if not repo_data or not repo_data.get("files_to_analyze"):
        print(f"No files found for repository {selected_repo}")
        return {
            "next": END,
            "messages": messages
        }
    
    if not selected_issues:
        print(f"No issues were selected to fix in repository {selected_repo}")
        return {
            "next": END,
            "messages": messages
        }
    
    files_to_analyze = repo_data["files_to_analyze"]
    
    # For logging
    print(f"Fixing {len(selected_issues)} selected issues in repository {selected_repo}")
    
    # Group issues by file path for more efficient processing
    issues_by_file = {}
    for issue in selected_issues:
        file_path = issue["file_path"]
        if file_path not in issues_by_file:
            issues_by_file[file_path] = []
        issues_by_file[file_path].append(issue)
    
    fixed_files = []
    
    # Create mock commit info for later use
    commit_info = {
        "org_name": GIT_ORG,
        "repo_name": selected_repo,
        "commit_id": "policy-check",  # Not a real commit
        "files_changed": files_to_analyze,
        "commit_message": "Fix selected policy violations",
        "commit_url": get_repo_web_url(GIT_ORG, selected_repo),
        "author": "policy-agent",
        "timestamp": datetime.now().isoformat()
    }
    
    # Process each file that needs changes
    for file_dict in files_to_analyze:
        for file_path, content in file_dict.items():
            # Check if this file has selected issues to fix
            if file_path not in issues_by_file:
                # No selected issues for this file, keep it as is
                fixed_files.append({file_path: content})
                continue
            
            # Get the issues to fix for this file
            file_issues = issues_by_file[file_path]
            
            # For logging
            print(f"Fixing {len(file_issues)} selected issues in file {file_path}")
            
            # Reset messages for each file to avoid context overflow
            file_messages = []
            
            # Construct prompt for fixing issues
            prompt = f"""
            You are a developer agent. Your task is to fix specific issues in code that violate company policies.
            
            Company Policies:
            Style Guidelines: {company_policy['style_guidelines']}
            Security Requirements: {company_policy['security_requirements']}
            Coding Standards: {company_policy['coding_standards']}
            
            Repository: {selected_repo}
            File: {file_path}
            
            Original code:
            ```
            {content}
            ```
            
            Selected issues to fix:
            """
            
            for i, issue in enumerate(file_issues):
                prompt += f"\n{i+1}. {issue['description']} (Severity: {issue['severity']})"
                
            prompt += """

IMPORTANT: Each issue description includes the problematic code highlighted with inline backticks (`code`).
Focus your fixes on these specific code segments identified in the issue descriptions.

Please provide the fixed version of the code that resolves ONLY these specific issues while maintaining the original functionality."""
            
            # Add the prompt as a human message
            file_messages.append(HumanMessage(content=prompt))
            
            # Log LLM input
            log_llm(f"DEVELOPER_AGENT_{file_path}", True, prompt)
            
            # Get the LLM response with timing
            start_time = time.time()
            response = llm.invoke(file_messages)
            elapsed_time = time.time() - start_time
            
            # Log LLM output
            log_llm(f"DEVELOPER_AGENT_{file_path}", False, response.content, elapsed_time)
            
            file_messages.append(response)
            
            # Extract fixed code from response
            fixed_code = response.content
            
            # Extract code block if present
            if "```" in fixed_code:
                code_blocks = fixed_code.split("```")
                for i, block in enumerate(code_blocks):
                    if i % 2 == 1:  # This is inside a code block
                        # Remove language identifier if present
                        if block.strip() and "\n" in block:
                            language_line, code = block.split("\n", 1)
                            fixed_code = code
                        else:
                            fixed_code = block
                        break
            
            fixed_files.append({file_path: fixed_code})
    
    # Log state transition to commit agent
    log_state_transition("DEVELOPER AGENT", "COMMIT AGENT", {
        "files_fixed": len(fixed_files),
        "issues_addressed": len(selected_issues)
    })
    
    return {
        "next": "commit_agent",
        "fixed_files": fixed_files,
        "commit_info": commit_info,
        "selected_issues": selected_issues,  # Pass the selected issues forward
        "messages": messages
    }

def commit_agent(state: AgentState) -> Dict[str, Union[str, List[Any]]]:
    """
    The commit agent that finalizes and commits the fixed code for selected issues
    """
    messages = state["messages"]
    commit_info = state["commit_info"]
    fixed_files = state["fixed_files"]
    selected_issues = state["selected_issues"]
    selected_repo = state["selected_repo"]  # Get the LLM-selected repository name
    
    debug_print("======== STARTING COMMIT AGENT ========", important=True)
    debug_print(f"Finalizing fixes for repository: {selected_repo}")
    
    # Summary of fixed files
    debug_print(f"Successfully fixed {len(fixed_files)} files:", important=True)
    for file_dict in fixed_files:
        for file_path in file_dict.keys():
            debug_print(f"  - {file_path}")
    
    # Summary of addressed issues
    debug_print(f"Addressed {len(selected_issues)} policy issues:", important=True)
    for issue in selected_issues:
        debug_print(f"  - {issue['severity'].upper()} in {issue['file_path']}: {issue['description']}")
    
    debug_print("Preparing commit message and PR...", important=True)
    
    # Construct prompt for generating commit message and branch name with fixed code
    prompt = f"""
    You are a commit agent. Your task is to create a meaningful commit message and descriptive branch name based on the specific issues that were fixed.
    
    Repository: {selected_repo}
    
    Issues fixed:
    """
    
    for i, issue in enumerate(selected_issues):
        prompt += f"\n{i+1}. {issue['description']} in {issue['file_path']} (Severity: {issue['severity']})"
    
    # Add fixed files with before/after content
    prompt += "\n\nFixed files showing changes:"
    
    for file_dict in fixed_files:
        for file_path, fixed_content in file_dict.items():
            # Find issues related to this file
            file_issues = [issue for issue in selected_issues if issue['file_path'] == file_path]
            
            if file_issues:
                # Get original content if available
                original_content = None
                
                # Look up original content from commit_info if available
                if commit_info and commit_info.get("files_changed"):
                    for orig_file_dict in commit_info.get("files_changed", []):
                        if file_path in orig_file_dict:
                            original_content = orig_file_dict[file_path]
                            break
                
                # Get a preview of the fixed code (limited to prevent token overflow)
                fixed_preview = fixed_content[:800] + "..." if len(fixed_content) > 800 else fixed_content
                
                prompt += f"\n\nFile: {file_path}"
                prompt += f"\nIssues fixed in this file: {len(file_issues)}"
                
                # Add the original code if available
                if original_content:
                    orig_preview = original_content[:800] + "..." if len(original_content) > 800 else original_content
                    prompt += f"\n\nOriginal code:\n```\n{orig_preview}\n```"
                
                # Add the fixed code
                prompt += f"\n\nFixed code:\n```\n{fixed_preview}\n```"
                
                # Add a simple diff if both original and fixed content are available
                if original_content:
                    prompt += "\n\nMain changes made:"
                    for issue in file_issues:
                        # Extract a description of what was fixed, focusing on code parts
                        desc = issue['description']
                        prompt += f"\n- {desc}"
    
    prompt += """
    
Please provide:

1. A clear and concise commit message that describes the specific fixes made.

2. A short, descriptive branch name suffix (up to 5 words, kebab-case) that summarizes the main fix or issue category.
   For example: "fix-security-vulnerabilities" or "update-styling-standards"

Your response should use this format:
COMMIT_MESSAGE: [Your commit message here]
BRANCH_SUFFIX: [Your branch name suffix here]
"""
    
    # Add the prompt as a human message
    messages.append(HumanMessage(content=prompt))
    
    # Log LLM input
    log_llm("COMMIT_AGENT", True, prompt)
    
    # Get the LLM response with timing
    start_time = time.time()
    response = llm.invoke(messages)
    elapsed_time = time.time() - start_time
    
    # Log LLM output
    log_llm("COMMIT_AGENT", False, response.content, elapsed_time)
    
    messages.append(response)
    
    # Extract commit message and branch suffix from response
    response_text = response.content
    final_commit_message = response_text
    branch_suffix = "general-policy-fixes"  # Default suffix
    
    # Parse the structured response
    if "COMMIT_MESSAGE:" in response_text:
        commit_section = response_text.split("COMMIT_MESSAGE:")[1]
        if "BRANCH_SUFFIX:" in commit_section:
            final_commit_message = commit_section.split("BRANCH_SUFFIX:")[0].strip()
            branch_suffix_section = response_text.split("BRANCH_SUFFIX:")[1].strip()
            branch_suffix = branch_suffix_section.split("\n")[0].strip() if "\n" in branch_suffix_section else branch_suffix_section.strip()
    
    # Create a descriptive branch name with timestamp and suffix from LLM
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    branch_name = f"policy-fixes-{selected_repo}-{timestamp}-{branch_suffix}"
    
    debug_print(f"Generated branch suffix: {branch_suffix}", important=True)
    
    # For local testing mode, just show what would be committed
    if LOCAL_MODE:
        debug_print("LOCAL MODE: Showing planned commit details instead of creating actual commit", important=True)
        debug_print(f"Repository: {selected_repo}")
        debug_print(f"Branch name: {branch_name}")
        debug_print(f"Commit message: {final_commit_message}")
        debug_print("Files to be committed:")
        
        # Find the repository path to show absolute file paths
        repo_path = next((repo["path"] for repo in state["repositories"] if repo["name"] == selected_repo), None)
        
        for file_dict in fixed_files:
            for rel_path, content in file_dict.items():
                if repo_path:
                    abs_path = os.path.join(repo_path, rel_path)
                    debug_print(f"  - {abs_path}")
                else:
                    debug_print(f"  - {rel_path}")
                
                # Always print the full content of fixed files
                debug_print(f"  FIXED CODE FOR {rel_path}:", important=True)
                debug_print(f"```\n{content}\n```")
                
                # Show diff of changes
                if repo_path and os.path.exists(os.path.join(repo_path, rel_path)):
                    try:
                        with open(os.path.join(repo_path, rel_path), 'r') as f:
                            original_content = f.read()
                            
                        debug_print(f"  CHANGES FOR {rel_path}:", important=True)
                        import difflib
                        diff = difflib.unified_diff(
                            original_content.splitlines(),
                            content.splitlines(),
                            fromfile=f'a/{rel_path}',
                            tofile=f'b/{rel_path}',
                            lineterm=''
                        )
                        for line in diff:
                            debug_print(f"    {line}")
                    except Exception as e:
                        debug_print(f"  Error generating diff: {str(e)}")
        
        # Generate mock PR description
        pr_body = f"""
        # Policy Compliance Fixes: {branch_suffix.replace('-', ' ')}

        This PR fixes specific policy compliance issues selected by the analysis agent.

        ## Issues Fixed:

        {', '.join([f"{issue['description']} ({issue['severity']})" for issue in selected_issues])}
        
        ## Files Changed:
        
        {', '.join([list(file.keys())[0] for file in fixed_files if list(file.keys())[0] in [issue['file_path'] for issue in selected_issues]])}
        
        _This PR was automatically generated by the Code Review Agent system in local testing mode._
        _Branch: {branch_name}_
        """
        
        debug_print("PR DESCRIPTION:", important=True)
        debug_print(pr_body)
                
    # Create a new branch, commit the fixed files, and merge directly if in GitHub mode
    elif GIT_API_TOKEN:
        # Create the changes on a branch and merge directly to main
        commit_success = create_branch_and_commit(
            commit_info['org_name'],
            selected_repo,  # Use the LLM-selected repository name
            "main",  # Assume main is the base branch
            branch_name,
            fixed_files,
            final_commit_message
        )
        
        if commit_success:
            # Log the successful direct merge
            debug_print(f"Changes successfully merged to main branch", important=True)
            debug_print(f"Fix summary: {branch_suffix.replace('-', ' ')}", important=True)
            debug_print(f"Issues fixed:", important=True)
            for issue in selected_issues:
                debug_print(f"- {issue['description']} ({issue['severity']})")
            
            # List the files that were changed
            debug_print(f"Files changed:", important=True)
            for file_dict in fixed_files:
                for file_path in file_dict.keys():
                    if file_path in [issue['file_path'] for issue in selected_issues]:
                        debug_print(f"- {file_path}")
        else:
            debug_print(f"Failed to merge changes to main branch", important=True)
    # We shouldn't reach this point because run_workflow checks for API token
    # But just in case, handle the error
    else:
        debug_print("Error: API token required when not in local mode", important=True)
        debug_print("Set GIT_API_TOKEN environment variable or enable LOCAL_MODE", important=True)
        import sys
        sys.exit(1)
    
    # Log final state transition
    log_state_transition("COMMIT AGENT", "END", {
        "repository": selected_repo,
        "commit_message": final_commit_message,
        "files_committed": len(fixed_files),
        "issues_fixed": len(selected_issues),
        "merged_to_main": "Yes" if GIT_API_TOKEN and not LOCAL_MODE else "No (local mode or no API token)"
    })
    
    return {
        "next": END,
        "final_commit_message": final_commit_message,
        "messages": messages,
        "selected_repo": selected_repo  # Ensure selected repo is passed through the entire workflow
    }

# Define the workflow
def build_workflow():
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analysis_agent", analysis_agent)  # Updated node name to match function name
    workflow.add_node("code_review_agent", code_review_agent)  # Updated node name to match function name
    workflow.add_node("developer_agent", developer_agent)
    workflow.add_node("commit_agent", commit_agent)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "analysis_agent",
        lambda x: x["next"]
    )
    
    workflow.add_conditional_edges(
        "code_review_agent",
        lambda x: x["next"]
    )
    
    # Add regular edges
    workflow.add_edge("developer_agent", "commit_agent")
    workflow.add_edge("commit_agent", END)
    
    # Set the entry point
    workflow.set_entry_point("code_review_agent")  # Start with code review agent
    
    return workflow

# Functions for local repository mode
def get_local_repositories() -> List[Dict[str, Any]]:
    """Get all local test repositories"""
    debug_print(f"Looking for local repositories in: {LOCAL_REPOS_PATH}")
    
    if not os.path.exists(LOCAL_REPOS_PATH):
        debug_print(f"Local repository path doesn't exist: {LOCAL_REPOS_PATH}")
        return []
    
    repos = []
    for repo_name in os.listdir(LOCAL_REPOS_PATH):
        repo_path = os.path.join(LOCAL_REPOS_PATH, repo_name)
        if os.path.isdir(repo_path):
            repos.append({
                "name": repo_name,
                "description": f"Local test repository: {repo_name}",
                "url": f"file://{repo_path}",
                "path": repo_path,
                "last_checked": None,
                "issues": []
            })
    
    debug_print(f"Found {len(repos)} local repositories: {[r['name'] for r in repos]}")
    return repos

def get_local_repo_files(repo_path: str, max_files: int = 10) -> List[Dict[str, str]]:
    """Get all files from a local repository"""
    debug_print(f"Scanning files in local repository: {repo_path}")
    
    files_to_analyze = []
    for root, _, files in os.walk(repo_path):
        for file_name in files:
            if len(files_to_analyze) >= max_files:
                break
                
            # Skip non-code files
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext not in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cs', '.go', '.rb', '.php']:
                continue
                
            file_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(file_path, repo_path)
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                files_to_analyze.append({rel_path: content})
                debug_print(f"Added file for analysis: {rel_path}")
            except Exception as e:
                debug_print(f"Error reading file {file_path}: {str(e)}")
    
    return files_to_analyze

# Main function to run the workflow
def run_workflow():
    # Load company policy
    company_policy = load_company_policy()
    
    # Get repositories based on mode
    if LOCAL_MODE:
        # Use local repositories
        repositories = get_local_repositories()
    elif GIT_API_TOKEN:
        # Use Git repositories from configured server
        repositories = get_organization_repos(GIT_ORG)
    else:
        # If not in local mode and no API token, this is an error
        print("Error: API token required when not in local mode")
        print("Set git.api_token in config.yaml or enable local_mode")
        import sys
        sys.exit(1)
    
    # Initialize the state
    initial_state = {
        "company_policy": company_policy,
        "repositories": repositories,
        "selected_repo": None,
        "selected_issues": [],
        "commit_info": None,
        "fixed_files": [],
        "messages": [],
        "final_commit_message": "",
        "continue_analysis": {},  # Track which repos need further analysis
        "analysis_iterations": {}  # Track how many times each repo has been analyzed
    }
    
    # Build the workflow
    workflow = build_workflow()
    app = workflow.compile()
    
    # Execute the workflow
    return app.invoke(initial_state)

# Continuous monitoring function
def monitor_repositories():
    if LOCAL_MODE:
        debug_print(f"🚀 STARTING LOCAL REPOSITORY ANALYSIS MODE", important=True)
        debug_print(f"Scanning local test repositories in: {LOCAL_REPOS_PATH}", important=True)
        debug_print(f"Debug mode: {'ENABLED' if DEBUG_MODE else 'DISABLED'}", important=True)
    else:
        # Display Git server information
        server_info = "GITHUB"
        if GIT_API_BASE_URL:
            server_info = f"CUSTOM GIT SERVER @ {GIT_API_BASE_URL}"
            
        debug_print(f"🚀 STARTING {server_info} MONITORING FOR ORGANIZATION: {GIT_ORG}", important=True)
        debug_print(f"Debug mode: {'ENABLED' if DEBUG_MODE else 'DISABLED'}", important=True)
        debug_print(f"Polling interval: {POLLING_INTERVAL} seconds", important=True)
    
    # For local testing, run just once
    if LOCAL_MODE:
        try:
            debug_print(f"Running policy compliance check at {datetime.now().isoformat()}", important=True)
            
            # Run the workflow
            result = run_workflow()
            
            # Log the results
            if result.get("selected_repo"):
                debug_print(f"Selected repository: {result['selected_repo']}", important=True)
                
                if result.get("selected_issues"):
                    debug_print(f"Fixed {len(result['selected_issues'])} selected issues", important=True)
                    for i, issue in enumerate(result.get("selected_issues", [])):
                        debug_print(f"Issue {i+1}: {issue['description']} ({issue['severity']})")
                        
                    if result.get("final_commit_message"):
                        debug_print(f"Created PR with message: {result['final_commit_message']}", important=True)
                else:
                    debug_print("No issues selected for fixing", important=True)
            else:
                debug_print("No repositories selected for review", important=True)
                
            debug_print("Local test run completed", important=True)
            
        except Exception as e:
            debug_print(f"Error during workflow execution: {str(e)}", important=True)
            import traceback
            debug_print(traceback.format_exc())
    else:
        # Run in continuous monitoring mode for GitHub
        while True:
            try:
                debug_print(f"Running policy compliance check at {datetime.now().isoformat()}", important=True)
                
                # Run the workflow
                result = run_workflow()
                
                # Log the results
                if result.get("selected_repo"):
                    debug_print(f"Selected repository: {result['selected_repo']}", important=True)
                    
                    if result.get("selected_issues"):
                        debug_print(f"Fixed {len(result['selected_issues'])} selected issues", important=True)
                        for i, issue in enumerate(result.get("selected_issues", [])):
                            debug_print(f"Issue {i+1}: {issue['description']} ({issue['severity']})")
                            
                        if result.get("final_commit_message"):
                            debug_print(f"Created PR with message: {result['final_commit_message']}", important=True)
                    else:
                        debug_print("No issues selected for fixing", important=True)
                else:
                    debug_print("No repositories selected for review", important=True)
                
                # Wait before the next check
                debug_print(f"Waiting {POLLING_INTERVAL} seconds before next check...", important=True)
                time.sleep(POLLING_INTERVAL)
            
            except Exception as e:
                debug_print(f"Error during workflow execution: {str(e)}", important=True)
                import traceback
                debug_print(traceback.format_exc())
                # Wait a bit before retrying
                time.sleep(60)

# Example usage
if __name__ == "__main__":
    if LOCAL_MODE:
        debug_print("Running in LOCAL MODE with test repositories", important=True)
    elif not GIT_API_TOKEN:
        debug_print(f"Error: Git API token not set in config.yaml", important=True)
        debug_print(f"API token required when not in local mode. Set git.api_token in config.yaml or enable local_mode in config.yaml.", important=True)
        import sys
        sys.exit(1)
    else:
        debug_print(f"Running with Git integration for organization: {GIT_ORG}", important=True)
    
    # Start monitoring
    monitor_repositories()