# Agentic Code Review System

> ⚠️ **DISCLAIMER: THIS IS A VULNERABLE DEMO SYSTEM FOR EDUCATIONAL PURPOSES ONLY** ⚠️

This system is intentionally designed with security vulnerabilities to demonstrate security issues in agentic systems.
It accompanies our threat modeling [blogpost](https://www.cyberark.com/resources/threat-research-blog/agents-under-attack-threat-modeling-agentic-ai).

**DO NOT USE THIS IN PRODUCTION OR WITH SENSITIVE REPOSITORIES**

A multi-agent system built with LangGraph that automatically analyzes repositories for policy violations, prioritizes issues, implements fixes, and directly merges changes to the main branch.

## Architecture

The system is implemented in src/main.py and consists of a four-agent system:

- **Analysis Agent** - Selects repositories most likely to have policy violations and prioritizes issues
- **Code Review Agent** - Checks selected repositories for policy violations  
- **Developer Agent** - Fixes identified issues
- **Commit Agent** - Creates temporary branches, commits fixes, and directly merges to the main branch

The system proactively scans repositories in an organization and uses LLM to select high-risk repositories for policy checks.

## Features

- Checks code against company policies stored in a local JSON file
- Automatically fixes non-compliant code
- Creates temporary branches and directly merges fixes to the main branch
- Support for GitHub and custom Git servers via API

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Edit the configuration in `config.yaml`:
   ```yaml
   # Git server settings
   git:
     # API token for Git server access
     api_token: "your-git-token-here"
     # Base URL for Git API calls - leave empty to use github.com, or set for custom server
     api_base_url: ""  # e.g., "http://10.211.55.13:3000/api/v1" for a custom Git server
     # Organization or project name
     org: "your-git-org"  # Replace with your organization/project name
   
   # General settings
   general:
     polling_interval: 300  # 5 minutes
     debug_mode: false
     local_mode: false  # Set to true to use local test repositories instead of GitHub
   
   # Local mode settings
   local:
     repos_path: "test_repos"  # Path relative to project root
   
   # Model settings
   model:
     name: "gpt-4o-mini"  # Model to use: gpt-4o-mini, gpt-3.5-turbo, etc.
     api_token: "your-openai-api-token-here"
   ```

3. Edit company policies in `company_policy.json`

4. Run the system:
   ```
   python src/main.py
   ```

## Customization

All configuration is stored in `config.yaml`. You can customize:

- Polling interval in `general.polling_interval` (default is 5 minutes)
- Debug mode in `general.debug_mode`
- Local mode in `general.local_mode`
- Git organization and API base URL in the `git` section
- LLM model in `model.name`

You can also customize:
- The issue detection logic in the agents
- The number of files sampled when scanning repositories
- Company policies in `company_policy.json`

## Debug Mode

The system includes a detailed debug mode that displays:

- LLM inputs and outputs with timing information
- State transitions between agents
- Decision-making process details
- Repository and issue selection rationale

Enable debug mode by setting `debug_mode: true` in the `config.yaml` file:
```yaml
general:
  debug_mode: true
```

This is useful for:
- Understanding the system's decision-making process
- Debugging issues with the workflow
- Testing new agent prompts or configurations

## Testing

The system includes a local testing mode:

### Local Repository Mode

This mode analyzes local files instead of Git repositories:

- Set `LOCAL_MODE = True` in the `config.yaml` file:
  ```yaml
  general:
    local_mode: true
  ```
- Place test code in the `test_repos/` directory
- The system will scan these directories as if they were repositories
- The commit agent will print what would be committed rather than making actual changes

When not in local mode, the system requires a valid GIT_API_TOKEN environment variable to be set.

## Contributing
Currently we are not seeking for active contribution and maintainers, please use the issues feature to open feature requests and bug reports

## License  
Copyright (c) 2025 CyberArk Software Ltd. All rights reserved  
This repository is licensed under  Apache-2.0 License - see [`LICENSE`](LICENSE.txt) for more details.
