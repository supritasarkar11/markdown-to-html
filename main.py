import os
import re
import json
import base64
import stat
import shutil
import asyncio
import logging
import sys
from typing import List, Optional
from datetime import datetime
import httpx
import git
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Initialize environment configuration from .env file


# ========== Configuration Management ==========
class ApplicationConfig(BaseSettings):
    GEMINI_API_KEY: str = Field(os.getenv("GEMINI_API_KEY"), env="GEMINI_API_KEY")
    GITHUB_TOKEN: str = Field(os.getenv("GITHUB_TOKEN"), env="GITHUB_TOKEN")
    GITHUB_USERNAME: str = Field(os.getenv("GITHUB_USERNAME"), env="GITHUB_USERNAME")
    STUDENT_SECRET: str = Field(os.getenv("STUDENT_SECRET"), env="STUDENT_SECRET")
    LOG_FILE_PATH: str = Field("logs/app.log", env="LOG_FILE_PATH")
    MAX_CONCURRENT_TASKS: int = Field(2, env="MAX_CONCURRENT_TASKS")
    KEEP_ALIVE_INTERVAL_SECONDS: int = Field(30, env="KEEP_ALIVE_INTERVAL_SECONDS")
    GITHUB_API_BASE: str = Field("https://api.github.com", env="GITHUB_API_BASE")
    GITHUB_PAGES_BASE: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

app_config = ApplicationConfig()
print(app_config)

# Set GitHub Pages base URL if not already configured
if not app_config.GITHUB_PAGES_BASE:
    github_pages_url = f"https://{app_config.GITHUB_USERNAME}.github.io"
    app_config.GITHUB_PAGES_BASE = github_pages_url

# ========== Logging Configuration ==========
log_directory = os.path.dirname(app_config.LOG_FILE_PATH)
os.makedirs(log_directory, exist_ok=True)

# Initialize application logger with custom name
app_logger = logging.getLogger("task_receiver")
app_logger.setLevel(logging.INFO)

# Setup console output handler
stdout_handler = logging.StreamHandler(sys.stdout)

# Setup file output handler with append mode
file_output_handler = logging.FileHandler(
    app_config.LOG_FILE_PATH, mode="a", encoding="utf-8"
)

# Define log message format with timestamp and level
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Apply formatter to both handlers
stdout_handler.setFormatter(log_formatter)
file_output_handler.setFormatter(log_formatter)

# Clear existing handlers and add new ones
app_logger.handlers = []
app_logger.addHandler(stdout_handler)
app_logger.addHandler(file_output_handler)
app_logger.propagate = False

def force_flush_all_logs():
    """Force flush all output streams and logger handlers."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    
    for handler in app_logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass

# ========== Data Models ==========
class FileAttachment(BaseModel):
    name: str
    url: str  # Can be data URI or http(s) URL

class IncomingTaskRequest(BaseModel):
    task: str
    email: str
    round: int
    brief: str
    evaluation_url: str
    nonce: str
    secret: str
    attachments: List[FileAttachment] = []

# ========== Application & Global State ==========
web_app = FastAPI(
    title="Automated Task Receiver & Processor", 
    description="LLM-driven code generation and deployment"
)
active_background_tasks: List[asyncio.Task] = []
concurrent_task_limiter = asyncio.Semaphore(app_config.MAX_CONCURRENT_TASKS)
most_recent_task: Optional[dict] = None
LLM_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# ========== Utility Functions ==========
def validate_secret_token(provided_secret: str) -> bool:
    """Verify that the provided secret matches the configured secret."""
    return provided_secret == app_config.STUDENT_SECRET

def ensure_directory_exists(directory_path: str):
    """Create directory structure if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)

def cleanup_local_directory(target_path: str):
    """Remove local directory with proper error handling."""
    if not os.path.exists(target_path):
        return
    
    def handle_removal_error(func, path_arg, exc_info):
        """Handle errors during directory removal by modifying permissions."""
        try:
            os.chmod(path_arg, stat.S_IWUSR)
            func(path_arg)
        except Exception as exc:
            app_logger.exception(f"Failed in rmtree on {path_arg}: {exc}")
            raise
    
    app_logger.info(f"[CLEANUP] Removing local directory: {target_path}")
    shutil.rmtree(target_path, onerror=handle_removal_error)
    force_flush_all_logs()

# ========== Attachment Processing Functions ==========
def check_if_image_data_uri(uri_string: str) -> bool:
    """Check if the provided string is a valid image data URI."""
    if not uri_string or not uri_string.startswith("data:"):
        return False
    pattern = r"data:image/[^;]+;base64,"
    return re.search(pattern, uri_string, re.IGNORECASE) is not None

def convert_data_uri_to_gemini_format(uri_string: str) -> Optional[dict]:
    """Convert a data URI to Gemini API compatible inline data format."""
    if not uri_string or not uri_string.startswith("data:"):
        return None
    
    # Extract MIME type and base64 data using regex
    uri_pattern = r"data:(?P<mime_type>[^;]+);base64,(?P<base64_data>.*)"
    pattern_match = re.search(uri_pattern, uri_string, re.IGNORECASE)
    
    if not pattern_match:
        return None
    
    content_mime_type = pattern_match.group("mime_type")
    content_base64_data = pattern_match.group("base64_data")
    
    # Only process image MIME types
    if not content_mime_type.startswith("image/"):
        return None
    
    return {
        "inlineData": {
            "data": content_base64_data, 
            "mimeType": content_mime_type
        }
    }

async def process_attachment_for_gemini(attachment_location: str) -> Optional[dict]:
    """Process an attachment URL and convert it to Gemini API format."""
    if not attachment_location:
        return None
    
    # Handle data URI format
    if attachment_location.startswith("data:"):
        return convert_data_uri_to_gemini_format(attachment_location)
    
    # Handle HTTP(S) URLs
    if attachment_location.startswith(("http://", "https://")):
        try:
            async with httpx.AsyncClient(timeout=15) as http_client:
                response = await http_client.get(attachment_location)
                response.raise_for_status()
                
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    app_logger.info(
                        f"[ATTACHMENT] Skipping non-image MIME: {content_type}"
                    )
                    return None
                
                encoded_content = base64.b64encode(response.content).decode("utf-8")
                return {
                    "inlineData": {
                        "data": encoded_content, 
                        "mimeType": content_type
                    }
                }
        except Exception as error:
            app_logger.warning(
                f"[ATTACHMENT] Failed to fetch/encode attachment "
                f"{attachment_location}: {error}"
            )
            return None
    
    return None

# ========== File System Operations ==========
async def persist_generated_files(task_identifier: str, file_contents: dict) -> str:
    """Save generated files to local file system."""
    workspace_dir = os.path.join(os.getcwd(), "generated_tasks")
    task_specific_dir = os.path.join(workspace_dir, task_identifier)
    ensure_directory_exists(task_specific_dir)
    
    app_logger.info(f"[LOCAL_SAVE] Saving generated files to: {task_specific_dir}")
    
    for file_name, file_content in file_contents.items():
        full_file_path = os.path.join(task_specific_dir, file_name)
        try:
            with open(full_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(file_content)
            app_logger.info(
                f" -> Saved: {file_name} (bytes: {len(file_content)})"
            )
        except Exception as save_error:
            app_logger.exception(
                f"Failed to save generated file {file_name}: {save_error}"
            )
            raise
    
    force_flush_all_logs()
    return task_specific_dir

async def persist_attachments_locally(
    task_directory: str, 
    attachment_list: List[FileAttachment]
) -> List[str]:
    """Download and save attachments to local file system."""
    successfully_saved = []
    app_logger.info(
        f"[ATTACHMENTS] Processing {len(attachment_list)} attachments "
        f"for {task_directory}"
    )
    
    async with httpx.AsyncClient(timeout=30) as http_client:
        for attachment_item in attachment_list:
            file_name = attachment_item.name
            file_url = attachment_item.url
            file_binary_data = None
            
            if not file_name or not file_url:
                app_logger.warning(
                    f"Skipping invalid attachment entry: {file_name}"
                )
                continue
            
            try:
                # Process data URI format
                if file_url.startswith("data:"):
                    base64_match = re.search(
                        r"base64,(.*)", file_url, re.IGNORECASE
                    )
                    if base64_match:
                        file_binary_data = base64.b64decode(
                            base64_match.group(1)
                        )
                
                # Process HTTP(S) URLs
                elif file_url.startswith(("http://", "https://")):
                    download_response = await http_client.get(file_url)
                    download_response.raise_for_status()
                    file_binary_data = download_response.content
                
                if file_binary_data is None:
                    app_logger.warning(
                        f"No content for attachment: {file_name}"
                    )
                    continue
                
                full_path = os.path.join(task_directory, file_name)
                with open(full_path, "wb") as binary_file:
                    binary_file.write(file_binary_data)
                
                app_logger.info(
                    f" -> Saved Attachment: {file_name} "
                    f"(bytes: {len(file_binary_data)})"
                )
                successfully_saved.append(file_name)
                
            except Exception as attachment_error:
                app_logger.exception(
                    f"Failed to save attachment {file_name}: "
                    f"{attachment_error}"
                )
    
    force_flush_all_logs()
    return successfully_saved

# ========== GitHub Repository Management ==========
async def initialize_or_clone_repository(
    local_directory: str, 
    repository_name: str, 
    auth_repo_url: str, 
    public_repo_url: str, 
    iteration_number: int
) -> git.Repo:
    """Setup local Git repository - create new or clone existing."""
    github_access_token = app_config.GITHUB_TOKEN
    api_headers = {
        "Authorization": f"token {github_access_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    async with httpx.AsyncClient(timeout=45) as api_client:
        try:
            if iteration_number == 1:
                app_logger.info(
                    f"[GIT] R1: Creating remote repo '{repository_name}'"
                )
                
                # Prepare repository creation payload
                repo_creation_payload = {
                    "name": repository_name, 
                    "private": False, 
                    "auto_init": True
                }
                
                # Attempt to create repository
                creation_response = await api_client.post(
                    f"{app_config.GITHUB_API_BASE}/user/repos", 
                    json=repo_creation_payload, 
                    headers=api_headers
                )
                
                # Handle case where repository already exists
                if creation_response.status_code == 422:
                    app_logger.warning(
                        f"[GIT] Repository '{repository_name}' already exists, "
                        f"using existing repository"
                    )
                else:
                    creation_response.raise_for_status()
                    app_logger.info("[GIT] Repository created successfully")
                
                # Clone the repository to local directory
                app_logger.info(
                    f"[GIT] Cloning existing repository {public_repo_url}"
                )
                local_repo = git.Repo.clone_from(auth_repo_url, local_directory)
                app_logger.info("[GIT] Local repo cloned")
            else:
                app_logger.info(
                    f"[GIT] R{iteration_number}: Cloning {public_repo_url}"
                )
                local_repo = git.Repo.clone_from(auth_repo_url, local_directory)
                app_logger.info("[GIT] Cloned repo")
            
            force_flush_all_logs()
            return local_repo
            
        except httpx.HTTPStatusError as http_error:
            error_text = getattr(http_error.response, "text", "")
            app_logger.exception(f"GitHub API error: {error_text}")
            raise
        except git.GitCommandError as git_error:
            app_logger.exception(f"Git command error: {git_error}")
            raise

async def commit_and_deploy_to_github(
    repository: git.Repo, 
    task_identifier: str, 
    iteration_number: int, 
    repository_name: str
) -> dict:
    """Commit changes and deploy to GitHub Pages."""
    github_user = app_config.GITHUB_USERNAME
    github_access_token = app_config.GITHUB_TOKEN
    
    api_headers = {
        "Authorization": f"token {github_access_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    public_repo_url = f"https://github.com/{github_user}/{repository_name}"
    
    async with httpx.AsyncClient(timeout=45) as api_client:
        try:
            # Stage all changes
            repository.git.add(A=True)
            
            # Create commit with descriptive message
            commit_description = (
                f"Task {task_identifier} - Round {iteration_number}: "
                f"automated update"
            )
            repository.index.commit(commit_description)
            current_commit_sha = repository.head.object.hexsha
            app_logger.info(f"[GIT] Committed: {current_commit_sha}")
            
            # Ensure main branch and push
            repository.git.branch("-M", "main")
            repository.git.push(
                "--set-upstream", "origin", "main", force=True
            )
            app_logger.info("[GIT] Pushed to origin/main")
            
            # Configure GitHub Pages with retry logic
            pages_config_url = (
                f"{app_config.GITHUB_API_BASE}/repos/{github_user}/"
                f"{repository_name}/pages"
            )
            pages_config_payload = {
                "source": {"branch": "main", "path": "/"}
            }
            
            max_retry_attempts = 5
            retry_base_delay = 3
            
            for attempt_number in range(max_retry_attempts):
                try:
                    # Check if Pages is already configured
                    check_response = await api_client.get(
                        pages_config_url, headers=api_headers
                    )
                    pages_already_configured = (
                        check_response.status_code == 200
                    )
                    
                    if pages_already_configured:
                        await api_client.put(
                            pages_config_url, 
                            json=pages_config_payload, 
                            headers=api_headers
                        )
                    else:
                        await api_client.post(
                            pages_config_url, 
                            json=pages_config_payload, 
                            headers=api_headers
                        )
                    
                    app_logger.info("[GIT] Pages configured")
                    break
                    
                except httpx.HTTPStatusError as pages_error:
                    error_text = getattr(pages_error.response, "text", "")
                    is_timing_issue = (
                        pages_error.response.status_code == 422 and 
                        "main branch must exist" in error_text and 
                        attempt_number < max_retry_attempts - 1
                    )
                    
                    if is_timing_issue:
                        retry_delay = retry_base_delay * (2 ** attempt_number)
                        app_logger.warning(
                            f"[GIT] Pages timing issue, retrying in "
                            f"{retry_delay}s"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    app_logger.exception(
                        f"[GIT] Pages configuration failed: {error_text}"
                    )
                    raise
            
            # Allow time for Pages deployment
            await asyncio.sleep(5)
            
            deployed_pages_url = (
                f"{app_config.GITHUB_PAGES_BASE}/{repository_name}/"
            )
            
            force_flush_all_logs()
            
            return {
                "repo_url": public_repo_url, 
                "commit_sha": current_commit_sha, 
                "pages_url": deployed_pages_url
            }
            
        except git.GitCommandError as git_error:
            app_logger.exception(
                "Git operation failed during deployment."
            )
            raise
        except httpx.HTTPStatusError as http_error:
            app_logger.exception(
                "GitHub API error during deployment."
            )
            raise

# ========== LLM API Integration ==========
async def invoke_gemini_llm(
    content_parts: list, 
    system_instructions: str, 
    output_schema: dict, 
    max_retry_attempts: int = 3, 
    request_timeout: int = 60
) -> dict:
    """Call Gemini API with retry logic and error handling."""
    api_request_payload = {
        "contents": content_parts,
        "systemInstruction": {"parts": [{"text": system_instructions}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": output_schema
        }
    }
    
    initial_retry_delay = 1
    
    for attempt_index in range(max_retry_attempts):
        try:
            if not app_config.GEMINI_API_KEY:
                raise Exception("GEMINI_API_KEY not configured.")
            
            api_url_with_key = (
                f"{LLM_API_ENDPOINT}?key={app_config.GEMINI_API_KEY}"
            )
            
            async with httpx.AsyncClient(
                timeout=request_timeout
            ) as api_client:
                llm_response = await api_client.post(
                    api_url_with_key, 
                    json=api_request_payload, 
                    headers={"Content-Type": "application/json"}
                )
                llm_response.raise_for_status()
                
                response_data = llm_response.json()
                response_candidates = response_data.get("candidates", [])
                
                if not response_candidates:
                    raise ValueError("No candidates in LLM response")
                
                first_candidate = response_candidates[0]
                content_data = first_candidate.get("content", {})
                content_parts_list = content_data.get("parts", [])
                
                if not content_parts_list:
                    raise ValueError("No content parts in candidate")
                
                json_response_text = content_parts_list[0].get("text")
                return json.loads(json_response_text)
                
        except httpx.HTTPStatusError as http_error:
            app_logger.warning(
                f"[GEMINI] HTTP error attempt {attempt_index+1}: "
                f"{http_error}"
            )
        except (
            httpx.RequestError, 
            KeyError, 
            json.JSONDecodeError, 
            ValueError
        ) as processing_error:
            app_logger.warning(
                f"[GEMINI] Processing error attempt {attempt_index+1}: "
                f"{processing_error}"
            )
        
        if attempt_index < max_retry_attempts - 1:
            retry_delay = initial_retry_delay * (2 ** attempt_index)
            app_logger.info(f"[GEMINI] Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
    
    raise Exception("LLM generation failed after retries")

# ========== Round 2 Surgical Update Logic ==========
async def perform_surgical_update_round2(
    task_identifier: str, 
    task_brief: str, 
    current_index_html: str
) -> dict:
    """Perform minimal surgical updates to existing code for Round 2."""
    system_instructions = (
        "You are an expert full-stack engineer tasked with making "
        "SURGICAL and MINIMAL changes. "
        "Your MOST CRITICAL instruction is to preserve the existing "
        "application's core logic and structure. "
        "Only apply the specific changes requested in the 'New Brief'. "
        "Return a JSON object with 'index.html', 'README.md', and 'LICENSE'. "
        "If README.md / LICENSE exist, copy them verbatim unless a change "
        "is strictly required."
    )
    
    output_structure = {
        "type": "OBJECT",
        "properties": {
            "index.html": {"type": "STRING"},
            "README.md": {"type": "STRING"},
            "LICENSE": {"type": "STRING"},
        },
        "required": ["index.html", "README.md", "LICENSE"]
    }
    
    update_prompt = (
        f"UPDATE INSTRUCTION (SAFE MODE):\n\n"
        f"New Brief: {task_brief}\n\n"
        f"--- EXISTING index.html START ---\n"
        f"{current_index_html}\n"
        f"--- EXISTING index.html END ---\n\n"
        "Only make the minimal changes necessary to implement the brief. "
        "Do NOT remove or break core scripts, event handlers, or layout. "
        "Return FULL JSON with 'index.html', 'README.md', 'LICENSE'. "
        "If you make no changes to README/LICENSE, copy their existing contents."
    )
    
    llm_content = [{"parts": [{"text": update_prompt}]}]
    
    try:
        llm_result = await invoke_gemini_llm(
            content_parts=llm_content, 
            system_instructions=system_instructions, 
            output_schema=output_structure, 
            max_retry_attempts=4, 
            request_timeout=90
        )
    except Exception as llm_error:
        app_logger.exception(f"[ROUND2] LLM call failed: {llm_error}")
        
        # Fallback: return existing content
        return {
            "index.html": current_index_html or "",
            "README.md": "", 
            "LICENSE": ""
        }
    
    # Safety validation checks
    updated_html = (llm_result.get("index.html") or "").strip()
    
    if not updated_html:
        app_logger.warning(
            "[SAFE] LLM returned empty index.html — reverting to existing."
        )
        llm_result["index.html"] = current_index_html
    else:
        # Prevent destructive rewrites by checking size reduction
        try:
            original_length = len(current_index_html or "")
            updated_length = len(updated_html)
            minimum_acceptable_length = max(200, int(original_length * 0.3))
            
            is_too_small = (
                original_length > 0 and 
                updated_length < minimum_acceptable_length
            )
            
            if is_too_small:
                app_logger.warning(
                    "[SAFE] LLM index.html appears destructive (too small). "
                    "Reverting to existing."
                )
                llm_result["index.html"] = current_index_html
        except Exception:
            llm_result["index.html"] = current_index_html
    
    # Ensure README and LICENSE are present
    llm_result["README.md"] = llm_result.get("README.md") or ""
    llm_result["LICENSE"] = llm_result.get("LICENSE") or ""
    
    return llm_result

# ========== Evaluation Server Notification ==========
async def send_notification_to_evaluator(
    evaluator_endpoint: str, 
    user_email: str, 
    task_identifier: str, 
    iteration_number: int, 
    task_nonce: str, 
    repository_url: str, 
    commit_hash: str, 
    pages_deployment_url: str
) -> bool:
    """Notify evaluation server with task completion details."""
    notification_data = {
        "email": user_email,
        "task": task_identifier,
        "round": iteration_number,
        "nonce": task_nonce,
        "repo_url": repository_url,
        "commit_sha": commit_hash,
        "pages_url": pages_deployment_url
    }
    
    max_notification_attempts = 3
    initial_delay = 1
    
    app_logger.info(
        f"[NOTIFY] Notifying evaluation server at {evaluator_endpoint}"
    )
    
    for attempt_index in range(max_notification_attempts):
        try:
            async with httpx.AsyncClient(timeout=10) as http_client:
                notification_response = await http_client.post(
                    evaluator_endpoint, 
                    json=notification_data
                )
                notification_response.raise_for_status()
                
                app_logger.info(
                    f"[NOTIFY] Notification succeeded: "
                    f"{notification_response.status_code}"
                )
                force_flush_all_logs()
                return True
                
        except httpx.HTTPStatusError as http_error:
            app_logger.warning(
                f"[NOTIFY] HTTP error attempt {attempt_index+1}: "
                f"{http_error}"
            )
        except httpx.RequestError as request_error:
            app_logger.warning(
                f"[NOTIFY] Request error attempt {attempt_index+1}: "
                f"{request_error}"
            )
        
        if attempt_index < max_notification_attempts - 1:
            retry_delay = initial_delay * (2 ** attempt_index)
            await asyncio.sleep(retry_delay)
    
    app_logger.error(
        "[NOTIFY] Failed to notify evaluation server after retries."
    )
    force_flush_all_logs()
    return False

# ========== Main Task Orchestration ==========
async def orchestrate_generation_and_deployment(
    incoming_task: IncomingTaskRequest
):
    """Main orchestration function for code generation and deployment."""
    semaphore_acquired = False
    
    try:
        await concurrent_task_limiter.acquire()
        semaphore_acquired = True
        
        app_logger.info(
            f"[PROCESS START] Task: {incoming_task.task} "
            f"Round: {incoming_task.round}"
        )
        force_flush_all_logs()
        
        # Extract task details
        task_id = incoming_task.task
        user_email = incoming_task.email
        current_round = incoming_task.round
        task_description = incoming_task.brief
        evaluator_url = incoming_task.evaluation_url
        task_nonce_value = incoming_task.nonce
        task_attachments = incoming_task.attachments or []
        
        # Prepare repository details
        repo_id = task_id.replace(" ", "-").lower()
        github_user = app_config.GITHUB_USERNAME
        github_token = app_config.GITHUB_TOKEN
        
        authenticated_repo_url = (
            f"https://{github_user}:{github_token}@github.com/"
            f"{github_user}/{repo_id}.git"
        )
        public_repo_url = (
            f"https://github.com/{github_user}/{repo_id}"
        )
        
        workspace_base = os.path.join(os.getcwd(), "generated_tasks")
        task_local_path = os.path.join(workspace_base, task_id)
        
        # Clean up existing local directory if present
        if os.path.exists(task_local_path):
            try:
                cleanup_local_directory(task_local_path)
            except Exception as cleanup_error:
                app_logger.exception(
                    f"Cleanup failed for {task_local_path}: {cleanup_error}"
                )
                raise
        
        ensure_directory_exists(task_local_path)
        
        # Initialize or clone repository
        git_repository = await initialize_or_clone_repository(
            task_local_path, 
            repo_id, 
            authenticated_repo_url, 
            public_repo_url, 
            current_round
        )
        
        # Process attachments for LLM
        attachment_image_parts = []
        for attachment_item in task_attachments:
            gemini_part = await process_attachment_for_gemini(
                attachment_item.url
            )
            if gemini_part:
                attachment_image_parts.append(gemini_part)
        
        # Build attachment reference documentation
        attachment_reference_text = ""
        if task_attachments:
            attachment_reference_text = (
                "\nThe following attachments are provided "
                "(saved in the same folder):\n"
            )
            for attachment_item in task_attachments:
                attachment_reference_text += f"- {attachment_item.name}\n"
            attachment_reference_text += (
                "Use these exact file names in your HTML (for example: "
                '<img src="filename.png">). Do NOT rename or use external '
                "links.\n"
            )
        
        # Execute Round 1 or Round 2+ logic
        if current_round == 1:
            app_logger.info("[WORKFLOW] Round 1: full generation")
            
            # Enhance task description with attachment information
            enhanced_task_description = (
                f"{task_description}\n\n{attachment_reference_text}"
            ).strip()
            
            system_prompt_text = (
                "You are an expert full-stack engineer. Produce a JSON "
                "object with keys 'index.html', 'README.md', and 'LICENSE'. "
                "index.html must be a single-file responsive HTML app using "
                "Tailwind CSS. "
                "If image attachments are mentioned below, reference them "
                "using exactly as provided. "
                "README.md should be professional, LICENSE should contain "
                "the full MIT license text."
            )
            
            json_output_schema = {
                "type": "OBJECT",
                "properties": {
                    "index.html": {"type": "STRING"},
                    "README.md": {"type": "STRING"},
                    "LICENSE": {"type": "STRING"},
                },
                "required": ["index.html", "README.md", "LICENSE"],
            }
            
            llm_content_input = []
            if attachment_image_parts:
                combined_parts = attachment_image_parts + [
                    {"text": enhanced_task_description}
                ]
                llm_content_input.append({"parts": combined_parts})
            else:
                llm_content_input.append({
                    "parts": [{"text": enhanced_task_description}]
                })
            
            generated_files = await invoke_gemini_llm(
                content_parts=llm_content_input,
                system_instructions=system_prompt_text,
                output_schema=json_output_schema,
                max_retry_attempts=4,
                request_timeout=120,
            )
            
        else:
            app_logger.info(
                "[WORKFLOW] Round 2+: surgical update (Base.py style). "
                "Loading existing index.html only."
            )
            
            # Load existing index.html for context
            existing_html_content = ""
            index_file_path = os.path.join(task_local_path, "index.html")
            
            if os.path.exists(index_file_path):
                try:
                    with open(
                        index_file_path, "r", encoding="utf-8"
                    ) as html_file:
                        existing_html_content = html_file.read()
                    app_logger.info(
                        "[WORKFLOW] Read existing index.html for context."
                    )
                except Exception as read_error:
                    app_logger.warning(
                        f"[WORKFLOW] Could not read existing index.html: "
                        f"{read_error}"
                    )
                    existing_html_content = ""
            
            # Add attachment information to Round 2 prompt
            enhanced_brief_with_attachments = (
                f"{task_description}\n\n{attachment_reference_text}"
            ).strip()
            
            generated_files = await perform_surgical_update_round2(
                task_identifier=task_id, 
                task_brief=enhanced_brief_with_attachments, 
                current_index_html=existing_html_content
            )
            
            # Preserve existing README/LICENSE if not returned by LLM
            readme_file_path = os.path.join(task_local_path, "README.md")
            license_file_path = os.path.join(task_local_path, "LICENSE")
            
            if not generated_files.get("README.md"):
                if os.path.exists(readme_file_path):
                    with open(
                        readme_file_path, "r", encoding="utf-8"
                    ) as readme_file:
                        generated_files["README.md"] = readme_file.read()
            
            if not generated_files.get("LICENSE"):
                if os.path.exists(license_file_path):
                    with open(
                        license_file_path, "r", encoding="utf-8"
                    ) as license_file:
                        generated_files["LICENSE"] = license_file.read()
        
        # Save generated files to disk
        await persist_generated_files(task_id, generated_files)
        
        # Save attachments to repository directory
        await persist_attachments_locally(
            os.path.join(workspace_base, task_id), 
            task_attachments
        )
        
        # Commit changes and deploy to GitHub Pages
        deployment_result = await commit_and_deploy_to_github(
            git_repository, 
            task_id, 
            current_round, 
            repo_id
        )
        
        # Send notification to evaluation server
        await send_notification_to_evaluator(
            evaluator_endpoint=evaluator_url,
            user_email=user_email,
            task_identifier=task_id,
            iteration_number=current_round,
            task_nonce=task_nonce_value,
            repository_url=deployment_result["repo_url"],
            commit_hash=deployment_result["commit_sha"],
            pages_deployment_url=deployment_result["pages_url"],
        )
        
        app_logger.info(
            f"[DEPLOYMENT] Success. Repo: {deployment_result['repo_url']} "
            f"Pages: {deployment_result['pages_url']}"
        )
        
    except Exception as orchestration_error:
        app_logger.exception(
            f"[CRITICAL FAILURE] Task "
            f"{getattr(incoming_task, 'task', 'unknown')} failed: "
            f"{orchestration_error}"
        )
    finally:
        if semaphore_acquired:
            concurrent_task_limiter.release()
        force_flush_all_logs()
        app_logger.info(
            f"[PROCESS END] Task: "
            f"{getattr(incoming_task, 'task', 'unknown')} "
            f"Round: {getattr(incoming_task, 'round', 'unknown')}"
        )

# ========== API Endpoint Handlers ==========
def handle_task_completion(background_task: asyncio.Task):
    """Callback to handle background task completion."""
    try:
        task_exception = background_task.exception()
        if task_exception:
            app_logger.error(
                f"[BACKGROUND TASK] Task finished with exception: "
                f"{task_exception}"
            )
            app_logger.exception(task_exception)
        else:
            app_logger.info("[BACKGROUND TASK] Task finished successfully.")
    except asyncio.CancelledError:
        app_logger.warning("[BACKGROUND TASK] Task was cancelled.")
    finally:
        force_flush_all_logs()

@web_app.post("/ready", status_code=200)
async def handle_incoming_task(
    task_request: IncomingTaskRequest, 
    request: Request
):
    """Endpoint to receive and process incoming tasks."""
    global most_recent_task, active_background_tasks
    
    # Validate secret token
    if not validate_secret_token(task_request.secret):
        client_host = (
            request.client.host if request.client else "unknown"
        )
        app_logger.warning(
            f"Unauthorized attempt for task {task_request.task} "
            f"from {client_host}"
        )
        raise HTTPException(
            status_code=401, 
            detail="Unauthorized: Secret mismatch"
        )
    
    # Store task metadata
    brief_preview = (
        (task_request.brief[:250] + "...") 
        if len(task_request.brief) > 250 
        else task_request.brief
    )
    
    most_recent_task = {
        "task": task_request.task,
        "email": task_request.email,
        "round": task_request.round,
        "brief": brief_preview,
        "time": datetime.utcnow().isoformat() + "Z"
    }
    
    # Create and schedule background task
    new_background_task = asyncio.create_task(
        orchestrate_generation_and_deployment(task_request)
    )
    new_background_task.add_done_callback(handle_task_completion)
    active_background_tasks.append(new_background_task)
    
    app_logger.info(
        f"Received task {task_request.task}. Background processing started."
    )
    force_flush_all_logs()
    
    return JSONResponse(
        status_code=200, 
        content={
            "status": "ready", 
            "message": (
                f"Task {task_request.task} received and processing started."
            )
        }
    )

@web_app.get("/")
async def root_endpoint():
    """Root endpoint providing service information."""
    return {
        "message": (
            "Task Receiver Service running. POST /ready to submit."
        )
    }

@web_app.get("/status")
async def get_service_status():
    """Get current service status and task information."""
    global most_recent_task, active_background_tasks
    
    if most_recent_task:
        # Clean up completed tasks
        active_background_tasks[:] = [
            task for task in active_background_tasks if not task.done()
        ]
        
        return {
            "last_received_task": most_recent_task, 
            "running_background_tasks": len(active_background_tasks)
        }
    
    return {"message": "Awaiting first task submission to /ready"}

@web_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@web_app.get("/logs")
async def retrieve_logs(lines: int = Query(200, ge=1, le=5000)):
    """Retrieve recent log entries."""
    log_file_location = app_config.LOG_FILE_PATH
    
    if not os.path.exists(log_file_location):
        return PlainTextResponse("Log file not found.", status_code=404)
    
    try:
        with open(log_file_location, "rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            total_file_size = log_file.tell()
            
            log_buffer = bytearray()
            chunk_size = 1024
            chunks_read = 0
            max_chunks = 1024
            max_buffer_size = lines * 2000
            
            # Read from end of file backwards
            while (
                total_file_size > 0 and 
                len(log_buffer) < max_buffer_size and 
                chunks_read < max_chunks
            ):
                bytes_to_read = min(chunk_size, total_file_size)
                log_file.seek(total_file_size - bytes_to_read)
                log_buffer.extend(log_file.read(bytes_to_read))
                total_file_size -= bytes_to_read
                chunks_read += 1
            
            # Decode and extract last N lines
            decoded_text = log_buffer.decode(errors="ignore")
            all_lines = decoded_text.splitlines()
            requested_lines = "\n".join(all_lines[-lines:])
            
            return PlainTextResponse(requested_lines)
            
    except Exception as log_read_error:
        app_logger.exception(f"Error reading log file: {log_read_error}")
        return PlainTextResponse(
            f"Error reading log file: {log_read_error}", 
            status_code=500
        )

# ========== Application Lifecycle Events ==========
@web_app.on_event("startup")
async def on_application_startup():
    """Execute tasks when application starts."""
    async def maintain_service_heartbeat():
        """Periodic heartbeat to keep service alive."""
        while True:
            try:
                app_logger.info("[KEEPALIVE] Service heartbeat")
                force_flush_all_logs()
            except Exception:
                pass
            await asyncio.sleep(app_config.KEEP_ALIVE_INTERVAL_SECONDS)
    
    asyncio.create_task(maintain_service_heartbeat())

@web_app.on_event("shutdown")
async def on_application_shutdown():
    """Execute cleanup tasks when application shuts down."""
    app_logger.info(
        "[SHUTDOWN] Waiting for background tasks to finish "
        "(graceful shutdown)..."
    )
    
    for background_task in active_background_tasks:
        if not background_task.done():
            try:
                background_task.cancel()
            except Exception:
                pass
    
    await asyncio.sleep(0.5)
    force_flush_all_logs()
