from huggingface_hub import get_hf_file_metadata, hf_hub_url, hf_hub_download, scan_cache_dir, whoami, list_models


def get_my_model_names(token):
    
    try:
        author = whoami(token=token)
        model_infos = list_models(author=author["name"], use_auth_token=token)
        return [model.modelId for model in model_infos], None
        
    except Exception as e:
        return [], e

def download_file(repo_id: str, filename: str, token: str):
    """Download a file from a repo on the Hugging Face Hub.

    Returns:
        file_path (:obj:`str`): The path to the downloaded file.
        revision (:obj:`str`): The commit hash of the file.
        """

    md = get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename=filename), token=token)
    revision = md.commit_hash

    file_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token)

    return file_path, revision

def delete_file(revision: str):
    """Delete a file from local cache.

    Args:
        revision (:obj:`str`): The commit hash of the file.
    Returns:
        None
        """
    scan_cache_dir().delete_revisions(revision).execute()

def get_pr_url(api, repo_id, title):
    try:
        discussions = api.get_repo_discussions(repo_id=repo_id)
    except Exception:
        return None
    for discussion in discussions:
        if (
            discussion.status == "open"
            and discussion.is_pull_request
            and discussion.title == title
        ):
            return f"https://huggingface.co/{repo_id}/discussions/{discussion.num}"