from pathlib import Path


def test_workflow_configuration():
    workflow = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'update-readme.yml'
    assert workflow.exists()
    text = workflow.read_text()

    assert 'name: Update README' in text
    assert 'branches:' in text
    assert '- master' in text
    assert 'workflow_dispatch:' in text
    assert 'contents: write' in text
    assert "github.actor != 'github-actions[bot]'" in text
    assert "README.md" in text
    assert '[skip readme-update]' in text


def test_readme_contains_workflow_update_markers():
    readme = Path(__file__).resolve().parents[1] / 'README.md'
    assert readme.exists()
    text = readme.read_text()

    assert '<!-- readme-update:start -->' in text
    assert '<!-- readme-update:end -->' in text


def test_readme_documents_stem_model_setup():
    readme = Path(__file__).resolve().parents[1] / 'README.md'
    text = readme.read_text()

    assert '### Setup' in text
    assert 'python3 scripts/download_stem_models.py --6stems' in text
    assert '### Troubleshooting' in text


def test_readme_documents_unraid_setup_and_assets():
    repo_root = Path(__file__).resolve().parents[1]
    readme = repo_root / 'README.md'
    text = readme.read_text()

    assert '## 🖥️ Unraid 7+ setup' in text
    assert '/boot/config/plugins/dockerMan/templates-user/shank.xml' in text
    assert '--gpus all' in text

    assert (repo_root / 'unraid' / 'shank.xml').exists()
    assert (repo_root / 'unraid' / 'shank-icon.png').exists()
