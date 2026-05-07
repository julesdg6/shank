from pathlib import Path


def test_workflow_configuration():
    workflow = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'update-readme.yml'
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
    text = readme.read_text()

    assert '<!-- readme-update:start -->' in text
    assert '<!-- readme-update:end -->' in text
