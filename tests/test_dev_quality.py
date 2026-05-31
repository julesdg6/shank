from pathlib import Path


def test_requirements_are_pinned():
    root = Path(__file__).resolve().parents[1]
    for relative_path in ('api/requirements.txt', 'worker/requirements.txt'):
        lines = (root / relative_path).read_text().splitlines()
        packages = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        assert packages
        assert all('==' in package for package in packages)


def test_ci_workflow_runs_quality_and_docker_build():
    workflow = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'ci.yml'
    text = workflow.read_text()

    assert 'name: CI' in text
    assert 'ruff check .' in text
    assert 'mypy' in text
    assert 'python -m pytest api/tests/ -v' in text
    assert 'python -m pytest worker/tests/ -v' in text
    assert 'python -m pytest tests/ -v' in text
    assert 'docker build -t shank:ci .' in text
