from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def feed_xml() -> str:
    return (FIXTURES / "arxiv_feed.xml").read_text()
