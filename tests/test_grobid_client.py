import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parsing_papers.grobid_client import GrobidClient

FIXTURE = Path(__file__).parent / "fixtures_sample.tei.xml"


def test_parse_tei_file():
    client = GrobidClient()
    parsed = client.parse_tei_file(FIXTURE)
    assert "cotton farmers" in parsed.title
    assert "1200 farmers" in parsed.abstract
    assert len(parsed.sections) == 2
    assert any("Random Forest" in s.text for s in parsed.sections)
    assert len(parsed.tables) == 1
    assert "0.87" in parsed.tables[0].text

    reduced = parsed.methods_and_results_text()
    assert "AUC of 0.87" in reduced
    assert "Table 1" in reduced
