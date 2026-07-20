import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parsing_papers.grobid_client import GrobidClient

FIXTURE = Path(__file__).parent / "fixtures_empty_table.tei.xml"


def test_tables_with_negligible_content_are_flagged_as_empty():
    """Tabelas cujo GROBID so capturou 1-2 palavras (ex: 'Importance ranking')
    sao efetivamente inuteis e devem ser tratadas como vazias, nao como
    conteudo real -- caso real observado em s10479-025-06998-7 onde Table 4
    e Table 11 vieram com so uma palavra e escapavam do filtro anterior."""
    client = GrobidClient()
    parsed = client.parse_tei_file(FIXTURE)

    assert len(parsed.tables) == 0
    # labels agora incluem sufixo de posicao "(#N de M tabelas no PDF)" para
    # desambiguar tabelas sem <head>/<label> proprio -- checa por substring.
    assert any("Table 4" in label for label in parsed.empty_table_labels)
    assert any("Table 5" in label for label in parsed.empty_table_labels)


def test_empty_table_warning_included_in_llm_text():
    client = GrobidClient()
    parsed = client.parse_tei_file(FIXTURE)
    reduced = parsed.methods_and_results_text()

    assert "AVISO DE PARSING" in reduced
    assert "Table 4" in reduced
    assert "Table 5" in reduced
    # o texto corrido com o AUC real deve continuar presente para o LLM achar
    assert "AUC of 0.93" in reduced
