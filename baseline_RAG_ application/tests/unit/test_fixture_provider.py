from src.generation.fixture_provider import ABSTENTION, FixtureProvider
from src.generation.rag_pipeline import RagPipeline


def test_fixture_uses_question_not_unrelated_context():
    prompt = """QUESTION: How many remote-work days are permitted?

CONTEXT:
SOURCE: employee_handbook.txt
CONTENT: Full-time employees receive 20 days of annual leave.
"""

    assert FixtureProvider().generate(prompt) == ABSTENTION


def test_fixture_returns_answer_when_question_evidence_is_present():
    prompt = """QUESTION: How many remote-work days are permitted?

CONTEXT:
SOURCE: employee_handbook.txt
CONTENT: Full-time employees receive 20 days of annual leave.

SOURCE: work_policy.txt
CONTENT: Eligible personnel may work outside corporate premises twice per week.
"""

    assert FixtureProvider().generate(prompt) == (
        "Employees may work remotely twice per week [work_policy.txt]."
    )


def test_prompt_forbids_mixed_abstention_and_answer():
    prompt = RagPipeline._build_prompt("A question?", [])

    assert "Never combine the abstention sentence with an answer" in prompt
    assert "output exactly this sentence and nothing else" in prompt
