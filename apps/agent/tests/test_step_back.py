from agent.graph.step_back import unique_retrieval_queries


def test_unique_retrieval_queries_drops_blanks_and_case_dupes() -> None:
    assert unique_retrieval_queries(
        "Why did close rate drop from 40% to 20%?",
        "What causes close rate to decline when lead volume stays high?",
        "why did close rate drop from 40% to 20%?",
        "  ",
        None,
    ) == [
        "Why did close rate drop from 40% to 20%?",
        "What causes close rate to decline when lead volume stays high?",
    ]
