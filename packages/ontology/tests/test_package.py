from ontology import CORE_ONTOLOGY, __version__


def test_ontology_package_version() -> None:
    assert __version__ == "0.1.0"
    assert CORE_ONTOLOGY.version
