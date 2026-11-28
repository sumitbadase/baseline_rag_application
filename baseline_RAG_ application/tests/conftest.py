from api import build_services

import pytest


@pytest.fixture(scope="session")
def pipeline():
    pipeline, _ = build_services()
    return pipeline
