import pytest

from hansenkit.data import generate_synthetic, load_dataset
from hansenkit.splitting import split_dataset


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    paths = generate_synthetic(tmp_path_factory.mktemp("synthetic"))
    dataset = load_dataset(*paths)
    return paths, dataset, split_dataset(dataset)
