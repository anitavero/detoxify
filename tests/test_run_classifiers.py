import pandas as pd
import pytest

from run_classifiers import run_zeroshot


def test_run_zeroshot():
    data_path = "tests/dummy_data/test_dataset.csv"
    data_path_wrong_format = "tests/dummy_data/test_dataset_wrongcolname.csv"

    run_zeroshot(
        data_path,
        model_name="t5-small",
        embeddings_file=None,
        prompt_embeddings_file=None,
        batch_size=1,
        device="cuda",
        prompt_pattern="This text is about {}",
        candidate_labels=None,
        save_to=None,
        save_embeddings=False,
        overwrite=True,
    )

    with pytest.raises(
        Exception, match="Prompt embedding labels are not matching with given labels:\nPrompt embedding:"
    ):
        run_zeroshot(
            data_path,
            model_name="t5-small",
            embeddings_file=None,
            prompt_embeddings_file="tests/dummy_data/prompt_embeddings_t5-small_test_dataset_This_text_is_about_{}.pkl",
            batch_size=1,
            device="cuda",
            prompt_pattern="This text is about {}",
            candidate_labels=["A", "B"],
            save_to=None,
            save_embeddings=False,
            overwrite=True,
        )

    with pytest.raises(
        Exception,
        match='The first two columns of the datset has to be "id" and "text",',
    ):
        run_zeroshot(
            data_path_wrong_format,
            model_name="t5-small",
            embeddings_file=None,
            prompt_embeddings_file="tests/dummy_data/prompt_embeddings_t5-small_test_dataset_This_text_is_about_{}.pkl",
            batch_size=1,
            device="cuda",
            prompt_pattern="This text is about {}",
            candidate_labels=None,
            save_to=None,
            save_embeddings=False,
            overwrite=True,
        )

    data_path = "tests/dummy_data/test_no_classes.csv"
    dataset = pd.DataFrame({"id": ["1-1", "1", "2"], "text": ["flower", "kill", "love"]}).to_csv(data_path, index=False)
    with pytest.raises(Exception, match="Either candidate_labels should be given or"):
        run_zeroshot(
            data_path,
            model_name="t5-small",
            embeddings_file=None,
            prompt_embeddings_file=None,
            batch_size=1,
            device="cuda",
            prompt_pattern="This text is about {}",
            candidate_labels=None,
            save_to=None,
            save_embeddings=True,
            overwrite=True,
        )


def test_save_load_embeddings():
    """Test if we ids match and are in string format."""
    data_path = "tests/dummy_data/test_ids.csv"
    data_path11 = "tests/dummy_data/test_ids_11.csv"
    data_path1 = "tests/dummy_data/test_ids_1.csv"
    data_path2 = "tests/dummy_data/test_ids_2.csv"
    dataset = pd.DataFrame({"id": ["1-1", "1", "2"], "text": ["plant", "violence", "love"]}).to_csv(
        data_path, index=False
    )
    dataset11 = pd.DataFrame({"id": ["1-1"], "text": ["plant"]}).to_csv(data_path11, index=False)
    dataset1 = pd.DataFrame({"id": ["1"], "text": ["violence"]}).to_csv(data_path1, index=False)
    dataset2 = pd.DataFrame({"id": ["2"], "text": ["love"]}).to_csv(data_path2, index=False)
    run_zeroshot(
        data_path,
        model_name="bart",
        embeddings_file=None,
        prompt_embeddings_file=None,
        batch_size=1,
        device="cuda",
        prompt_pattern="This text is about {}",
        candidate_labels=["violence", "plant"],
        save_to="tests/dummy_data/TEST_IDS",
        save_embeddings=True,
        overwrite=True,
    )
    run_zeroshot(
        data_path11,
        model_name="bart",
        embeddings_file=None,
        prompt_embeddings_file=None,
        batch_size=1,
        device="cuda",
        prompt_pattern="This text is about {}",
        candidate_labels=["violence", "plant"],
        save_to="tests/dummy_data/TEST_IDS_1-1",
        save_embeddings=True,
        overwrite=True,
    )
    run_zeroshot(
        data_path1,
        model_name="bart",
        embeddings_file=None,
        prompt_embeddings_file=None,
        batch_size=1,
        device="cuda",
        prompt_pattern="This text is about {}",
        candidate_labels=["violence", "plant"],
        save_to="tests/dummy_data/TEST_IDS_1",
        save_embeddings=True,
        overwrite=True,
    )
    run_zeroshot(
        data_path2,
        model_name="bart",
        embeddings_file=None,
        prompt_embeddings_file=None,
        batch_size=1,
        device="cuda",
        prompt_pattern="This text is about {}",
        candidate_labels=["violence", "plant"],
        save_to="tests/dummy_data/TEST_IDS_2",
        save_embeddings=True,
        overwrite=True,
    )
    results = pd.read_csv("tests/dummy_data/results_TEST_IDS.csv", sep=",", dtype={"id": "string"})
    results_11 = pd.read_csv("tests/dummy_data/results_TEST_IDS_1-1.csv", sep=",", dtype={"id": "string"})
    results_1 = pd.read_csv("tests/dummy_data/results_TEST_IDS_1.csv", sep=",", dtype={"id": "string"})
    results_2 = pd.read_csv("tests/dummy_data/results_TEST_IDS_2.csv", sep=",", dtype={"id": "string"})
    assert results.sort_values(by="id").id.to_list() == ["1", "1-1", "2"]
    assert results.loc[results.id == "1-1"].violence.to_list()[0] == results_11.violence[0]
    assert results.loc[results.id == "1"].violence.to_list()[0] == results_1.violence[0]
    assert results.loc[results.id == "2"].violence.to_list()[0] == results_2.violence[0]
    assert results.loc[results.id == "1-1"].plant.to_list()[0] == results_11.plant[0]
    assert results.loc[results.id == "1"].plant.to_list()[0] == results_1.plant[0]
    assert results.loc[results.id == "2"].plant.to_list()[0] == results_2.plant[0]
