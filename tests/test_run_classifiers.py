import pandas as pd
import pytest

from scripts.run_classifiers import run_zeroshot


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
        save_embeddings_to="pickle",
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
            save_embeddings_to="",
            overwrite=True,
        )

    with pytest.raises(
        Exception,
        match="The first two columns of the datset has to be <id_column> and <text_column>",
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
            save_embeddings_to="",
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
            save_embeddings_to="pickle",
            overwrite=True,
        )


def test_save_load_embeddings(tmp_path):
    """Test if we ids match and are in string format."""
    data_path = tmp_path / "tests/test_ids.csv"
    data_path11 = tmp_path / "tests/test_ids_11.csv"
    data_path1 = tmp_path / "tests/test_ids_1.csv"
    data_path2 = tmp_path / "tests/test_ids_2.csv"
    data_path.parent.mkdir()
    data_path.touch()
    data_path11.touch()
    data_path1.touch()
    data_path2.touch()

    dataset = pd.DataFrame({"id": ["1-1", "1", "2"], "text": ["plant", "violence", "love"]}).to_csv(
        data_path, index=False
    )
    dataset11 = pd.DataFrame({"id": ["1-1"], "text": ["plant"]}).to_csv(data_path11, index=False)
    dataset1 = pd.DataFrame({"id": ["1"], "text": ["violence"]}).to_csv(data_path1, index=False)
    dataset2 = pd.DataFrame({"id": ["2"], "text": ["love"]}).to_csv(data_path2, index=False)

    def test_indices(format):
        ext = {"pickle": "pkl", "hdf5": "h5"}[format]

        run_zeroshot(
            data_path,
            model_name="bart",
            embeddings_file=None,
            prompt_embeddings_file=None,
            batch_size=1,
            device="cuda",
            prompt_pattern="This text is about {}",
            candidate_labels=["violence", "plant"],
            save_to=tmp_path / "tests/TEST_IDS",
            save_embeddings_to=format,
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
            save_to=tmp_path / "tests/TEST_IDS_1-1",
            save_embeddings_to=format,
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
            save_to=tmp_path / "tests/TEST_IDS_1",
            save_embeddings_to=format,
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
            save_to=tmp_path / "tests/TEST_IDS_2",
            save_embeddings_to=format,
            overwrite=True,
        )
        # Test indices in result
        results = pd.read_csv(tmp_path / "tests/results_TEST_IDS.csv", sep=",", dtype={"id": "string"})
        results_11 = pd.read_csv(tmp_path / "tests//results_TEST_IDS_1-1.csv", sep=",", dtype={"id": "string"})
        results_1 = pd.read_csv(tmp_path / "tests/results_TEST_IDS_1.csv", sep=",", dtype={"id": "string"})
        results_2 = pd.read_csv(tmp_path / "tests/results_TEST_IDS_2.csv", sep=",", dtype={"id": "string"})
        assert results.sort_values(by="id").id.to_list() == ["1", "1-1", "2"]
        assert results.loc[results.id == "1-1"].violence.to_list()[0] == results_11.violence[0]
        assert results.loc[results.id == "1"].violence.to_list()[0] == results_1.violence[0]
        assert results.loc[results.id == "2"].violence.to_list()[0] == results_2.violence[0]
        assert results.loc[results.id == "1-1"].plant.to_list()[0] == results_11.plant[0]
        assert results.loc[results.id == "1"].plant.to_list()[0] == results_1.plant[0]
        assert results.loc[results.id == "2"].plant.to_list()[0] == results_2.plant[0]

        # Test indiced of loaded embeddings
        run_zeroshot(
            data_path,
            model_name="bart",
            embeddings_file=tmp_path / f"tests/embeddings_TEST_IDS.{ext}",
            prompt_embeddings_file=tmp_path / f"tests/prompt_embeddings_TEST_IDS.{ext}",
            batch_size=1,
            device="cuda",
            prompt_pattern="This text is about {}",
            candidate_labels=["violence", "plant"],
            save_to=tmp_path / "tests/TEST_IDS_EMB",
            save_embeddings_to=format,
            overwrite=True,
        )
        results_emb = pd.read_csv(tmp_path / "tests/results_TEST_IDS_EMB.csv", sep=",", dtype={"id": "string"})
        assert (
            results.loc[results.id == "1-1"].violence.to_list()[0]
            == results_emb.loc[results.id == "1-1"].violence.to_list()[0]
        )
        assert (
            results.loc[results.id == "1"].violence.to_list()[0]
            == results_emb.loc[results.id == "1"].violence.to_list()[0]
        )
        assert (
            results.loc[results.id == "2"].violence.to_list()[0]
            == results_emb.loc[results.id == "2"].violence.to_list()[0]
        )

    # Test for pickle TODO: potentially add other formats
    test_indices("pickle")
