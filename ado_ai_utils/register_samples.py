from azureml.core.dataset import Dataset


def register_improvements_sample(blob_datastore, ws):
    datastore_paths = [(blob_datastore, 'improvements.csv')]
    ai_impact_scores = Dataset.Tabular.from_delimited_files(path=datastore_paths)
    ai_impact_scores.register(workspace=ws,
                              name="improvements_sample",
                              description="subset of improvements items from SQL")

    return ai_impact_scores


def register_feedback_sample(blob_datastore, ws):
    datastore_paths = [(blob_datastore, 'feedback_items.csv')]
    ai_impact_scores = Dataset.Tabular.from_delimited_files(path=datastore_paths)
    ai_impact_scores.register(workspace=ws,
                              name="feedback_items_sample",
                              description="subset of feedback items from SQL")

    return ai_impact_scores


def register_ai_feedback_sample(blob_datastore, ws):
    datastore_paths = [(blob_datastore, 'ai_impact_scores.csv')]

    ai_impact_scores = Dataset.Tabular.from_delimited_files(path=datastore_paths)

    ai_impact_scores.register(workspace=ws,
                              name="ai_impact_scores",
                              description="ai subset of feedback items")

    return ai_impact_scores
