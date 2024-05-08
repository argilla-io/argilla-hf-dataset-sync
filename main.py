import os
from datasets import Dataset
import argilla as rg

# Required environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
ARGILLA_API_URL = os.getenv("ARGILLA_API_URL")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")
SOURCE_DATASET = os.getenv("SOURCE_DATASET")
RAW_DATASET = os.getenv("RAW_DATASET")

# Optional environment variables
REQUIRED_RESPONSES = int(os.getenv("REQUIRED_RESPONSES", "2"))
RESULTS_DATASET = os.getenv("RESULTS_DATASET", f"{SOURCE_DATASET}-results")
SOURCE_WORKSPACE = os.getenv("SOURCE_WORKSPACE", "admin")
RESULTS_WORKSPACE = os.getenv("RESULTS_WORKSPACE", "results")
DELETE_SOURCE_RECORDS = os.getenv("DELETE_SOURCE_RECORDS", "False").lower() == 'true'
PARSED_RESULTS_DATASET = os.getenv("PARSED_RESULTS_DATASET", f"{RAW_DATASET}-parsed")

def completed_with_overlap(records, required_responses):
    """
    Filters records to find those with responses equal to or greater than the required amount.
    """
    completed = [r for r in records if len(r.responses) >= required_responses]
    return completed

def build_parsed_results(dataset):
    """
    Constructs a new dataset from the original, extracting relevant fields and adding additional
    fields for parsed results.
    """
    questions = [(question.name, question.type) for question in dataset.questions]
    results = []
    for record in dataset:
        result = {
            "fields": dict(record.fields),
            "metadata": dict(record.metadata),
            "num_responses": len(record.responses),
            "user_ids": [str(response.user_id) for response in record.responses]
        }
        for question, _ in questions:
            result[question] = []
        for response in record.responses:
            for question, kind in questions:
                if question in response.values:
                    value = response.values[question].value
                    if value is not None:
                        if kind == 'span':
                            result[question].append([dict(v) for v in value])
                        else:
                            result[question].append(value)
        results.append(result)
    return Dataset.from_list(results)

rg.init(api_url=ARGILLA_API_URL, api_key=ARGILLA_API_KEY)

# Ensure workspace exists
if RESULTS_WORKSPACE not in [workspace.name for workspace in rg.Workspace.list()]:
    rg.Workspace.create(RESULTS_WORKSPACE)

dataset = rg.FeedbackDataset.from_argilla(SOURCE_DATASET, workspace=SOURCE_WORKSPACE)
print(f"Current dataset size: {len(dataset)}")

submitted_so_far = dataset.filter_by(response_status="submitted")
print(f"Submitted: {len(submitted_so_far)}")

completed_remote_records = completed_with_overlap(submitted_so_far.records, REQUIRED_RESPONSES)
print(f"Completed so far: {len(completed_remote_records)}")

if completed_remote_records:
    local_submitted = submitted_so_far.pull()
    completed_local_records = completed_with_overlap(local_submitted.records, REQUIRED_RESPONSES)
    try:
        results = rg.FeedbackDataset.from_argilla(RESULTS_DATASET, workspace=RESULTS_WORKSPACE)
        results.add_records(completed_local_records)
        if DELETE_SOURCE_RECORDS:
            dataset.delete_records([r.id for r in completed_remote_records])
    except Exception as e:
        results = local_submitted.push_to_argilla(RESULTS_DATASET, workspace=RESULTS_WORKSPACE)
        if DELETE_SOURCE_RECORDS:
            dataset.delete_records([r.id for r in completed_remote_records])

    parsed_results_dataset = build_parsed_results(results)
    parsed_results_dataset.push_to_hub(PARSED_RESULTS_DATASET, token=HF_TOKEN)
    results.push_to_huggingface(RAW_DATASET, token=HF_TOKEN)
    print(f"Updated and pushed datasets with {len(completed_remote_records)} records.")
