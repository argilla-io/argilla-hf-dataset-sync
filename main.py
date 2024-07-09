import os
from datasets import Dataset, load_dataset, concatenate_datasets
import argilla as rg

# Required environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
ARGILLA_API_URL = os.getenv("ARGILLA_API_URL")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")
SOURCE_DATASET = os.getenv("SOURCE_DATASET")
PARSED_RESULTS_DATASET = os.getenv("HF_DATASET_RESULTS")
SOURCE_WORKSPACE = os.getenv("SOURCE_WORKSPACE", "admin")


client = rg.Argilla(
    api_url=ARGILLA_API_URL,
    api_key=ARGILLA_API_KEY
)

ds = client.datasets(SOURCE_DATASET, workspace=SOURCE_WORKSPACE)

# Get submitted records (at least 1 user response)
filter = rg.Filter(("response.status", "==", "submitted"))
submitted = ds.records(query=rg.Query(filter=filter))
to_delete = list(submitted)
print(f"Number of records to delete: {len(to_delete)}")
submitted = ds.records(query=rg.Query(filter=filter))
record_list = submitted.to_list(flatten=False)
print(f"Number of records to persist: {len(record_list)}")

if len(record_list)>0:
  hf_ds = Dataset.from_list(record_list)
  # we need to remove this, otherwise it fails
  hf_ds = hf_ds.remove_columns(["vectors"])
    
  # Load existing hf dataset
  previous_hf_ds = load_dataset(HF_DATASET_RESULTS, split="train")
  print(f"Current HF dataset size: {len(previous_hf_ds)}")
    
  # Add new submitted records
  concatenated = concatenate_datasets([previous_hf_ds,hf_ds])
  print(f"New HF dataset size:  {len(concatenated)}")
  concatenated.push_to_hub(HF_DATASET_RESULTS, private=True)
  print(f"New HF dataset size:  {len(concatenated)}")

  print(f"Deleting records")
  # this won't be needed with rc3 just ds.delete(to_delete)
  count = 0
  for r in to_delete:
    ds.records.delete([r])
    count +=1
    print(f"Deleted: {count}")
