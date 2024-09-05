from datasets import load_dataset, load_from_disk
from transformers import LlamaForConditionalGeneration, LlamaTokenizer
from collections import defaultdict

med_datasets = [
    'pubmed',
    'medline',
    'clinical_trials',
    'medical_questions',
    'mimic-iii-clinical-database',
    'i2b2',
    'medchat'
]

data = load_dataset(med_datasets[0], split='train', trust_remote_code=True, streaming=True)
data.save_to_disk('tutorAI/data/text_data')

