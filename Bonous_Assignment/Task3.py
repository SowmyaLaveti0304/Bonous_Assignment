# 1. Install dependencies (if you haven’t already):
#    pip install transformers torch

from transformers import pipeline

# 2. Initialize the QA pipeline with the deepset/roberta-base-squad2 model
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

# 3. Your custom 2–3 sentence context about Abdul Kalam
context = """
Avul Pakir Jainulabdeen Abdul Kalam (15 October 1931 – 27 July 2015) was an Indian aerospace scientist 
and statesman who served as the 11th President of India from 2002 to 2007. He played a leading role 
in India’s civilian space program and military missile development, earning him the nickname 
“Missile Man of India.”
"""

# 4. Two different questions
questions = [
    "What nickname did he earn?",
    "During which years did he serve as President of India?"
]

# 5. Run both queries and print the results
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print({
        "question": question,
        "answer":   result["answer"],
        "score":    round(result["score"], 2),
        "start":    result["start"],
        "end":      result["end"]
    })
