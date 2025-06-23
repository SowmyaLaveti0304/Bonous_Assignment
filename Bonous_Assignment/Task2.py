from transformers import pipeline

# Initialize with the custom pretrained model
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

# Same Babbage context as before
context = """
Charles Babbage (26 December 1791 – 18 October 1871) was an English polymath. 
He is best known today as the “father of the computer” for originating the concept 
of a programmable computer. Babbage is credited with inventing the first mechanical 
computer, the Analytical Engine, which had most of the elements of the modern computer.
"""

question = "Who is known as the father of the computer?"

# Run with the custom model
result = qa_pipeline(question=question, context=context)

print(result)
