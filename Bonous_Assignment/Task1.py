# 1. Install dependencies (if you haven’t already):
#    pip install transformers torch

# 2. Import and initialize the QA pipeline
from transformers import pipeline

qa_pipeline = pipeline("question-answering")

# 3. Define your context and question
context = """
Charles Babbage (26 December 1791 – 18 October 1871) was an English polymath. 
He is best known today as the “father of the computer” for originating the concept 
of a programmable computer. Babbage is credited with inventing the first mechanical 
computer, the Analytical Engine, which had most of the elements of the modern computer.
"""

question = "Who is known as the father of the computer?"

# 4. Run the pipeline
result = qa_pipeline(question=question, context=context)

# 5. Inspect the output
print(result)
