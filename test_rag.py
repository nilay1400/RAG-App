from src.rag import generate_answer

question = "Explain the benefit of transformers."
answer = generate_answer(question)
print("\nAnswer:\n", answer)
