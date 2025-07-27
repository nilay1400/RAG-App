from rag_chain import answer_question

q = "What is the main contribution of the DeepVigor paper?"
answer, chunks = answer_question(q)

print("Answer:\n", answer)
