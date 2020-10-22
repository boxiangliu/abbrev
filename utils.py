def extract_examples(ab3p):
    contexts = []
    questions = []
    answers = []

    for i, row in ab3p.iterrows():
        lf = format_answer(row["lf"])
        sentence = row["sent"]
        if lf in sentence:
            contexts.append(sentence)
            questions.append("What does %s stand for?" % row["sf"])
            answers.append({"text": lf})
    return contexts, questions, answers