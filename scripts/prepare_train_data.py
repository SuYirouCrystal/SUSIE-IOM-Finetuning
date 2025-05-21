import json

qa_pairs = json.load(open('data/processed/qa_pairs.json', 'r', encoding='utf-8'))

with open('data/processed/train.jsonl', 'w', encoding='utf-8') as fout:
    for qa in qa_pairs:
        question = qa["question"]
        answer   = qa["answer"]
        source_text = f"question: {question}"
        target_text = answer
        record = {"source": source_text, "target": target_text}
        fout.write(json.dumps(record) + "\n")