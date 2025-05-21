import re
import json
import argparse
import spacy
from owlready2 import get_ontology, onto_path
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from tqdm import tqdm

# Pronouns and short tokens to filter out
PRONOUNS = {"i","you","he","she","it","we","they","this","that","these","those"}
MIN_TOKEN_LEN = 3  # minimum length for subject/object tokens


def load_ontology(owl_path: str):
    """
    Load the COPE ontology from the given OWL file path.
    """
    onto_path.append("data/ontology")
    return get_ontology(owl_path).load()


def map_to_ontology(onto, label: str):
    """
    Try to match `label` to an ontology class by its rdfs:label or name.
    Returns the class name or None.
    """
    key = label.lower().strip()
    for cls in onto.classes():
        for lbl in cls.label:
            if lbl.lower() == key:
                return cls.name
    norm = re.sub(r'\W+', '', key)
    for cls in onto.classes():
        if cls.name.lower() == norm:
            return cls.name
    return None


def valid_triple(subj: str, obj: str) -> bool:
    """
    Filter out pronouns, too-short, or malformed triples.
    """
    if subj.lower() in PRONOUNS or obj.lower() in PRONOUNS:
        return False
    if len(subj) < MIN_TOKEN_LEN or len(obj) < MIN_TOKEN_LEN:
        return False
    return True


def extract_svo_triples(sent) -> list:
    """
    Extract all (subject, verb, object) triples using noun chunks.
    Returns a list of valid tuples.
    """
    triples = []
    verbs = [token for token in sent if token.dep_ == "ROOT"]
    if not verbs:
        return triples
    verb = verbs[0].lemma_
    subs = [chunk.text.strip() for chunk in sent.doc.noun_chunks if any(tok.dep_ == "nsubj" for tok in chunk)]
    objs = [chunk.text.strip() for chunk in sent.doc.noun_chunks if any(tok.dep_ in ("dobj","pobj") for tok in chunk)]
    for subj in subs:
        for obj in objs:
            if valid_triple(subj, obj):
                triples.append((subj, verb, obj))
    return triples


def generate(segments_path: str, output_qas: str, output_triples: str,
             ontology_path: str, use_qg: bool):
    # Load NLP model
    nlp = spacy.load("en_core_web_sm")
    onto = load_ontology(ontology_path)

    # Initialize QG pipeline if requested
    if use_qg:
        tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-prepend", use_fast=False)
        model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-prepend")
        qg = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            framework="pt",
            device=-1,
            max_length=64,
            num_beams=3,
            num_return_sequences=1
        )

    # Read all segments to get total count for progress bar
    with open(segments_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    triples = []
    qa_pairs = []

    # Iterate with tqdm using total for percentage
    for line in tqdm(lines, desc="Processing segments", unit="seg", total=len(lines)):
        text = line.strip()
        if not text:
            continue
        doc = nlp(text)
        for sent in doc.sents:
            qa_subj_cls, qa_obj_cls = "Unknown", "Unknown"
            svos = extract_svo_triples(sent)
            for subj, verb, obj in svos:
                subj_cls = map_to_ontology(onto, subj)
                obj_cls = map_to_ontology(onto, obj)
                if subj_cls is None and obj_cls is None:
                    continue
                if qa_subj_cls == "Unknown":
                    qa_subj_cls = subj_cls or "Unknown"
                    qa_obj_cls = obj_cls or "Unknown"
                triples.append({
                    "subject_text": subj,
                    "relation": verb,
                    "object_text": obj,
                    "subject_cls": subj_cls or "Unknown",
                    "object_cls": obj_cls or "Unknown"
                })
            # Generate QA pairs
            if use_qg:
                context = sent.text.strip()
                prompt = f"context: {context} </s> generate question:"
                outputs = qg([prompt])
                for out in outputs:
                    gen = out['generated_text'].strip()
                    if '?' in gen:
                        q, a = gen.split('?', 1)
                        question = q.strip() + '?'
                        answer = a.strip()
                    else:
                        question, answer = gen, ''
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "subject_cls": qa_subj_cls,
                        "object_cls": qa_obj_cls
                    })
            else:
                for subj, verb, obj in svos:
                    if verb.lower() in ("be", "is", "are"):
                        question = f"What is {subj}?"
                        answer = obj
                    else:
                        question = f"What does {subj} {verb}?"
                        answer = obj
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "subject_cls": qa_subj_cls,
                        "object_cls": qa_obj_cls
                    })

    # Save outputs
    with open(output_triples, 'w', encoding='utf-8') as f:
        json.dump(triples, f, indent=2)
    with open(output_qas, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate QA data with optional QG and ontology mapping"
    )
    parser.add_argument("--segments",       default="data/processed/iom_segments.txt")
    parser.add_argument("--output_qas",     default="data/processed/qa_pairs.json")
    parser.add_argument("--output_triples", default="data/processed/triples.json")
    parser.add_argument("--ontology",       default="data/ontology/COPE_pharma.owl")
    parser.add_argument("--use_qg",         action="store_true",
                        help="Enable question-generation (slower)")
    args = parser.parse_args()
    generate(
        args.segments,
        args.output_qas,
        args.output_triples,
        args.ontology,
        use_qg=args.use_qg
    )