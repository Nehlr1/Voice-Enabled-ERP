import gradio as gr
import pandas as pd
import speech_recognition as sr
from dataclasses import dataclass
from transformers import pipeline

@dataclass
class MoneyRequest:
    """
    A data class that represents a money request with three fields:
    - project_id: Identifier for the project
    - amount: The amount of money requested
    - reason: The reason for the money request
    """
    project_id = None
    amount = None
    reason = None

class ERPAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Initializing the transformer models
        self.ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        self.qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        self.zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    def _extract_project_id(self, text):
        if "project" not in text.lower():
            return None
        
        try:
            # Using NER to identify potential project identifiers
            entities = self.ner(text)
            project_part = text.lower().split("project")[1]

            # If immediately followed by "to buy", there's no project specified
            if "to" in project_part.split()[0].lower() or "buy" in project_part.split()[0].lower():
                return None
            
            # Checking NER results first
            for entity in entities:
                if entity["word"] in project_part:
                    if entity["entity"] in ["ORG", "MISC", "NUM"]:
                        return entity["word"]
            
            # If no NER results, checking for keywords
            project_part = project_part.split()
            for word in project_part:
                if word.isdigit():
                    return word
                elif word not in ["to", "buy", "the", "buy"]:
                    possible_name = text.split("project")[1].split("to buy")[0].strip()
                    if possible_name and "amount" not in possible_name.lower():
                        return possible_name
            return None
        except:
            return None