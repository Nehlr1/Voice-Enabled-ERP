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
    
    def _extract_amount(self, text):
        try:
            # Using QA model to find amount
            question = "How many riyals are requested?"
            result = self.qa_pipeline(question=question, context=text)

            # Trying to extract number from QA result
            answer = result["answer"]
            numbers = "".join(filter(str.isdigit, answer))
            if numbers:
                return float(numbers)

            # If no QA results, checking for keywords
            words = text.lower().split()
            for i, word in enumerate(words):
                if word.isdigit():
                    if i + 1 < len(words) and "riyal" in words[i + 1].lower():
                        # Making sure this number isn't a project number
                        project_part = text.split("project")[1].split() if "project" in text else []
                        if not (project_part and word in project_part[:2]):
                            return float(word)
            return None
        except:
            return None

    def _extract_reason(self, text):
        try:
            if "buy" not in text.lower():
                return None

            # Defining reason categories
            catergories = ["purchase", "equipment", "supplies", "services", "maintenance"]

            # Getting the part after "buy"
            buy_part = text.lower().split("buy")[1]
            for splitter in ["and", "the amount", "i need"]:
                if splitter in buy_part:
                    buy_part = buy_part.split(splitter)[0]
            
            # Using zero-shot classification to categorize the reason
            result = self.zero_shot(buy_part.strip(), catergories)
            top_category = result["labels"][0]

            return f"{top_category}: buy {buy_part.strip()}"
        except:
            # If no zero-shot results, checking for keywords
            if "buy" in text.lower():
                buy_part = text.lower().split("buy")[1]
                for splitter in ["and", "the amount", "i need"]:
                    if splitter in buy_part:
                        buy_part = buy_part.split(splitter)[0]
                return f"buy {buy_part.strip()} for the project"
            return None

    def _get_confidence(self, field, value):
        # Confidence scoring
        if value is None:
            return 0.0
        if field == "amount" and isinstance(value, (int, float)):
            return 1.0
        if field == "project_id" and (value.isdigit() or len(value) > 2):
            return 0.9
        if field == "reason" and len(value) > 10:
            return 0.8
        return 0.5

    def process_input(self, audio_path=None, text_input=None):
        if audio_path:
            try:
                with sr.AudioFile(audio_path) as source:
                    audio = self.recognizer.record(source)
                    text_input = self.recognizer.recognize_whisper(audio)
            except Exception as e:
                print(f"Audio processing error: {str(e)}")
                return "Could not understand audio", "", False, None
        
        if not text_input:
            return "No input provided", "", False, None
        
        # Processing the request
        request = self.process_request(text_input)

        # Checking for missing fields one by one
        if not request.project_id:
            return (
                text_input,
                "Can you provide me the project name you are trying to request money for?",
                False,
                "project"
            )
        elif not request.amount:
            return (
                text_input,
                "Can you specify the amount you need in riyals?",
                False,
                "amount"
            )
        elif not request.reason:
            return (
                text_input,
                "Can you provide the reason for this money request?",
                False,
                "reason"
            )
        
        # Generating confirmation message if all fields are present
        confirmation = self.generate_confirmation_message(request)
        return text_input, confirmation, True, None

    def process_request(self, text):
        request = MoneyRequest()

        # Extracting all fields
        results = {
            "project_id": self._extract_project_id(text),
            "amount": self._extract_amount(text),
            "reason": self._extract_reason(text)
        }

        # Calculating confidence scores
        confidences = {
            field: self._get_confidence(field, value)
            for field, value in results.items()
        }

        # Using results based on confidence threshold
        threshold = 0.5
        request.project_id = results["project_id"] if confidences["project_id"] > threshold else None
        request.amount = results["amount"] if confidences["amount"] > threshold else None
        request.reason = results["reason"] if confidences["reason"] > threshold else None

        return request

    def validate_request(self, request):
        return {
            "project": bool(request.project_id),
            "amount": bool(request.amount),
            "reason": bool(request.reason)
        }
    
    def generate_confirmation_message(self, request):
        return f"""You are going to add request money for
project: {request.project_id}
request amount: {request.amount}
reason: {request.reason}
Are you sure you want to proceed?"""

    def confirm_request(self, text_input):
        if not text_input:
            return "No request to confirm"
        
        request = self.process_request(text_input)

        df = pd.DataFrame({
            "project": [request.project_id],
            "request_amount": [request.amount],
            "reason": [request.reason]
        })

        df.to_csv("requests.csv", mode="a", header=False, index=False)
        return "Request saved successfully"