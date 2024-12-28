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
            project_words = project_part.split()
            for word in project_words:
                if word.isdigit():
                    return word
                elif word not in ["to", "for", "the", "buy"]:
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
            categories = ["purchase", "equipment", "supplies", "services", "maintenance"]
            
            # Getting the part after "buy"
            buy_part = text.lower().split("buy")[1]
            for splitter in ["and", "the amount", "i need"]:
                if splitter in buy_part:
                    buy_part = buy_part.split(splitter)[0]
            
            # Using zero-shot classification to categorize the reason
            result = self.zero_shot(buy_part.strip(), categories)
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

    def confirm_and_save(self, text_input):
        if not text_input:
            return "No request to confirm"
            
        request = self.process_request(text_input)
        
        df = pd.DataFrame({
            'project': [request.project_id],
            'request_amount': [request.amount],
            'reason': [request.reason]
        })
        
        df.to_csv("requests.csv", mode='a', header=False, index=False)
        return "Request saved successfully!"

def create_interface():
    assistant = ERPAssistant()
    last_request = {"text": None, "missing_field": None}
    
    def process(audio, text):
        # nonlocal last_request
        recognized_text, response, show_confirm, missing_field = assistant.process_input(audio, text)
        last_request["text"] = recognized_text
        last_request["missing_field"] = missing_field
        
        if missing_field:
            return (
                recognized_text, 
                f"Please speak the {missing_field}", 
                gr.update(visible=False),
                gr.update(visible=True),
            )
        return (
            recognized_text, 
            response, 
            gr.update(visible=True), 
            gr.update(visible=False)
        )
    
    def handle_additional_input(audio, text):
        # nonlocal last_request
        if audio:  # Handling voice input
            with sr.AudioFile(audio) as source:
                try:
                    audio_text = assistant.recognizer.recognize_whisper(assistant.recognizer.record(source))
                    full_text = last_request["text"]
                    
                    if last_request["missing_field"] == "project":
                        full_text = last_request["text"].replace("project", f"project {audio_text}")
                    elif last_request["missing_field"] == "amount":
                        full_text = f"{last_request['text']} {audio_text} riyals"
                    elif last_request["missing_field"] == "reason":
                        full_text = f"{last_request['text']} to {audio_text}"
                        
                    recognized_text, response, show_confirm, _ = assistant.process_input(None, full_text)
                    return (
                        recognized_text, 
                        response, 
                        gr.update(visible=False),
                        gr.update(visible=True)
                    )
                except:
                    return "", "Could not understand audio. Please try again.", gr.update(visible=True), gr.update(visible=False)
        elif text:  # Handling text input
            full_text = last_request["text"]
            if last_request["missing_field"] == "project":
                full_text = last_request["text"].replace("project", f"project {text}")
            elif last_request["missing_field"] == "amount":
                full_text = f"{last_request['text']} {text} riyals"
            elif last_request["missing_field"] == "reason":
                full_text = f"{last_request['text']} to {text}"
                
            recognized_text, response, show_confirm, _ = assistant.process_input(None, full_text)
            return (
                recognized_text, 
                response, 
                gr.update(visible=False),
                gr.update(visible=True)
            )
        return "", "Invalid input", gr.update(visible=True), gr.update(visible=False)

    with gr.Blocks(theme=gr.themes.Base()) as interface:
        gr.Markdown("## Voice-Enabled ERP Assistant")
        gr.Markdown("Speak or type your request to create a money request for your project.")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources="microphone", type="filepath", label="Voice Input")
                text_input = gr.Textbox(label="Text Input (Alternative)")
                
                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    submit_btn = gr.Button("Submit", variant="primary")
              
            with gr.Column():
                recognized_text = gr.Textbox(label="Recognized Text", interactive=False)
                response_text = gr.Textbox(label="Assistant Response", interactive=False)
                
                # Grouping the additional input elements
                with gr.Group(visible=False) as additional_input_group:
                    additional_audio_input = gr.Audio(sources="microphone", type="filepath", label="Record Additional Input")
                    additional_text_input = gr.Textbox(label="Additional Text Input")
                    additional_submit_btn = gr.Button("Submit Additional Input")
                
                # Grouping for confirmation elements
                with gr.Group(visible=False) as confirmation_group:
                    confirm_btn = gr.Button("OK")
                    result = gr.Textbox(label="Status", interactive=False)

        # Event handlers
        submit_btn.click(
            process, 
            inputs=[audio_input, text_input], 
            outputs=[recognized_text, response_text, confirmation_group, additional_input_group]
        )
        
        # Handling additional input
        additional_submit_btn.click(
            handle_additional_input,
            inputs=[additional_audio_input, additional_text_input],
            outputs=[recognized_text, response_text, additional_input_group, confirmation_group]
        )
        
        confirm_btn.click(
            assistant.confirm_and_save,
            inputs=[recognized_text],
            outputs=[result]
        )
        
        clear_btn.click(
            lambda: (None, None, "", "", gr.update(visible=False), gr.update(visible=False)),
            outputs=[audio_input, text_input, recognized_text, response_text, confirmation_group, additional_input_group]
        )

        # Examples
        gr.Examples(
            examples=[
                [None, "I need to request money for project 223 to buy some tools the amount I need is 500 riyals"],
                [None, "I need to request money for project to buy some tools the amount I need is 50 riyals"]
            ],
            inputs=[audio_input, text_input]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    try:
        interface.launch(show_error=True)
    except Exception as e:
        print(f"Error launching interface: {str(e)}")