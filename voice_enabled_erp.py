import gradio as gr
import pandas as pd
from dataclasses import dataclass
from transformers import pipeline

@dataclass
class MoneyRequest:
    project_id = None
    amount = None
    reason = None

class ERPAssistant:
    def __init__(self):
        pass