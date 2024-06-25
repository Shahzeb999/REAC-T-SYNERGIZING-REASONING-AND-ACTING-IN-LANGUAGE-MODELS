import gradio as gr
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Load the model and tokenizer for AG News classification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

dataset = load_dataset("ag_news", split="test[:10%]")  # Using 10% of the dataset for evaluation

# Label mapping for AG News dataset
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Define the ReAct function
def react(news_article, human_input_needed=False, human_input=None):
    # Step 1: Reasoning (Chain-of-Thought)
    reasoning_trace = f"Received article: '{news_article}'"
    
    # Step 2: Acting (Classification)
    candidate_labels = list(label_map.values())
    classification = classifier(news_article, candidate_labels)
    label = classification['labels'][0]
    score = classification['scores'][0]
    action_trace = f"Classified as '{label}' with confidence score of {score:.2f}"
    
    if human_input_needed and score < 0.8:
        action_trace += " [Human input required]"
        if human_input:
            label = human_input
            action_trace += f" | Human updated label to '{label}'"
    
    return reasoning_trace, action_trace, label, score

# Gradio interface function
def classify_news(news_article, human_input=None):
    reasoning, action, label, score = react(news_article, human_input_needed=True, human_input=human_input)
    response = f"Reasoning Trace: {reasoning}\nAction Trace: {action}\n\nFinal Label: {label}\nConfidence Score: {score:.2f}"
    return response

# Evaluation function
def evaluate_model(dataset, human_input_needed=False):
    predictions = []
    labels = []
    for item in dataset:
        news_article = item['text']
        true_label = label_map[item['label']]
        _, _, predicted_label, _ = react(news_article, human_input_needed=human_input_needed, human_input=true_label if human_input_needed and random.random() < 0.5 else None)
        predictions.append(predicted_label)
        labels.append(true_label)
    
    accuracy = accuracy_score(labels, predictions)
    return accuracy

# Perform evaluation
base_accuracy = evaluate_model(dataset, human_input_needed=False)
human_in_the_loop_accuracy = evaluate_model(dataset, human_input_needed=True)
print(f"Base Model Accuracy: {base_accuracy:.2f}")
print(f"Human-in-the-Loop Model Accuracy: {human_in_the_loop_accuracy:.2f}")

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# News Article Classifier with Human-in-the-Loop using ReAct")
    news_input = gr.Textbox(lines=5, label="Enter a news article:")
    human_input = gr.Textbox(lines=1, label="Provide your input if needed:", placeholder="e.g., World, Sports, Business, Sci/Tech")
    output = gr.Textbox(lines=10, label="Output")
    
    btn = gr.Button("Classify")
    btn.click(classify_news, inputs=[news_input, human_input], outputs=output)

# Launch the interface
demo.launch()