import gradio as gr
import google.generativeai as palm
from react_agent import ReActAgent  # Make sure this is the updated version

# Configure PaLM
palm.configure(api_key='AIzaSyCFPXxgS8Stxq975EZSM9Rn71Q5naJXHqs')

agent = ReActAgent()

current_task = None
actions_taken = []
task_complete = False
final_action = None

def start_task():
    global current_task, actions_taken, task_complete, final_action
    current_task = agent.get_random_task()
    actions_taken = []
    task_complete = False
    final_action = None
    return f"Current Task: {current_task['description']}", ""

def process_step(feedback, modified_action):
    global current_task, actions_taken, task_complete, final_action
    
    if task_complete:
        return "Task is already complete. Start a new task.", ""
    
    thought = agent.reason(current_task)
    action = agent.act(thought, current_task)
    
    result = f"Thought: {thought}\nProposed Action: {action}\n\n"
    
    if feedback == "Approve":
        actions_taken.append(action)
        result += f"Executing: {action}"
    elif feedback == "Modify":
        if modified_action:
            actions_taken.append(modified_action)
            result += f"Executing modified action: {modified_action}"
        else:
            result += "Please provide a modified action."
    else:  # Reject
        result += "Action rejected. Rethinking..."
    
    if len(actions_taken) >= len(current_task['possible_actions']):
        task_complete = True
        evaluation = agent.evaluate_outcome(current_task, actions_taken)
        final_action = agent.final_task(current_task, actions_taken, evaluation)
        result += f"\n\nTask Completed!\nEvaluation: {evaluation}\nFinal Task: {final_action}"
    
    result += "\n\nActions Taken:\n" + "\n".join(f"{i+1}. {action}" for i, action in enumerate(actions_taken))
    
    return result, ""

def get_history():
    return "Thought History:\n" + "\n".join(f"{i+1}. {thought}" for i, thought in enumerate(agent.thought_history))

with gr.Blocks() as demo:
    gr.Markdown("# Human-in-the-loop ReAct Agent")
    
    with gr.Row():
        start_btn = gr.Button("Start New Task")
        task_output = gr.Textbox(label="Current Task")
    
    with gr.Row():
        feedback = gr.Radio(["Approve", "Modify", "Reject"], label="Your input")
        modified_action = gr.Textbox(label="Modified Action (if applicable)")
    
    submit_btn = gr.Button("Submit Feedback")
    
    result_output = gr.Textbox(label="Result")
    history_btn = gr.Button("Show Thought History")
    history_output = gr.Textbox(label="Thought History")
    
    start_btn.click(start_task, outputs=[task_output, modified_action])
    submit_btn.click(process_step, inputs=[feedback, modified_action], outputs=[result_output, modified_action])
    history_btn.click(get_history, outputs=history_output)

demo.launch()