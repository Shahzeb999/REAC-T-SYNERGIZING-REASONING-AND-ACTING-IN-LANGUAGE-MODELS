import json
import random
import google.generativeai as palm

class ReActAgent:
    def __init__(self):
        self.thought_history = []
        self.load_tasks()

    def load_tasks(self):
        with open('task_dataset.json', 'r') as f:
            self.tasks = json.load(f)['tasks']

    def get_random_task(self):
        return random.choice(self.tasks)

    def generate_text(self, prompt):
        response = palm.generate_text(prompt=prompt)
        return response.result

    def reason(self, task):
        prompt = f"Task: {task['description']}\nPossible actions: {', '.join(task['possible_actions'])}\nThought: Let's approach this step-by-step:"
        thought = self.generate_text(prompt)
        self.thought_history.append(thought)
        return thought

    def act(self, thought, task):
        prompt = f"Based on the thought: {thought}\nPossible actions: {', '.join(task['possible_actions'])}\nAction: The next action should be:"
        action = self.generate_text(prompt)
        return action

    def human_input(self, action, task):
        print(f"Task: {task['description']}")
        print(f"Proposed action: {action}")
        human_feedback = input("Your input (approve/modify/reject): ").lower()
        if human_feedback == 'approve':
            return action
        elif human_feedback == 'modify':
            return input("Please provide the modified action: ")
        else:
            return None

    def evaluate_outcome(self, task, actions_taken):
        prompt = f"Task: {task['description']}\nActions taken: {', '.join(actions_taken)}\nExpected outcome: {task['expected_outcome']}\nEvaluation: How well do the actions align with the expected outcome?"
        return self.generate_text(prompt)

    def final_task(self, task, actions_taken, evaluation):
        prompt = f"""
        Task: {task['description']}
        Actions taken: {', '.join(actions_taken)}
        Evaluation: {evaluation}
        
        Based on the completed task and its evaluation, what final action or recommendation should be made?
        Final task:
        """
        return self.generate_text(prompt)

    def execute_task(self):
        task = self.get_random_task()
        actions_taken = []
        while True:
            thought = self.reason(task)
            action = self.act(thought, task)
            approved_action = self.human_input(action, task)
            if approved_action:
                print(f"Executing: {approved_action}")
                actions_taken.append(approved_action)
                if len(actions_taken) >= len(task['possible_actions']):
                    break
            else:
                print("Action rejected. Rethinking...")
        
        evaluation = self.evaluate_outcome(task, actions_taken)
        print(f"Task completed. Evaluation: {evaluation}")
        
        final_action = self.final_task(task, actions_taken, evaluation)
        print(f"Final task: {final_action}")
        return final_action

# Usage
palm.configure(api_key='AIzaSyCFPXxgS8Stxq975EZSM9Rn71Q5naJXHqs')
agent = ReActAgent()
# Commenting out the automatic execution
# agent.execute_task()