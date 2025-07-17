from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime


# DirectPromptAgent class definition
class DirectPromptAgent:
    """
    A Direct Prompt Agent that sends user input directly to the LLM without modification.
    Uses only the LLM's general knowledge to respond.
    """

    def __init__(self, openai_api_key):
        """Initialize the agent with OpenAI API key."""
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        """Generate a response using the OpenAI API directly."""
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content

        

# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    """
    An Augmented Prompt Agent that responds according to a predefined persona.
    Uses a system prompt to define behavior and context.
    """
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using OpenAI API with persona context."""
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )

        # Declare a variable 'response' that calls OpenAI's API for a chat completion.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": f"You are {self.persona}. Forget all previous context."
                },
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        return response.choices[0].message.content


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    """
    A Knowledge Augmented Prompt Agent that incorporates specific provided knowledge
    alongside a defined persona when responding to prompts.
    """
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using the OpenAI API with specific knowledge."""
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )

        system_message = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge} "
            f"Answer the prompt based on this knowledge, not your own."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
               {"role": "system", "content": system_message},
               {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.
    
        Parameters:
        text (str): Text to split into chunks.
    
        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()
    
        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]
    
        chunks, start, chunk_id = [], 0, 0
    
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)
    
            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })
    
            start = end - self.chunk_overlap
            chunk_id += 1
    
        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})
    
        return chunks


    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content


class EvaluationAgent:
    """
    An Evaluation Agent that assesses responses from a worker agent against given criteria
    and provides iterative feedback for improvement.
    """
    
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        """Initialize the EvaluationAgent with given attributes."""
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = worker_agent
        self.max_interactions = max_interactions


    def evaluate(self, initial_prompt):
        """Manages interactions between agents to achieve a solution."""
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )
        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            # Step 1: Worker agent generates a response
            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.agent_to_evaluate.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            # Step 2: Evaluator agent judges the response
            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria} "
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            # Step 3: Check if evaluation is positive
            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                # Step 4: Generate instructions to correct the response
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.persona},
                        {"role": "user", "content": instruction_prompt}
                    ],
                    temperature=0
                )

                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                # Step 5: Send feedback to worker agent for refinement
                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            "final_response": response_from_worker,
            "final_evaluation": evaluation,
            "iterations": i + 1
        }   


class RoutingAgent:
    """
    A Routing Agent that directs user prompts to the most appropriate specialized agent
    based on semantic similarity between prompts and agent descriptions.
    """
    def __init__(self, openai_api_key, agents):

        """Initialize the agent with given attributes."""
        self.openai_api_key = openai_api_key
        self.agents = agents if agents is not None else []

    def get_embedding(self, text):
        """Calculate text embeddings using OpenAI's embedding model."""
        # client = OpenAI(api_key=self.openai_api_key)
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )

        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )

        # Extract and return the embedding vector from the response
        embedding = response.data[0].embedding
        return embedding 

    # Define a method to route user prompts to the appropriate agent
    def route(self, user_input):
        """Route user prompts to the most appropriate agent."""
        # Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
           # Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            # Calculate cosine similarity
            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(f"Similarity with {agent['name']}: {similarity:.3f}")

            # Select the best agent based on similarity score
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:
    """
    An Action Planning Agent that extracts and lists steps required to execute
    a task described in a user's prompt using provided knowledge.
    """

    def __init__(self, openai_api_key, knowledge):

        """Initialize the agent attributes."""
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        """Extract actionable steps from a user prompt."""
         # Instantiate the OpenAI client
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
        )

        # Call the OpenAI API with system and user prompts
        system_prompt = (
            f"You are an action planning agent. Using your knowledge, you extract from the user prompt "
            f"the steps requested to complete the action the user is asking for. You return the steps as a list. "
            f"Only return the steps in your knowledge. Forget any previous context. "
            f"This is your knowledge: {self.knowledge}"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        # Extract the response text        
        response_text = response.choices[0].message.content

        # Clean and format the extracted steps
        steps = [step.strip() for step in response_text.split("\n") if step.strip()]
        # Remove numbering if present and clean up
        cleaned_steps = []
        for step in steps:
            # Remove common numbering patterns
            cleaned_step = re.sub(r'^\d+\.\s*', '', step)
            cleaned_step = re.sub(r'^-\s*', '', cleaned_step)
            cleaned_step = re.sub(r'^\*\s*', '', cleaned_step)
            if cleaned_step:
                cleaned_steps.append(cleaned_step)

        return cleaned_steps
