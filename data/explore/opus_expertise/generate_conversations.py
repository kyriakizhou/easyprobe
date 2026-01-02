#!/usr/bin/env python3
"""
Script to generate synthetic conversation data for expertise levels.
Generates 250 conversations each for novice, medium, and expert levels.
"""

import anthropic
import os
import sys
import time
import random

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

client = anthropic.Anthropic()

# Topics for variety
TOPICS = [
    "programming", "machine learning", "web development", "databases", "cybersecurity",
    "cloud computing", "data science", "algorithms", "networking", "operating systems",
    "software architecture", "devops", "mobile development", "game development", "blockchain",
    "natural language processing", "computer vision", "robotics", "quantum computing", "cryptography",
    "API design", "microservices", "containerization", "version control", "testing",
    "debugging", "performance optimization", "system design", "distributed systems", "compilers",
    "mathematics", "statistics", "physics", "chemistry", "biology",
    "finance", "economics", "cooking", "photography", "music production",
    "writing", "graphic design", "video editing", "3D modeling", "animation",
    "fitness", "nutrition", "mental health", "meditation", "productivity",
    "language learning", "history", "philosophy", "psychology", "sociology",
    "gardening", "woodworking", "electronics", "automotive repair", "home improvement"
]

PROMPTS = {
    "novice": [
        """Generate a conversation between a human user and an AI assistant. This human user is a novice (beginner) in the topic being discussed. Make sure the conversation reflects this user's expertise level as a complete beginner. The user should:
- Ask basic, foundational questions
- Use simple language and may not know technical terminology
- Need explanations of fundamental concepts
- Express uncertainty or confusion about basic ideas
- May ask "what is" or "how do I start" type questions

Be creative on the topics of conversation. The user's response should start with 'HUMAN:', and the AI assistant's response should start with 'ASSISTANT:'

Generate a natural conversation of 3-6 exchanges about {topic}.""",

        """Generate a conversation between a human user and an AI assistant. This human user has novice-level expertise. Be creative on the topics of conversation. Make sure the conversation reflects this user's beginner expertise level. This may be reflected by:
- Their lack of familiarity with jargon
- Asking for step-by-step guidance
- Needing clarification on basic concepts
- Showing enthusiasm but limited background knowledge

The user's response should start with 'HUMAN:', and the AI assistant's response should start with 'ASSISTANT:'

Generate a natural conversation of 3-6 exchanges about {topic}."""
    ],

    "medium": [
        """Generate a conversation between a human user and an AI assistant. This human user has intermediate/medium expertise in the topic being discussed. Make sure the conversation reflects this user's medium expertise level. The user should:
- Understand basic concepts but seek deeper knowledge
- Use some technical terminology correctly
- Ask about best practices or optimization
- Want to understand "why" not just "how"
- May have tried something and encountered specific issues

Be creative on the topics of conversation. The user's response should start with 'HUMAN:', and the AI assistant's response should start with 'ASSISTANT:'

Generate a natural conversation of 3-6 exchanges about {topic}.""",

        """Generate a conversation between a human user and an AI assistant. This human user has medium-level expertise. Be creative on the topics of conversation. Make sure the conversation reflects this user's intermediate expertise level. This may be reflected by:
- Their comfortable use of domain-specific language
- Asking about trade-offs between different approaches
- Discussing specific scenarios or edge cases
- Building on existing knowledge

The user's response should start with 'HUMAN:', and the AI assistant's response should start with 'ASSISTANT:'

Generate a natural conversation of 3-6 exchanges about {topic}."""
    ],

    "expert": [
        """Generate a conversation between a human user and an AI assistant. This human user is an expert in the topic being discussed. Make sure the conversation reflects this user's expert expertise level. The user should:
- Use technical terminology fluently and correctly
- Ask about advanced, nuanced topics
- Discuss edge cases, optimizations, or cutting-edge developments
- Challenge or discuss complex trade-offs
- May share their own insights while seeking additional perspectives

Be creative on the topics of conversation. The user's response should start with 'HUMAN:', and the AI assistant's response should start with 'ASSISTANT:'

Generate a natural conversation of 3-6 exchanges about {topic}.""",

        """Generate a conversation between a human user and an AI assistant. This human user has expert-level expertise. Be creative on the topics of conversation. Make sure the conversation reflects this user's advanced expertise level. This may be reflected by:
- Deep technical discussions with precise terminology
- Exploring niche or specialized aspects of the topic
- Questioning assumptions or discussing limitations
- Engaging as a peer rather than a student

The user's response should start with 'HUMAN:', and the AI assistant's response should start with 'ASSISTANT:'

Generate a natural conversation of 3-6 exchanges about {topic}."""
    ]
}

def generate_conversation(expertise_level: str, topic: str, prompt_variant: int) -> str:
    """Generate a single conversation."""
    prompt_template = PROMPTS[expertise_level][prompt_variant]
    prompt = prompt_template.format(topic=topic)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text

def get_existing_files(output_dir: str) -> set:
    """Get set of existing conversation numbers for each level."""
    existing = {"novice": set(), "medium": set(), "expert": set()}
    for filename in os.listdir(output_dir):
        if filename.startswith("conversation_") and filename.endswith(".txt"):
            parts = filename.replace(".txt", "").split("_")
            if len(parts) >= 3:
                try:
                    num = int(parts[1])
                    level = parts[-1]
                    if level in existing:
                        existing[level].add(num)
                except ValueError:
                    pass
    return existing

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    expertise_levels = ["novice", "medium", "expert"]
    conversations_per_level = 250

    # Check existing files
    existing = get_existing_files(output_dir)

    for level in expertise_levels:
        existing_count = len(existing[level])
        print(f"\n{'='*50}", flush=True)
        print(f"Level: {level}", flush=True)
        print(f"Existing: {existing_count} conversations", flush=True)
        print(f"Need to generate: {conversations_per_level - existing_count} more", flush=True)
        print(f"{'='*50}", flush=True)

        for i in range(conversations_per_level):
            if i in existing[level]:
                continue

            topic = random.choice(TOPICS)
            prompt_variant = 0 if random.random() < 0.72 else 1

            try:
                conversation = generate_conversation(level, topic, prompt_variant)

                filename = f"conversation_{i}_expertise_{level}.txt"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'w') as f:
                    f.write(conversation)

                count_so_far = len([f for f in os.listdir(output_dir) if f.endswith(f"expertise_{level}.txt")])
                print(f"[{count_so_far}/{conversations_per_level}] Generated: {filename} (topic: {topic})", flush=True)

            except Exception as e:
                print(f"Error generating conversation {i} for {level}: {e}", flush=True)
                time.sleep(2)
                continue

    print(f"\n{'='*50}", flush=True)
    print("Generation complete!", flush=True)

    # Final count
    for level in expertise_levels:
        count = len([f for f in os.listdir(output_dir) if f.endswith(f"expertise_{level}.txt")])
        print(f"{level}: {count} conversations", flush=True)
    print(f"{'='*50}", flush=True)

if __name__ == "__main__":
    main()
