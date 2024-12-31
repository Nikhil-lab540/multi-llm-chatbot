import streamlit as st
from mem0 import Memory
import os
from litellm import completion

st.title("Multi-LLM App with Shared Memory 🧠")
st.caption("LLM App with a personalized memory layer that remembers each user's choices and interests across multiple users and LLMs")

# Input API keys for Groq and Gemini
st.sidebar.header("API Keys")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if groq_api_key and gemini_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["GEMINI_API_KEY"] = gemini_api_key

    # Initialize Mem0 with Qdrant
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": "localhost",
                "port": 6333,
            }
        },
    }

    memory = Memory.from_config(config)

    user_id = st.sidebar.text_input("Enter your Username")
    llm_choice = st.sidebar.radio("Select LLM", ('Groq LLM', 'Gemini LLM'))

    if llm_choice == 'Groq LLM':
        # Configure Groq client (replace with actual Groq API integration code)
        groq_config = {
            "llm": {
                "provider": "groq",
                "config": {
                    "api_key": groq_api_key,
                    "temperature": 0.7,
                    "max_tokens": 1500,
                }
            }
        }
        client = Memory.from_config(groq_config)

    elif llm_choice == 'Gemini LLM':
        # Configure Gemini client (replace with actual Gemini API integration code)
        gemini_config = {
            "llm": {
                "provider": "gemini",
                "config": {
                    "api_key": gemini_api_key,
                    "temperature": 0.6,
                    "max_tokens": 2000,
                }
            }
        }
        client = Memory.from_config(gemini_config)

    prompt = st.text_input("Ask the LLM")

    if st.button('Chat with LLM'):
        with st.spinner('Searching...'):
            relevant_memories = memory.search(query=prompt, user_id=user_id)
            context = "Relevant past information:\n"
            if relevant_memories and "results" in relevant_memories:
                for memory in relevant_memories["results"]:
                    if "memory" in memory:
                        context += f"- {memory['memory']}\n"

            full_prompt = f"{context}\nHuman: {prompt}\nAI:"

            if llm_choice == 'Groq LLM':
                # Replace with actual Groq API call
                response = completion(model="groq-model", prompt=full_prompt)
                answer = response['choices'][0]['text']

            elif llm_choice == 'Gemini LLM':
                # Replace with actual Gemini API call
                response = completion(model="gemini-model", prompt=full_prompt)
                answer = response['choices'][0]['text']

            st.write("Answer: ", answer)

            memory.add(answer, user_id=user_id)

    # Sidebar option to show memory
    st.sidebar.title("Memory Info")
    if st.button("View My Memory"):
        memories = memory.get_all(user_id=user_id)
        if memories and "results" in memories:
            st.write(f"Memory history for **{user_id}**:")
            for mem in memories["results"]:
                if "memory" in mem:
                    st.write(f"- {mem['memory']}")
        else:
            st.sidebar.info("No learning history found for this user ID.")
