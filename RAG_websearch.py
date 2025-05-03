import requests
import os
import urllib
import json
from duckduckgo_search import DDGS
from llama_index.llms.gemini import Gemini

GOOGLE_API_KEY = "AIzaSyB48j08Xi5rLdjix8NwV6CSe8ae6m0Vp58"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


llm = Gemini(
    model="models/gemini-2.0-flash",
)
template_router = "You’re a LLM that detects intent from user queries."\
                   "Your special domain is an insurance assistant"\
                   "Your task is to classify the user's intent based on their query."\
                   "Below are the possible intents with brief descriptions."\
                   "Use these to accurately determine the user's goal, and output only the intent topic."\
                  "- Insurance contract guiding:Questions regarding creating insurance contract"\
                  "- Product Information: Questions regarding insurance product details, specifications, availability, or compatibility"\
                  "- Insurance claim: Queries related to making insurance claim, guiding the insurance claim process."\
                  "- Other: Choose this if the query doesn’t fall into any of the other intents."\
                  "Question:"
 


def search_duckduckgo(query):
    """
    Perform a search query using DuckDuckGo and return the search results.
    """
    retval = []
    try:
        res = DDGS().text(query, max_results=5)
        body_values = [obj['body'] for obj in res]
        return body_values
    except Exception as ex:
        #print(ex)
        retval = []
    return retval

def generate_response_with_rag(question, retrieved_texts):
    """
    Generate a response using ChatGPT, incorporating retrieved texts.
    """
    prompt = (
        "You are an AI-powered search agent that takes in a user’s search query, retrieves relevant search results, and provides an accurate and concise answer based on the provided context."

        "## **Guidelines**"

        "### 1. **Prioritize Reliable Sources**"
        "- Use **Retrieved Information** when available, as it is the most likely authoritative source."
        "- Prefer **Wikipedia** if present in the search results for general knowledge queries."
        "- If there is a conflict between **Wikipedia** and the **Retrieved Information**, rely on **Wikipedia**."
        "- Prioritize **government (.gov), educational (.edu), reputable organizations (.org), and major news outlets** over less authoritative sources."
        "- When multiple sources provide conflicting information, prioritize the most **credible, recent, and consistent** source."

        "### 2. **Extract the Most Relevant Information**"
        "- Focus on **directly answering the query** using the information from the **Retrieved Information** or **SEARCH RESULTS**."
        "- Use **additional information** only if it provides **directly relevant** details that clarify or expand on the query."
        "- Ignore promotional, speculative, or repetitive content."

        "### 3. **Provide a Clear and Concise Answer**"
        "- Keep responses **brief (1–3 sentences)** while ensuring accuracy and completeness."
        "- If the query involves **numerical data** (e.g., prices, statistics), return the **most recent and precise value** available."
        "- If the source is available, then mention it in the answer to the question. If you're relying on the Retrieved Information, then do not mention the source if it's not there."
        "- For **diverse or expansive queries** (e.g., explanations, lists, or opinions), provide a more detailed response when the context justifies it."

        "### 4. **Handle Uncertainty and Ambiguity**"
        "- If **conflicting answers** are present, acknowledge the discrepancy and mention the different perspectives if relevant."
        "- If **no relevant information** is found in the context, explicitly state that the query could not be answered."

        "### 5. **Answer Validation**"
        "- Only return answers that can be **directly validated** from the provided context."
        "- Do not generate speculative or outside knowledge answers. If the context does not contain the necessary information, state that the answer could not be found."

        "### 6. **Bias and Neutrality**"
        "- Maintain **neutral language** and avoid subjective opinions."
        "- For controversial topics, present multiple perspectives if they are available and relevant..\n\n"
        "Question:" + question+ "\n\n"
        "Retrieved Information:\n"
         +  str(retrieved_texts)  + "\n"
       
    )
    
    return llm.complete(prompt)

def chat_on_duck():
    # User input: the question they want answered
    question = input("Enter your question: ")
    
    # Step 1: Retrieve information using DuckDuckGo
    retrieved_texts = search_duckduckgo(question)
    print(retrieved_texts[0] )
    if not retrieved_texts:
        print("No relevant information found.")
        retrieved_texts = []
        #return
    
    # Step 2: Generate a response using the retrieved information and ChatGPT
    response = generate_response_with_rag(question, retrieved_texts)
    
    # Output the response
    print("\nGenerated Response:\n")
    print(response)
def main():
    # User input: the question they want answered
    question = "Ninh Thuan"
    # Step 1: Retrieve information using DuckDuckGo
    retrieved_texts = search_duckduckgo(question)
    print(retrieved_texts )
    if not retrieved_texts:
        print("No relevant information found.")
        retrieved_texts = []
        #return
    
    # Step 2: Generate a response using the retrieved information and ChatGPT
    response = generate_response_with_rag(question, retrieved_texts)
    
    # Output the response
    print("\nGenerated Response:\n")
    print(response)

    # Step 3: Generate a response without the retrieved information and ChatGPT
    # response = generate_response_with_rag(question, [])
    
    # # Output the response
    # print("\nGenerated Response:\n")
    # print(response)

if __name__ == "__main__":
    main()