import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="Choreo-RAG")

client = OpenAI()

while True:
    user_query = input("\n\nWhat do you want to know about Choreo platform? (Type 'exit' to quit)\n\n")

    # Exit the loop if the user wants to quit
    if user_query.lower() in ["exit", "quit"]:
        print("Thank you for using the assistant! Goodbye!")
        break

    # Fetch the query results from the collection
    results = collection.query(
        query_texts=[user_query],
        n_results=1
    )

    # Prepare the system prompt
    system_prompt = """
    You are a helpful assistant. You answer questions about Choreo platform by WSO2. 
    Answer the questions based on knowledge I'm providing you. Do not engage in any other conversations that isn't related to Choreo platform. If the user is asking for other information, you can respond with the following message:

        `I apologize for the inconvenience, but I am only able to provide the information on the Choreo platform.`
    --------------------
    You should provide a detailed answer in an explanatory manner to the user's query. Here is the data you need to provide the answer:
    The data:
    """ + str(results['documents']) + """
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    print("\n\n---------------------\n")
    print(response.choices[0].message.content)
