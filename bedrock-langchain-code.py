import json
import boto3
from langchain.llms import Bedrock
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import DynamoDBChatMessageHistory, ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Initialize AWS clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-2')
dynamodb = boto3.resource('dynamodb')

# Initialize Bedrock LLM
llm = Bedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    client=bedrock_runtime,
    model_kwargs={"temperature": 0.7, "max_tokens_to_sample": 500}
)

# Create a custom retriever function that uses Bedrock's retrieval
def bedrock_retriever(query):
    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-3-haiku-20240307-v1:0',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            "query": query,
            "numberOfResults": 3  # Adjust as needed
        })
    )
    result = json.loads(response['body'].read())
    return result['passages']  # Assuming the response contains relevant passages

# Create ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=bedrock_retriever,
    return_source_documents=True,
    combine_docs_chain_kwargs={
        "prompt": PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}
            Answer:""",
            input_variables=["context", "question"],
        ),
    }
)

def lambda_handler(event, context):
    # Parse the incoming event
    body = json.loads(event['body'])
    query = body.get('query')
    conversation_id = body.get('conversation_id', 'default')

    # Set up DynamoDB chat message history
    message_history = DynamoDBChatMessageHistory(
        table_name="ChatHistory",
        session_id=conversation_id
    )

    # Set up ConversationBufferMemory with DynamoDB backend
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        return_messages=True
    )

    # Generate a response
    result = qa_chain({
        "question": query,
        "chat_history": memory.chat_memory.messages
    })
    
    answer = result['answer']
    source_documents = result['source_documents']

    # Save the new message to chat history
    memory.save_context({"input": query}, {"output": answer})

    return {
        'statusCode': 200,
        'body': json.dumps({
            'answer': answer,
            'sources': source_documents,
            'conversation_id': conversation_id
        }),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }

