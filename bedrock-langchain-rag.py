import json
import os
import boto3
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
from langchain.schema import AIMessage, BaseMessage, ChatResult, HumanMessage, SystemMessage, ChatGeneration
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chat_models.base import BaseChatModel
from pydantic import Field

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
bedrock_runtime = boto3.client("bedrock-runtime")

class BedrockChatModel(BaseChatModel):
    model_id: str = Field(..., description="The Bedrock model ID")
    client: Any = Field(..., description="The Bedrock client")
    guardrails: Optional[str] = Field(None, description="Optional guardrails for the model")

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "bedrock-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "temperature": 0.7,
            "messages": [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content} for m in messages],
        }

        if self.guardrails:
            request_body["system"] = self.guardrails

        if stop:
            request_body["stop_sequences"] = stop

        response = self.client.invoke_model_with_response_stream(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
        
        full_response = ""
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'])
            if chunk['type'] == 'content_block_delta':
                full_response += chunk['delta']['text']
                if run_manager:
                    run_manager.on_llm_new_token(chunk['delta']['text'])
        
        logger.info(f"Bedrock model response: {full_response}")
        
        ai_message = AIMessage(content=full_response)
        chat_generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[chat_generation])

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_id": self.model_id}

def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event, indent=2)}")

    # Get environment variables
    bedrock_model_id = os.environ['BEDROCK_MODEL_ID']
    knowledge_base_id = os.environ['KNOWLEDGE_BASE_ID']

    logger.info(f"Using Bedrock Model ID: {bedrock_model_id}")
    logger.info(f"Using Knowledge Base ID: {knowledge_base_id}")

    # Define guardrails
    guardrails = """
    You are an AI assistant for product information. Be concise in your responses. Always be polite and professional.
    Never provide information about competitors' products.
    Do not discuss availability. If asked, say this information changes frequently and encourage users to visit our website or contact customer service for the most up-to-date information.
    Always respect user privacy and do not ask for or store personal information.
    """

    # Initialize BedrockChatModel with guardrails
    llm = BedrockChatModel(
        model_id=bedrock_model_id, 
        client=bedrock_runtime, 
        guardrails=guardrails
    )

    try:
        # Get user input from the event
        user_input = event.get('query')

        logger.info(f"User input: {user_input}")

        if not user_input:
            logger.error("No query provided in the event")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No query provided in the event'})
            }

        # Initialize AmazonKnowledgeBasesRetriever
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id,
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 3}},
            region_name="us-east-2"  # Adjust the region if necessary
        )

        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in docs])

        # Create a custom prompt template
        prompt_template = """
        You are an assistant for question-answering tasks for product information. 
        Be concise in your responses. Always be polite and professional.
        Never provide information about competitors' products.
        Do not discuss availability. If asked, say this information changes frequently and encourage users to visit our website or contact customer service for the most up-to-date information.
        Always respect user privacy and do not ask for or store personal information.

        Use the following context to answer the question:
        {context}

        Question: {question}
        Answer:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Create LLMChain
        chain = LLMChain(llm=llm, prompt=prompt)

        # Generate response
        response = chain.run(context=context, question=user_input)
        
        logger.info(f"Final AI response: {response}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'response': response
            })
        }

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'An unexpected error occurred.',
                'details': str(e)
            })
        }
