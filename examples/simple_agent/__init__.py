"""Simple agent example for AgentEval."""

from agenteval.types import (
    AgentInput,
    AgentMetadata,
    AgentOutput,
    StepType,
    TrajectoryStep,
)


def simple_qa_agent(input: AgentInput) -> AgentOutput:
    """A simple Q&A agent for demonstration.

    This agent simulates a basic question-answering workflow:
    1. Analyzes the query
    2. Searches for relevant information
    3. Generates a response

    Args:
        input: The agent input containing a query.

    Returns:
        AgentOutput with the response and trajectory.
    """
    query = input.query.lower()

    # Simulate step 1: Query analysis
    steps = [
        TrajectoryStep(
            step_index=0,
            step_type=StepType.REASONING,
            name="analyze_query",
            parameters={"query": input.query},
            output={"intent": "question", "topics": ["general"]},
            duration_ms=10,
            tokens=20,
        )
    ]

    # Simulate step 2: Search (if needed)
    if "weather" in query or "price" in query or "flight" in query:
        steps.append(
            TrajectoryStep(
                step_index=1,
                step_type=StepType.TOOL_CALL,
                name="search",
                parameters={"query": input.query},
                output={"results": ["result 1", "result 2"]},
                duration_ms=50,
                tokens=100,
            )
        )

    # Simulate step 3: Generate response
    if "hello" in query or "hi" in query:
        response = "Hello! How can I help you today?"
    elif "weather" in query:
        response = "The weather today is sunny with a high of 72F."
    elif "flight" in query:
        response = "I found several flights. The cheapest flight is $299."
    elif "help" in query:
        response = "I can help you with questions, searches, and various tasks."
    else:
        response = f"I understand you're asking about: {input.query}. Let me help you with that."

    steps.append(
        TrajectoryStep(
            step_index=len(steps),
            step_type=StepType.OUTPUT,
            name="generate_response",
            output=response,
            duration_ms=30,
            tokens=50,
        )
    )

    total_tokens = sum(s.tokens for s in steps)
    total_duration = sum(s.duration_ms for s in steps)

    return AgentOutput(
        output=response,
        steps=steps,
        metadata=AgentMetadata(
            total_tokens=total_tokens,
            prompt_tokens=total_tokens // 2,
            completion_tokens=total_tokens // 2,
            cost=total_tokens * 0.00001,  # $0.01 per 1000 tokens
            duration_ms=total_duration,
        ),
        success=True,
    )


def unreliable_agent(input: AgentInput) -> AgentOutput:
    """An agent that fails randomly for testing purposes.

    This agent has approximately 70% success rate.
    """
    import random

    if random.random() < 0.3:
        # Simulate failure
        return AgentOutput(
            output="I encountered an error processing your request.",
            steps=[
                TrajectoryStep(
                    step_index=0,
                    step_type=StepType.REASONING,
                    name="error",
                    output="Internal error",
                    duration_ms=5,
                )
            ],
            metadata=AgentMetadata(
                total_tokens=10,
                cost=0.0001,
                duration_ms=5,
            ),
            success=False,
            error="Random failure for testing",
        )

    return simple_qa_agent(input)
