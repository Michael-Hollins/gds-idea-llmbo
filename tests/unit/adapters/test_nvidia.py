from conftest import ExampleOutput

from llmbo.adapters import NvidiaAdapter
from llmbo.models import ModelInput


def test_build_tool():
    """Test building a tool definition for NVIDIA Nemotron."""
    tool = NvidiaAdapter.build_tool(ExampleOutput)

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "ExampleOutput"
    assert "parameters" in tool["function"]


def test_prepare_model_input():
    """Test system prompt migration and anthropic_version removal."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
        anthropic_version="bedrock-2023-05-31",
        system="You are helpful.",
    )

    result = NvidiaAdapter.prepare_model_input(model_input)

    assert result.anthropic_version is None
    assert result.system is None
    assert result.messages[0]["role"] == "system"


def test_prepare_model_input_with_tool():
    """Test preparing model input with an output model."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
    )

    result = NvidiaAdapter.prepare_model_input(model_input, ExampleOutput)

    assert result.tools is not None
    assert result.tool_choice == "required"


def test_validate_result_valid():
    """Test validate_result with a valid native response."""
    valid_result = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"name": "Ada", "age": 36}',
                                "name": "ExampleOutput",
                            },
                            "type": "function",
                        }
                    ],
                },
            }
        ],
    }

    result = NvidiaAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "Ada"
    assert result.age == 36


def test_validate_result_empty():
    """Test validate_result with empty result."""
    result = NvidiaAdapter.validate_result({}, ExampleOutput)
    assert result is None
