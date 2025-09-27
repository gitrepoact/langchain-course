from typing import List, Any, Dict, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs.llm_result import LLMResult

class   AgentCallBackHandler(BaseCallbackHandler):

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running.

        .. ATTENTION::
            This method is called for non-chat models (regular LLMs). If you're
            implementing a handler for a chat model, you should use
            ``on_chat_model_start`` instead.

        Args:
            serialized (dict[str, Any]): The serialized LLM.
            prompts (list[str]): The prompts.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            metadata (Optional[dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """
        print(f"*** Prompt to LLM was:***\n{prompts[0]}")
        print("****")

    def on_llm_end(
        self,
        response: LLMResult,
        *, # type: ignore
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The response which was generated.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """

        print(f"***LLM Response was:***\n{response.generations[0][0].text}")
        print("***")



