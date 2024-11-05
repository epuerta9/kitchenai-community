from kitchenai.contrib.kitchenai_sdk.kitchenai import KitchenAIApp
from ninja import Schema
from django.http import StreamingHttpResponse

import os

from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from toolhouse import Toolhouse, Provider
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

os.environ[
    "TOOLHOUSE_API_KEY"
]
os.environ[
    "GROQ_API_KEY"
] 


kitchen = KitchenAIApp()

class WebsiteContentEvent(Event):
    contents: str


class WebSearchEvent(Event):
    results: str


class RankingEvent(Event):
    results: str


class LogEvent(Event):
    msg: str


class SalesRepWorkflow(Workflow):

    llm = OpenAI()

    th = Toolhouse(provider=Provider.LLAMAINDEX)
    th.set_metadata("id", "llamaindex_agent")
    th.set_metadata("timezone", 0)

    agent = ReActAgent(
        tools=th.get_tools(bundle="llamaindex test"),
        llm=llm,
        memory=ChatMemoryBuffer.from_defaults(),
    )

    @step
    async def get_company_info(
        self, ctx: Context, ev: StartEvent
    ) -> WebsiteContentEvent:
        ctx.write_event_to_stream(
            LogEvent(msg=f"Getting the contents of {ev.url}…")
        )
        prompt = f"Get the contents of {ev.url}, then summarize its key value propositions in a few bullet points."
        contents = await self.agent.achat(prompt)
        return WebsiteContentEvent(contents=str(contents.response))

    @step
    async def find_prospects(
        self, ctx: Context, ev: WebsiteContentEvent
    ) -> WebSearchEvent:
        ctx.write_event_to_stream(
            LogEvent(
                msg=f"Performing web searches to identify companies who can benefit from the business's offerings."
            )
        )
        prompt = f"With that you know about the business, perform a web search to find 5 tech companies who may benefit from the business's product. Only answer with the names of the companies you chose."
        results = await self.agent.achat(prompt)
        return WebSearchEvent(results=str(results.response))

    @step
    async def select_best_company(
        self, ctx: Context, ev: WebSearchEvent
    ) -> RankingEvent:
        ctx.write_event_to_stream(
            LogEvent(
                msg=f"Selecting the best company who can benefit from the business's offering…"
            )
        )
        prompt = "Select one company that can benefit from the business's product. Only use your knowledge to select the company. Respond with just the name of the company. Do not use tools."
        results = await self.agent.achat(prompt)
        ctx.write_event_to_stream(
            LogEvent(
                msg=f"The agent selected this company: {results.response}"
            )
        )
        return RankingEvent(results=str(results.response))

    @step
    async def prepare_email(self, ctx: Context, ev: RankingEvent) -> StopEvent:
        ctx.write_event_to_stream(
            LogEvent(msg=f"Drafting a short email for sales outreach…")
        )
        prompt = f"Draft a short cold sales outreach email for the company you picked. Do not use tools."
        email = await self.agent.achat(prompt)
        ctx.write_event_to_stream(
            LogEvent(msg=f"Here is the email: {email.response}")
        )
        return StopEvent(result=str(email.response))
    

class SalesRepWorkflowInput(Schema):
    url: str

@kitchen.runnable("sales-rep-workflow", streaming=True)
async def sales_rep_workflow(request, input: SalesRepWorkflowInput):
    workflow = SalesRepWorkflow(timeout=None)
    handler = workflow.run(url=input.url)

    async for event in handler.stream_events():
        if isinstance(event, LogEvent):
            yield event.msg
