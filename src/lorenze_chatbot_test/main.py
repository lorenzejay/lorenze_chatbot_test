#!/usr/bin/env python
from pydantic import Field

from crewai import Agent, Flow
from crewai.flow import listen, start
from crewai.experimental.conversational import (
    ConversationConfig,
    ConversationState,
)
from crewai_tools import SerperDevTool

from lorenze_chatbot_test.crews.poem_crew.poem_crew import PoemCrew


# Hardcoded profiles -> route permissions. In a real app this comes from auth.
PROFILES = {
    "lorenze": {
        "internet_search": True,   # can run the deep-research route
        "poem_generation": False,  # cannot run the generic poem route
    },
}

# Which permission each route requires. Routes not listed are open to everyone.
ROUTE_PERMISSIONS = {
    "DEEP_RESEARCH": "internet_search",
    "POEM": "poem_generation",
}


class ChatState(ConversationState):
    user: str = ""
    permissions: dict[str, bool] = Field(default_factory=dict)
    denied_route: str = ""


@ConversationConfig(defer_trace_finalization=True, llm="gpt-5.4-mini")
class ProfileChatFlow(Flow[ChatState]):
    conversational = True

    @start()
    def load_profile(self) -> None:
        """Start step: identify the user, load route permissions, announce them.

        In a conversational flow custom @start() methods run (in order) before the
        built-in router each turn, so permissions are ready by the time route_turn
        runs. The guard keeps it to a one-time load per session.
        """
        if self.state.user:
            return  # already loaded for this session
        self.state.user = "lorenze"
        self.state.permissions = dict(PROFILES[self.state.user])
        allowed = [r for r, perm in ROUTE_PERMISSIONS.items()
                   if self.state.permissions.get(perm)]
        profile = (
            f"Profile loaded for {self.state.user}. "
            f"Allowed routes: {', '.join(allowed) or 'none'}."
        )
        print(profile)
        self.append_assistant_message(profile)

    def _has_permission(self, route: str) -> bool:
        required = ROUTE_PERMISSIONS.get(route)
        if required is None:
            return True  # open route
        return bool(self.state.permissions.get(required, False))

    def route_turn(self, context):
        # load_profile (an @start) has already run this turn, so permissions are set.
        message = (self.state.current_user_message or "").lower()

        if "poem" in message:
            intent = "POEM"
        elif any(k in message for k in ("research", "search", "find", "look up", "news")):
            intent = "DEEP_RESEARCH"
        else:
            return "converse"

        if not self._has_permission(intent):
            self.state.denied_route = intent
            return "PERMISSION_DENIED"
        return intent

    def research_agent(self) -> Agent:
        return Agent(
            role="Deep Research Specialist",
            goal="Answer the user's question with thorough, source-backed research.",
            backstory=(
                "You scour the web for current, credible information and always "
                "cite your sources."
            ),
            tools=[SerperDevTool()],
        )

    @listen("DEEP_RESEARCH")
    def handle_deep_research(self) -> str:
        """Internet-backed deep research on the user's topic."""
        result = self.research_agent().kickoff(self.state.current_user_message)
        reply = result.raw
        self.append_assistant_message(reply)
        return reply

    @listen("POEM")
    def handle_poem(self) -> str:
        """Generic route: have an LLM write a short poem."""
        result = PoemCrew().crew().kickoff(inputs={"sentence_count": 3})
        reply = result.raw
        self.append_assistant_message(reply)
        return reply

    @listen("PERMISSION_DENIED")
    def handle_permissions_denied(self) -> str:
        """Politely refuse routes the current user is not allowed to use."""
        route = self.state.denied_route or "that"
        required = ROUTE_PERMISSIONS.get(route, "the required")
        reply = (
            f"Sorry {self.state.user}, you don't have access to the '{route}' route "
            f"(it requires the '{required}' permission)."
        )
        self.append_assistant_message(reply)
        self.state.denied_route = ""
        return reply


def kickoff():
    ProfileChatFlow().chat()


def plot():
    ProfileChatFlow().plot()


if __name__ == "__main__":
    kickoff()
