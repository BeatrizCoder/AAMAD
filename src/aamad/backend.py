from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


KNOWLEDGE_BASE: dict[str, list[str]] = {
    "Order Issues": [
        "Order Delivery Delays",
        "Tracking Your Package",
        "Order Status & Estimated Delivery",
    ],
    "Billing": [
        "Refunds and Billing Questions",
        "Understanding Your Invoice",
        "Payment Methods & Authorization Holds",
    ],
    "Account Access": [
        "Resetting Your Password",
        "Recovering a Locked Account",
        "Updating Account Information",
    ],
    "Technical Issue": [
        "Troubleshooting Login Errors",
        "Resolving Site Performance Problems",
        "Clearing Browser Cache",
    ],
    "General Support": [
        "Contacting Customer Service",
        "Using Our Help Center",
    ],
}

CATEGORY_KEYWORDS = {
    "Order Issues": [
        "order",
        "tracking",
        "shipment",
        "delivery",
        "package",
        "arrived",
        "delay",
    ],
    "Billing": [
        "bill",
        "charge",
        "refund",
        "invoice",
        "payment",
        "price",
    ],
    "Account Access": [
        "account",
        "login",
        "password",
        "sign in",
        "locked",
        "profile",
    ],
    "Technical Issue": [
        "error",
        "bug",
        "crash",
        "failed",
        "site",
        "website",
        "problem",
    ],
    "General Support": [
        "question",
        "help",
        "support",
        "information",
        "info",
        "request",
    ],
}

SENTIMENT_NEGATIVE = [
    "angry",
    "upset",
    "frustrated",
    "disappointed",
    "unhappy",
    "mad",
    "annoyed",
    "worried",
    "concerned",
    "not happy",
    "terrible",
]

SENTIMENT_URGENT = ["urgent", "asap", "immediately", "right away", "now"]

RESPONSE_TEMPLATES = {
    "Order Issues": (
        "I'm sorry to hear about the delay with your order. "
        "It looks like your shipment is experiencing a temporary delay, and it should arrive within the next 2-3 business days. "
        "If you'd like, I can provide more details on tracking and next steps."
    ),
    "Billing": (
        "I understand your concern about your billing statement. "
        "Our records show that the charge was processed correctly, but I can help review the payment details or initiate a refund request if needed."
    ),
    "Account Access": (
        "I can help you regain access to your account. "
        "Please try resetting your password from the sign-in page, and if the account is locked, I will escalate to our account recovery team."
    ),
    "Technical Issue": (
        "Thank you for letting us know about this technical issue. "
        "I recommend refreshing the page or clearing your browser cache, and if the problem persists, I can escalate it to our support engineers."
    ),
    "General Support": (
        "Thanks for reaching out. "
        "I can help answer your question or point you to the right support article so you can get a fast resolution."
    ),
}


class SupportTicket(BaseModel):
    inquiry: str


class SupportResponse(BaseModel):
    inquiry: str
    category: str
    category_confidence: int
    sentiment: str
    sentiment_confidence: int
    urgency: str
    articles: list[str]
    response: str
    response_confidence: int
    escalation_required: bool
    escalation_reason: str
    reference_id: str
    steps: list[dict[str, Any]]


class SupportContext:
    def __init__(self, inquiry: str) -> None:
        self.inquiry = inquiry
        self.category = "General Support"
        self.category_confidence = 0
        self.sentiment = "Neutral"
        self.sentiment_confidence = 0
        self.urgency = "Low"
        self.articles: list[str] = []
        self.response = ""
        self.response_confidence = 0
        self.escalation_required = False
        self.escalation_reason = ""
        self.reference_id = ""
        self.steps: list[dict[str, Any]] = []

    def log_step(self, name: str, details: dict[str, Any]) -> None:
        self.steps.append({"agent": name, "details": details})


class SupportAgent(ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, context: SupportContext) -> None:
        ...


class ClassifierAgent(SupportAgent):
    def __init__(self) -> None:
        super().__init__("Classifier Agent")

    def run(self, context: SupportContext) -> None:
        normalized = context.inquiry.lower()
        scores = {cat: sum(word in normalized for word in keywords) for cat, keywords in CATEGORY_KEYWORDS.items()}
        best = max(scores, key=scores.get)
        count = scores[best]
        confidence = min(95, 40 + count * 15)
        if best == "General Support" and count == 0:
            confidence = 55
        context.category = best
        context.category_confidence = confidence
        context.log_step(
            self.name,
            {
                "category": context.category,
                "confidence": context.category_confidence,
            },
        )


class SentimentAnalysisAgent(SupportAgent):
    def __init__(self) -> None:
        super().__init__("Sentiment Analysis Agent")

    def run(self, context: SupportContext) -> None:
        normalized = context.inquiry.lower()
        found_negative = any(term in normalized for term in SENTIMENT_NEGATIVE)
        found_urgent = any(term in normalized for term in SENTIMENT_URGENT)
        if found_negative:
            label = "Concerned"
            confidence = 80
        elif found_urgent:
            label = "Urgent"
            confidence = 70
        else:
            label = "Neutral"
            confidence = 65
        urgency = "High" if found_urgent else "Medium" if found_negative else "Low"
        context.sentiment = label
        context.sentiment_confidence = confidence
        context.urgency = urgency
        context.log_step(
            self.name,
            {
                "sentiment": context.sentiment,
                "confidence": context.sentiment_confidence,
                "urgency": context.urgency,
            },
        )


class KnowledgeRetrievalAgent(SupportAgent):
    def __init__(self) -> None:
        super().__init__("Knowledge Retrieval Agent")

    def run(self, context: SupportContext) -> None:
        context.articles = KNOWLEDGE_BASE.get(context.category, KNOWLEDGE_BASE["General Support"])
        context.log_step(
            self.name,
            {
                "articles_found": len(context.articles),
                "articles": context.articles[:3],
            },
        )


class ResponseGenerationAgent(SupportAgent):
    def __init__(self) -> None:
        super().__init__("Response Generation Agent")

    def run(self, context: SupportContext) -> None:
        template = RESPONSE_TEMPLATES.get(context.category, RESPONSE_TEMPLATES["General Support"])
        response = template
        if context.urgency == "High":
            response += " I have flagged this as a priority, and our support team will respond as soon as possible."
        confidence = 60
        if context.category != "General Support":
            confidence += 15
        if context.sentiment == "Neutral":
            confidence += 5
        if len(context.articles) >= 2:
            confidence += 10
        context.response = response
        context.response_confidence = min(95, confidence)
        context.log_step(
            self.name,
            {
                "response": context.response,
                "confidence": context.response_confidence,
            },
        )


class EscalationAgent(SupportAgent):
    def __init__(self) -> None:
        super().__init__("Escalation Agent")

    def run(self, context: SupportContext) -> None:
        escalate = False
        reason = "Sufficient confidence in automated response."
        if context.response_confidence < 55:
            escalate = True
            reason = "Low confidence in automated response."
        elif context.sentiment == "Concerned" and context.response_confidence < 70:
            escalate = True
            reason = "Sensitive issue with low confidence."
        elif not context.articles:
            escalate = True
            reason = "Insufficient knowledge base support."
        context.escalation_required = escalate
        context.escalation_reason = reason
        context.reference_id = self._build_reference_id()
        context.log_step(
            self.name,
            {
                "escalation_required": context.escalation_required,
                "reason": context.escalation_reason,
                "reference_id": context.reference_id,
            },
        )

    def _build_reference_id(self) -> str:
        return f"ESC-2026-{random.randint(1000, 9999)}"


class FrontlineCrew:
    def __init__(self) -> None:
        self.agents: list[SupportAgent] = [
            ClassifierAgent(),
            SentimentAnalysisAgent(),
            KnowledgeRetrievalAgent(),
            ResponseGenerationAgent(),
        ]

    def execute(self, context: SupportContext) -> None:
        for agent in self.agents:
            agent.run(context)


class EscalationCrew:
    def __init__(self) -> None:
        self.agents: list[SupportAgent] = [EscalationAgent()]

    def execute(self, context: SupportContext) -> None:
        for agent in self.agents:
            agent.run(context)


class SupportCrew:
    def __init__(self) -> None:
        self.frontline = FrontlineCrew()
        self.escalation = EscalationCrew()

    def process(self, inquiry: str) -> SupportContext:
        context = SupportContext(inquiry=inquiry)
        self.frontline.execute(context)
        self.escalation.execute(context)
        return context


def build_response_model(context: SupportContext) -> SupportResponse:
    return SupportResponse(
        inquiry=context.inquiry,
        category=context.category,
        category_confidence=context.category_confidence,
        sentiment=context.sentiment,
        sentiment_confidence=context.sentiment_confidence,
        urgency=context.urgency,
        articles=context.articles,
        response=context.response,
        response_confidence=context.response_confidence,
        escalation_required=context.escalation_required,
        escalation_reason=context.escalation_reason,
        reference_id=context.reference_id,
        steps=context.steps,
    )


app = FastAPI(
    title="AAMAD Support Backend",
    description="Backend API for the CrewAI multi-agent support interface.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

support_crew = SupportCrew()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "aamad backend"}


@app.post("/api/support", response_model=SupportResponse)
async def create_support_ticket(ticket: SupportTicket) -> SupportResponse:
    inquiry = ticket.inquiry.strip()
    if not inquiry:
        raise HTTPException(status_code=400, detail="Inquiry text cannot be empty.")
    context = support_crew.process(inquiry)
    return build_response_model(context)


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    uvicorn.run(
        "aamad.backend:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
