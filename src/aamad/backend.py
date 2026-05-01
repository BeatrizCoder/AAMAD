from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml
import random
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load configuration
with open("config/agents.yaml", "r") as f:
    agents_config = yaml.safe_load(f)

with open("config/tasks.yaml", "r") as f:
    tasks_config = yaml.safe_load(f)

# Constants for tools
CATEGORY_KEYWORDS = {
    "Order Issues": ["order", "tracking", "shipment", "delivery", "package", "arrived", "delay"],
    "Billing": ["bill", "charge", "refund", "invoice", "payment", "price"],
    "Account Access": ["account", "login", "password", "sign in", "locked", "profile"],
    "Technical Issue": ["error", "bug", "crash", "failed", "site", "website", "problem"],
    "General Support": ["question", "help", "support", "information", "info", "request"],
}

KNOWLEDGE_BASE = {
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

SENTIMENT_NEGATIVE = [
    "angry", "upset", "frustrated", "disappointed", "unhappy", "mad", "annoyed",
    "worried", "concerned", "not happy", "terrible",
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

class ClassificationTool(BaseTool):
    name: str = "Classification Tool"
    description: str = "Classifies customer inquiries into support categories"

    def _run(self, inquiry: str) -> dict[str, Any]:
        """Classify inquiry using keyword matching"""
        inquiry_lower = inquiry.lower()
        scores = {cat: sum(word in inquiry_lower for word in keywords)
                 for cat, keywords in CATEGORY_KEYWORDS.items()}
        best = max(scores, key=scores.get)
        count = scores[best]
        confidence = min(95, 40 + count * 15)
        if best == "General Support" and count == 0:
            confidence = 55

        return {
            "category": best,
            "confidence": confidence,
            "scores": scores
        }


class SentimentTool(BaseTool):
    name: str = "Sentiment Analysis Tool"
    description: str = "Analyzes sentiment and urgency in customer messages"

    def _run(self, inquiry: str) -> dict[str, Any]:
        """Analyze sentiment using keyword matching"""
        inquiry_lower = inquiry.lower()
        found_negative = any(term in inquiry_lower for term in SENTIMENT_NEGATIVE)
        found_urgent = any(term in inquiry_lower for term in SENTIMENT_URGENT)

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

        return {
            "sentiment": label,
            "confidence": confidence,
            "urgency": urgency,
            "found_negative": found_negative,
            "found_urgent": found_urgent
        }


class KnowledgeTool(BaseTool):
    name: str = "Knowledge Retrieval Tool"
    description: str = "Retrieves knowledge base articles for support categories"

    def _run(self, category: str) -> dict[str, Any]:
        """Retrieve relevant articles for a category"""
        articles = KNOWLEDGE_BASE.get(category, KNOWLEDGE_BASE["General Support"])

        return {
            "articles": articles,
            "count": len(articles),
            "category": category
        }


class ResponseTool(BaseTool):
    name: str = "Response Generation Tool"
    description: str = "Generates appropriate customer responses based on context"

    def _run(self, category: str, urgency: str, articles_count: int) -> dict[str, Any]:
        """Generate appropriate response based on context"""
        template = RESPONSE_TEMPLATES.get(category, RESPONSE_TEMPLATES["General Support"])
        response = template
        if urgency == "High":
            response += " I have flagged this as a priority, and our support team will respond as soon as possible."

        confidence = 60
        if category != "General Support":
            confidence += 15
        if urgency == "Low":
            confidence += 5
        if articles_count >= 2:
            confidence += 10

        return {
            "response": response,
            "confidence": min(95, confidence),
            "template_used": category
        }


class EscalationTool(BaseTool):
    name: str = "Escalation Evaluation Tool"
    description: str = "Evaluates if cases need escalation to human support"

    def _run(self, response_confidence: int, sentiment: str, articles_count: int) -> dict[str, Any]:
        """Evaluate if case needs escalation"""
        escalate = False
        reason = "Sufficient confidence in automated response."

        if response_confidence < 55:
            escalate = True
            reason = "Low confidence in automated response."
        elif sentiment == "Concerned" and response_confidence < 70:
            escalate = True
            reason = "Sensitive issue with low confidence."
        elif articles_count == 0:
            escalate = True
            reason = "Insufficient knowledge base support."

        reference_id = f"ESC-2026-{random.randint(1000, 9999)}" if escalate else ""

        return {
            "escalation_required": escalate,
            "reason": reason,
            "reference_id": reference_id
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


class SupportState(BaseModel):
    inquiry: str = ""
    category: str = "General Support"
    category_confidence: int = 0
    sentiment: str = "Neutral"
    sentiment_confidence: int = 0
    urgency: str = "Low"
    articles: list[str] = []
    response: str = ""
    response_confidence: int = 0
    escalation_required: bool = False
    escalation_reason: str = ""
    reference_id: str = ""
    steps: list[dict[str, Any]] = []

    def log_step(self, agent_name: str, details: dict[str, Any]) -> None:
        self.steps.append({"agent": agent_name, "details": details})


class SupportCrew:
    def __init__(self):
        self.agents = {}
        self._create_agents()

    def _create_agents(self):
        """Create agents from configuration with tools"""
        for agent_name, config in agents_config.items():
            # Add tools based on agent role
            tools = []
            if agent_name == "classifier_agent":
                tools.append(ClassificationTool())
            elif agent_name == "sentiment_agent":
                tools.append(SentimentTool())
            elif agent_name == "knowledge_agent":
                tools.append(KnowledgeTool())
            elif agent_name == "response_agent":
                tools.append(ResponseTool())
            elif agent_name == "escalation_agent":
                tools.append(EscalationTool())

            self.agents[agent_name] = Agent(
                role=config["role"],
                goal=config["goal"],
                backstory=config["backstory"],
                tools=tools,
                verbose=True
            )

    def process_support_request(self, inquiry: str) -> Dict[str, Any]:
        """Process a support request using tools directly"""
        # Use tools directly for processing
        classification_result = self.agents["classifier_agent"].tools[0]._run(inquiry)
        sentiment_result = self.agents["sentiment_agent"].tools[0]._run(inquiry)
        knowledge_result = self.agents["knowledge_agent"].tools[0]._run(classification_result["category"])
        response_result = self.agents["response_agent"].tools[0]._run(
            classification_result["category"],
            sentiment_result["urgency"],
            knowledge_result["count"]
        )
        escalation_result = self.agents["escalation_agent"].tools[0]._run(
            response_result["confidence"],
            sentiment_result["sentiment"],
            knowledge_result["count"]
        )

        return {
            "category": classification_result["category"],
            "category_confidence": classification_result["confidence"],
            "sentiment": sentiment_result["sentiment"],
            "urgency": sentiment_result["urgency"],
            "response": response_result["response"],
            "response_confidence": response_result["confidence"],
            "escalation_required": escalation_result["escalation_required"],
            "escalation_reason": escalation_result["reason"],
            "reference_id": escalation_result["reference_id"],
            "steps": [
                {"agent": "Classifier Agent", "action": f"Classified as {classification_result['category']} with {classification_result['confidence']}% confidence", "tool_used": "ClassificationTool"},
                {"agent": "Sentiment Agent", "action": f"Analyzed sentiment as {sentiment_result['sentiment']}, urgency {sentiment_result['urgency']}", "tool_used": "SentimentTool"},
                {"agent": "Knowledge Agent", "action": f"Retrieved {knowledge_result['count']} knowledge articles", "tool_used": "KnowledgeTool"},
                {"agent": "Response Agent", "action": f"Generated response with {response_result['confidence']}% confidence", "tool_used": "ResponseTool"},
                {"agent": "Escalation Agent", "action": f"Escalation {'required' if escalation_result['escalation_required'] else 'not required'}", "tool_used": "EscalationTool"}
            ]
        }





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

    # Process the support request using the crew with tools
    result = support_crew.process_support_request(inquiry)

    # Build response model with all required fields
    response = SupportResponse(
        inquiry=inquiry,
        category=result["category"],
        category_confidence=result["category_confidence"],
        sentiment=result["sentiment"],
        sentiment_confidence=75,  # Default confidence for sentiment analysis
        urgency=result["urgency"],
        articles=[],  # Knowledge articles would be populated here
        response=result["response"],
        response_confidence=result["response_confidence"],
        escalation_required=result["escalation_required"],
        escalation_reason=result["escalation_reason"],
        reference_id=result["reference_id"],
        steps=result["steps"],
    )

    return response


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
