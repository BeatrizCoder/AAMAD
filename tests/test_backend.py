"""Unit tests for the AAMAD backend support crew."""

from __future__ import annotations

from aamad.backend import SupportCrew, build_response_model, SupportTicket


def test_support_crew_process_returns_valid_context():
    crew = SupportCrew()
    context = crew.process("My order #12345 hasn't arrived yet.")

    assert context.inquiry
    assert context.category == "Order Issues"
    assert context.sentiment in {"Concerned", "Urgent", "Neutral"}
    assert len(context.articles) >= 1
    assert context.response
    assert context.reference_id.startswith("ESC-2026-")


def test_build_response_model_roundtrip():
    crew = SupportCrew()
    context = crew.process("I cannot access my account and I need help.")
    response = build_response_model(context)

    assert response.inquiry == context.inquiry
    assert response.category == context.category
    assert response.urgency == context.urgency
    assert response.escalation_required == context.escalation_required
    assert response.steps[0]["agent"] == "Classifier Agent"


def test_support_ticket_model():
    ticket = SupportTicket(inquiry="Please help with payment issues.")
    assert ticket.inquiry.startswith("Please help")
