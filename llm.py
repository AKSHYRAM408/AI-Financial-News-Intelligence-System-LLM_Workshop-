"""
llm.py — LLM-powered market report generator & investor Q&A.

Uses the Mistral AI Python SDK to generate:
1. A structured Daily Market Intelligence Report (JSON dict).
2. An answer to the investor's specific question (plain text).
"""

import json
import logging
import os
from typing import Dict

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------
REPORT_SECTIONS = [
    "Executive Summary",
    "Market Overview",
    "Sector Performance",
    "Key Companies",
    "Macro Signals",
    "Risk Signals",
    "Outlook",
    "Disclaimer",
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
REPORT_SYSTEM_PROMPT = """You are a senior financial analyst at a leading investment bank.
Your task is to produce a **Daily Market Intelligence Report** based SOLELY on the
news articles provided by the user.

STRICT RULES:
1. Use ONLY the information from the provided articles. Do NOT hallucinate or invent data.
2. If a section cannot be filled because the articles lack relevant information, write:
   "Insufficient data from provided sources."
3. Be concise, factual, and professional.
4. Use bullet points where appropriate.
5. Respond with VALID JSON only — no markdown fences, no extra text.

Respond with a JSON object containing exactly these keys:
- "Executive Summary"   — 3-4 sentence overview of today's market developments.
- "Market Overview"     — Major index movements, trading volume context, and overall market sentiment.
- "Sector Performance"  — Which sectors outperformed/underperformed and why.
- "Key Companies"       — Notable company-specific news, earnings, deals, or leadership changes.
- "Macro Signals"       — Interest rates, inflation, central bank policies, GDP, employment data.
- "Risk Signals"        — Geopolitical risks, regulatory changes, supply chain issues.
- "Outlook"             — Short-term market expectations based on the provided news.
- "Disclaimer"          — Standard investment disclaimer: "This report is for informational purposes only and does not constitute investment advice. Past performance is not indicative of future results."
"""

REPORT_USER_TEMPLATE = """Analyse the following financial news articles and generate the Daily Market Intelligence Report in JSON format.

===== NEWS ARTICLES =====
{articles}
===== END OF ARTICLES =====

Respond with valid JSON only."""


QA_SYSTEM_PROMPT = """You are a knowledgeable stock market analyst and financial advisor assistant.
You answer investor questions based STRICTLY on the news articles provided.

STRICT RULES:
1. Use ONLY information present in the provided articles. Do NOT invent or hallucinate data.
2. If the articles do not contain enough information to answer the question, clearly say so.
3. Be concise, clear, and professional.
4. Use bullet points for structured answers when appropriate.
5. Always end with a brief disclaimer that this is for informational purposes only.
6. Respond in plain text (no JSON), using markdown formatting for readability.
"""

QA_USER_TEMPLATE = """Based on the following news articles, answer the investor's question.

===== NEWS ARTICLES =====
{articles}
===== END OF ARTICLES =====

**Investor's Question:** {question}

Provide a clear, well-structured answer based on the articles above."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client():
    """Build and return a Mistral client using env config."""
    api_key = os.getenv("MISTRAL_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key in ("your-api-key-here", "USE MISTRAL API KEY"):
        logger.error("MISTRAL_API_KEY is not set.")
        return None, "API key not configured. Please set MISTRAL_API_KEY in your .env file."

    client = Mistral(api_key=api_key)
    return client, None


def _empty_report(reason: str = "") -> Dict[str, str]:
    """Return an empty report structure with an optional reason."""
    report = {section: "No data available." for section in REPORT_SECTIONS}
    if reason:
        report["Executive Summary"] = reason
    report["Disclaimer"] = (
        "This report is for informational purposes only and does not "
        "constitute investment advice. Past performance is not indicative "
        "of future results."
    )
    return report


def _parse_report_response(raw: str) -> Dict[str, str]:
    """Attempt to parse the LLM response into a report dict."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM response as JSON: %s", exc)
        report = _empty_report("Report generation succeeded but JSON parsing failed.")
        report["Market Overview"] = raw[:2000]
        return report

    report: Dict[str, str] = {}
    for section in REPORT_SECTIONS:
        report[section] = data.get(section, "Insufficient data from provided sources.")
    return report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_market_report(cleaned_text: str) -> Dict[str, str]:
    """
    Generate a structured market intelligence report from cleaned article text.

    Parameters
    ----------
    cleaned_text : str
        Merged, cleaned article text produced by ``processor.clean_and_merge_articles``.

    Returns
    -------
    dict[str, str]
        A dictionary with keys matching ``REPORT_SECTIONS``.
    """
    if not cleaned_text or not cleaned_text.strip():
        logger.warning("Empty input text — returning empty report.")
        return _empty_report("No article text was provided for analysis.")

    client, error = _get_client()
    if error:
        return _empty_report(error)

    model = os.getenv("MISTRAL_MODEL", "") or os.getenv("OPENAI_MODEL", "mistral-small-latest")

    print("\n" + "=" * 80)
    print("🤖 LLM MARKET ANALYSIS — START")
    print("=" * 80)
    print(f"\n📤 SENDING TO LLM:")
    print(f"   Model: {model}")
    print(f"   Provider: Mistral AI")
    print(f"   Input text: {len(cleaned_text):,} characters")
    print(f"   Temperature: 0.3")
    print(f"   Max tokens: 2500")
    print(f"   System prompt: {len(REPORT_SYSTEM_PROMPT)} chars")
    print(f"   Expected output: JSON with {len(REPORT_SECTIONS)} sections")

    try:
        logger.info("Sending %d chars to LLM for report (model=%s).", len(cleaned_text), model)

        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": REPORT_USER_TEMPLATE.format(articles=cleaned_text)},
            ],
            temperature=0.3,
            max_tokens=2500,
        )

        raw_reply = response.choices[0].message.content or ""

        print(f"\n📥 LLM RESPONSE RECEIVED:")
        print(f"   Response length: {len(raw_reply):,} characters")
        print(f"   Preview: {raw_reply[:200]}...")

        report = _parse_report_response(raw_reply)

        print(f"\n📊 PARSED REPORT SECTIONS:")
        print("-" * 60)
        for section in REPORT_SECTIONS:
            content = report.get(section, "")
            status = "✅" if content and content != "No data available." else "⚠️"
            print(f"   {status} {section}: {len(content)} chars")

        print("\n" + "=" * 80)
        print("🤖 LLM MARKET ANALYSIS — COMPLETE")
        print("=" * 80 + "\n")

        logger.info("Received %d chars from LLM (report).", len(raw_reply))
        return report

    except Exception as exc:
        logger.error("LLM API call failed (report): %s", exc, exc_info=True)
        print(f"\n❌ LLM API FAILED: {exc}")
        return _empty_report(f"LLM API call failed: {exc}")


def answer_investor_question(cleaned_text: str, question: str) -> str:
    """
    Answer an investor's question using the provided news articles as context.

    Parameters
    ----------
    cleaned_text : str
        Merged, cleaned article text.
    question : str
        The investor's question.

    Returns
    -------
    str
        The AI-generated answer, or an error message.
    """
    if not cleaned_text or not cleaned_text.strip():
        return "❌ No news data available to answer your question. Please generate a report first."

    if not question or not question.strip():
        return "❌ Please enter a question."

    client, error = _get_client()
    if error:
        return f"❌ {error}"

    model = os.getenv("MISTRAL_MODEL", "") or os.getenv("OPENAI_MODEL", "mistral-small-latest")

    try:
        logger.info("Sending Q&A request to LLM (model=%s): '%s'", model, question[:80])

        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": QA_USER_TEMPLATE.format(
                    articles=cleaned_text,
                    question=question,
                )},
            ],
            temperature=0.4,
            max_tokens=1500,
        )

        answer = response.choices[0].message.content or ""
        logger.info("Received %d chars from LLM (Q&A).", len(answer))
        return answer.strip()

    except Exception as exc:
        logger.error("LLM API call failed (Q&A): %s", exc, exc_info=True)
        return f"❌ Failed to get answer: {exc}"


# ---------------------------------------------------------------------------
# Direct LLM query (fallback when no news data is available)
# ---------------------------------------------------------------------------

DIRECT_QUERY_SYSTEM_PROMPT = """You are a senior stock market analyst and financial advisor.
The user is asking a market-related question, but no recent news articles were available.
Answer based on your general financial knowledge.

RULES:
1. Be concise, professional, and helpful.
2. Clearly state that your answer is based on general knowledge, NOT on live market data.
3. Use bullet points and markdown formatting for readability.
4. Include relevant context about market trends, company fundamentals, or economic indicators.
5. Always end with a disclaimer that this is for informational purposes only and not investment advice.
6. If the question is not related to finance/markets, politely redirect the user.
"""

DIRECT_QUERY_USER_TEMPLATE = """The investor asked the following question, but no live news data was available from the API.
Please answer based on your general financial knowledge.

**Investor's Question:** {question}

Provide a clear, well-structured answer."""


def direct_llm_query(question: str) -> str:
    """
    Answer an investor's question directly using the LLM's own knowledge.
    Used as a fallback when the news API returns no results.

    Parameters
    ----------
    question : str
        The investor's question.

    Returns
    -------
    str
        The AI-generated answer, or an error message.
    """
    if not question or not question.strip():
        return "❌ Please enter a question."

    client, error = _get_client()
    if error:
        return f"❌ {error}"

    model = os.getenv("MISTRAL_MODEL", "") or os.getenv("OPENAI_MODEL", "mistral-small-latest")

    try:
        logger.info("Direct LLM query (model=%s): '%s'", model, question[:80])

        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": DIRECT_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": DIRECT_QUERY_USER_TEMPLATE.format(
                    question=question,
                )},
            ],
            temperature=0.4,
            max_tokens=2000,
        )

        answer = response.choices[0].message.content or ""
        logger.info("Received %d chars from LLM (direct query).", len(answer))
        return answer.strip()

    except Exception as exc:
        logger.error("LLM API call failed (direct query): %s", exc, exc_info=True)
        return f"❌ Failed to get answer: {exc}"
