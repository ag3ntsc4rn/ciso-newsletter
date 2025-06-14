"""
main.py

This module fetches recent CVEs, summarizes them, and sends a formatted report via email.
It uses OpenAI agents for subject writing and HTML conversion, and SendGrid for email delivery.
"""

import os
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

import requests
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
from dotenv import load_dotenv

from agents import Agent, Runner, trace, function_tool

# Load environment variables from .env file
load_dotenv(override=True)


def get_nvd_api_params() -> Dict[str, str]:
    """
    Generate parameters for the NVD API request for CVEs in the last 24 hours.

    Returns:
        dict: Parameters for the NVD API request.
    """
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    return {
        "pubStartDate": yesterday.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "pubEndDate": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "resultsPerPage": 10,
    }


def parse_cves(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Parse the CVE data from the NVD API response.

    Args:
        data (dict): The JSON response from the NVD API.

    Returns:
        list: List of dictionaries with CVE IDs and summaries.
    """
    cves = []
    for item in data.get("vulnerabilities", []):
        cve = item.get("cve", {})
        cves.append(
            {
                "id": cve.get("id"),
                "summary": cve.get("descriptions", [{}])[0].get("value", ""),
            }
        )
    return cves


@function_tool
def get_recent_cves() -> Dict[str, Any]:
    """
    Fetch CVEs reported in the last 24 hours from the NVD API.

    Returns:
        dict: Dictionary with a list of CVEs or an error message.
    """
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    api_key = os.getenv("NVD_API_KEY")
    params = get_nvd_api_params()
    headers = {"apiKey": api_key} if api_key else {}

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        cves = parse_cves(data)
        return {"cves": cves}
    except Exception as e:
        return {"error": str(e)}


@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """
    Send an HTML email with the given subject and body.

    Args:
        subject (str): The subject of the email.
        html_body (str): The HTML content of the email.

    Returns:
        dict: Status of the email sending operation.
    """
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
    from_email = Email(os.environ.get("SENDER_EMAIL"))
    to_email = To(os.environ.get("RECIPIENT_EMAIL"))
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    response = sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}


def build_agents() -> Dict[str, Agent]:
    """
    Build and return all agents used in the workflow.

    Returns:
        dict: Dictionary of agent instances.
    """
    subject_writer = Agent(
        name="Email subject writer",
        instructions=(
            "You can write a subject for a CVE report for the executive team. "
            "You are given a message and you need to write a subject for an email "
            "that is well understood by the executive team."
        ),
        model="gpt-4o-mini",
    )
    subject_tool = subject_writer.as_tool(
        tool_name="subject_writer",
        tool_description="Write a subject for a cold sales email",
    )

    html_converter = Agent(
        name="HTML email body converter",
        instructions=(
            "You can convert a text email body to an HTML email body. "
            "You are given a text email body which might have some markdown "
            "and you need to convert it to an HTML email body with simple, clear, compelling layout and design."
        ),
        model="gpt-4o-mini",
    )
    html_tool = html_converter.as_tool(
        tool_name="html_converter",
        tool_description="Convert a text email body to an HTML email body",
    )

    emailer_agent = Agent(
        name="Email Manager",
        instructions=(
            "You are an email formatter and sender. You receive the body of an email to be sent. "
            "You first use the subject_writer tool to write a subject for the email, then use the html_converter tool "
            "to convert the body to HTML. Finally, you use the send_html_email tool to send the email with the subject and HTML body."
        ),
        tools=[subject_tool, html_tool, send_html_email],
        model="gpt-4o-mini",
        handoff_description="Convert an email to HTML and send it",
    )

    cve_monitor = Agent(
        name="CVE Monitor",
        instructions=(
            "You are a security analyst. Your task is to monitor and report on recent CVEs (Common Vulnerabilities and Exposures). "
            "Use the get_recent_cves function to fetch CVEs reported in the last 24 hours. "
            "Summarize the findings and prepare a report for your executive team. "
            "Format the report in a nice markdown format. "
            "If there are no recent CVEs, state that clearly. "
            "Once the report is ready, hand it off to the emailer_agent to send it out."
        ),
        tools=[get_recent_cves],
        handoffs=[emailer_agent],
        model="gpt-4o-mini",
    )

    return {
        "subject_writer": subject_writer,
        "html_converter": html_converter,
        "emailer_agent": emailer_agent,
        "cve_monitor": cve_monitor,
    }


async def main() -> None:
    """
    Main entry point for the CVE monitoring and reporting workflow.
    Prepares a report on recent CVEs and sends it to the executive team.
    """
    agents = build_agents()
    cve_monitor = agents["cve_monitor"]
    message = "Prepare a report on recent CVEs for the executive team."

    with trace("CVE Monitoring"):
        await Runner.run(cve_monitor, message)


if __name__ == "__main__":
    asyncio.run(main())
