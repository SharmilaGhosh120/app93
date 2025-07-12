
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
import logging
from typing import List, Dict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
import uuid
import asyncio

# Download NLTK data for text processing
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging for monitoring and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app for RESTful API endpoints
app = FastAPI()

# Pydantic model for issue input
class IssueInput(BaseModel):
    customer_id: str
    issue_description: str
    product_id: str
    timestamp: str

# Database setup (using SQLite for demonstration; replace with MySQL in production)
def init_db():
    conn = sqlite3.connect('support_issues.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS issues (
            issue_id TEXT PRIMARY KEY,
            customer_id TEXT,
            product_id TEXT,
            issue_description TEXT,
            severity TEXT,
            criticality TEXT,
            status TEXT,
            created_at TEXT,
            last_updated TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            issue_id TEXT,
            message TEXT,
            sender TEXT,
            timestamp TEXT,
            FOREIGN KEY (issue_id) REFERENCES issues (issue_id)
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Function to analyze issue severity based on keywords
def analyze_severity(description: str) -> str:
    critical_keywords = ['crash', 'failure', 'urgent', 'down', 'critical']
    high_keywords = ['error', 'bug', 'issue', 'problem']
    try:
        tokens = word_tokenize(description.lower())
        tokens = [t for t in tokens if t not in stopwords.words('english') and t not in string.punctuation]
    except Exception as e:
        logger.error(f"Error in tokenization: {str(e)}")
        return "Low"  # Fallback to Low severity if tokenization fails

    if any(keyword in tokens for keyword in critical_keywords):
        return "High"
    elif any(keyword in tokens for keyword in high_keywords):
        return "Normal"
    else:
        return "Low"

# Function to find similar issues based on description
def find_similar_issues(description: str) -> List[Dict]:
    conn = sqlite3.connect('support_issues.db')
    c = conn.cursor()
    c.execute('SELECT issue_id, issue_description, status FROM issues')
    all_issues = c.fetchall()
    conn.close()

    similar_issues = []
    try:
        tokens_new = set(word_tokenize(description.lower()))
        tokens_new = {t for t in tokens_new if t not in stopwords.words('english') and t not in string.punctuation}
    except Exception as e:
        logger.error(f"Error in tokenization for similarity search: {str(e)}")
        return similar_issues  # Return empty list if tokenization fails

    for issue in all_issues:
        issue_id, issue_desc, status = issue
        try:
            tokens_old = set(word_tokenize(issue_desc.lower()))
            tokens_old = {t for t in tokens_old if t not in stopwords.words('english') and t not in string.punctuation}
            common_tokens = tokens_new.intersection(tokens_old)
            if len(common_tokens) > 2:  # Threshold for similarity
                similar_issues.append({"issue_id": issue_id, "description": issue_desc, "status": status})
        except Exception as e:
            logger.error(f"Error processing issue {issue_id}: {str(e)}")
            continue

    return similar_issues

# Function to check for unattended critical issues
def check_unattended_critical(customer_id: str, product_id: str) -> List[Dict]:
    conn = sqlite3.connect('support_issues.db')
    c = conn.cursor()
    c.execute('''
        SELECT issue_id, created_at FROM issues
        WHERE customer_id = ? AND product_id = ? AND criticality = 'High' AND status = 'Open'
    ''', (customer_id, product_id))
    issues = c.fetchall()
    conn.close()

    unattended = []
    now = datetime.now()
    for issue_id, created_at in issues:
        try:
            created_time = datetime.fromisoformat(created_at)
            if now - created_time > timedelta(hours=24):
                unattended.append({"issue_id": issue_id, "created_at": created_at})
        except ValueError as e:
            logger.error(f"Error parsing timestamp for issue {issue_id}: {str(e)}")
            continue

    return unattended

# Function to generate message template
def generate_message_template(issue: IssueInput, past_issues: int, similar_issues: List[Dict], severity: str, criticality: str) -> str:
    template = f"""
    Dear Customer,

    Thank you for reaching out regarding your issue with {issue.product_id}.
    We have identified your issue as {severity} severity and {criticality} criticality.

    Based on your history ({past_issues} past issues) and similar resolved cases, we recommend the following steps:
    """

    if similar_issues:
        template += "\nSimilar issues found:\n"
        for sim_issue in similar_issues:
            template += f"- Issue ID: {sim_issue['issue_id']} (Status: {sim_issue['status']})\n"

    template += "\nPlease provide additional details or confirm if the suggested steps resolve your issue.\nBest regards,\nSupport Team"

    return template

# Function to summarize conversation
def summarize_conversation(issue_id: str) -> str:
    conn = sqlite3.connect('support_issues.db')
    c = conn.cursor()
    # Retrieve issue description
    c.execute('SELECT issue_description, status FROM issues WHERE issue_id = ?', (issue_id,))
    issue_data = c.fetchone()
    # Retrieve conversation messages
    c.execute('SELECT message, sender, timestamp FROM conversations WHERE issue_id = ? ORDER BY timestamp', (issue_id,))
    messages = c.fetchall()
    conn.close()

    if not issue_data:
        return f"No issue found for Issue ID {issue_id}."

    issue_description, issue_status = issue_data
    summary = f"Summary for Issue ID {issue_id} ({issue_status}):\n"

    if not messages:
        summary += f"Customer reported: {issue_description[:100]}... No further conversation recorded."
        return summary

    # Extract key points using basic NLP
    try:
        customer_messages = [msg[0] for msg in messages if msg[1].lower() == 'customer']
        support_messages = [msg[0] for msg in messages if msg[1].lower() == 'support']

        # Tokenize and summarize issue description
        issue_sentences = sent_tokenize(issue_description)
        main_issue = issue_sentences[0] if issue_sentences else "Issue details not available."

        # Summarize customer and support interactions
        customer_summary = ""
        if customer_messages:
            customer_text = " ".join(customer_messages)
            customer_sentences = sent_tokenize(customer_text)[:2]  # Take first two sentences for brevity
            customer_summary = " ".join(customer_sentences)[:150] + "..." if len(customer_text) > 150 else customer_text

        support_summary = ""
        if support_messages:
            support_text = " ".join(support_messages)
            support_sentences = sent_tokenize(support_text)[:2]  # Take first two sentences for brevity
            support_summary = " ".join(support_sentences)[:150] + "..." if len(support_text) > 150 else support_text

        # Construct concise summary
        summary += f"Customer reported: {main_issue[:100]}... "
        if customer_summary:
            summary += f"Customer elaborated: {customer_summary} "
        if support_summary:
            summary += f"Support responded: {support_summary} "
        summary += f"Current status: {issue_status}."

    except Exception as e:
        logger.error(f"Error summarizing conversation for issue {issue_id}: {str(e)}")
        summary += f"Customer reported: {issue_description[:100]}... Conversation details unavailable due to processing error."

    return summary

# FastAPI endpoint for creating a new issue
@app.post("/api/issues")
async def create_issue(issue: IssueInput):
    try:
        issue_id = str(uuid.uuid4())
        severity = analyze_severity(issue.issue_description)
        criticality = "High" if severity == "High" else "Normal"

        # Count past issues
        conn = sqlite3.connect('support_issues.db')
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM issues WHERE customer_id = ?', (issue.customer_id,))
        past_issues = c.fetchone()[0]

        # Find similar issues
        similar_issues = find_similar_issues(issue.issue_description)

        # Check unattended critical issues
        unattended_issues = check_unattended_critical(issue.customer_id, issue.product_id)

        # Insert new issue
        c.execute('''
            INSERT INTO issues (issue_id, customer_id, product_id, issue_description, severity, criticality, status, created_at, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (issue_id, issue.customer_id, issue.product_id, issue.issue_description, severity, criticality, 'Open', issue.timestamp, issue.timestamp))
        conn.commit()
        conn.close()

        # Generate message template
        message_template = generate_message_template(issue, past_issues, similar_issues, severity, criticality)

        return {
            "issue_id": issue_id,
            "severity": severity,
            "criticality": criticality,
            "past_issues": past_issues,
            "similar_issues": similar_issues,
            "unattended_critical": unattended_issues,
            "message_template": message_template
        }
    except Exception as e:
        logger.error(f"Error creating issue: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint for conversation summary
@app.get("/api/conversation_summary/{issue_id}")
async def get_conversation_summary(issue_id: str):
    try:
        summary = summarize_conversation(issue_id)
        return {"issue_id": issue_id, "summary": summary}
    except Exception as e:
        logger.error(f"Error summarizing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit app
def main():
    st.set_page_config(page_title="Support Copilot", layout="wide")
    st.title("AI-Powered Support Copilot")

    # Initialize session state
    if 'issue_submitted' not in st.session_state:
        st.session_state.issue_submitted = False
        st.session_state.issue_response = None

    # Issue submission form
    with st.form("issue_form"):
        st.header("Submit a New Issue")
        customer_id = st.text_input("Customer ID")
        product_id = st.text_input("Product ID")
        issue_description = st.text_area("Issue Description")
        submitted = st.form_submit_button("Submit Issue")

        if submitted and customer_id and product_id and issue_description:
            issue_data = {
                "customer_id": customer_id,
                "product_id": product_id,
                "issue_description": issue_description,
                "timestamp": datetime.now().isoformat()
            }
            try:
                # Simulate API call (in production, use requests.post to call the FastAPI endpoint)
                async def submit_issue():
                    return await create_issue(IssueInput(**issue_data))

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(submit_issue())

                st.session_state.issue_submitted = True
                st.session_state.issue_response = response
            except Exception as e:
                st.error(f"Error submitting issue: {str(e)}")

    # Display results
    if st.session_state.issue_submitted and st.session_state.issue_response:
        response = st.session_state.issue_response
        st.header("Issue Analysis")
        st.write(f"Issue ID: {response['issue_id']}")
        st.write(f"Severity: {response['severity']}")
        st.write(f"Criticality: {response['criticality']}")
        st.write(f"Past Issues by Customer: {response['past_issues']}")

        if response['similar_issues']:
            st.subheader("Similar Issues Found")
            for issue in response['similar_issues']:
                st.write(f"- Issue ID: {issue['issue_id']} (Status: {issue['status']})")

        if response['unattended_critical']:
            st.subheader("Unattended Critical Issues")
            for issue in response['unattended_critical']:
                st.write(f"- Issue ID: {issue['issue_id']} (Created: {issue['created_at']})")

        st.subheader("Recommended Message Template")
        st.text_area("Message", response['message_template'], height=200)

        # Conversation summary
        st.header("Conversation Summary")
        summary = summarize_conversation(response['issue_id'])
        st.text_area("Summary", summary, height=150)

# Run FastAPI in a separate thread
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Start FastAPI server in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    # Run Streamlit app
    main()