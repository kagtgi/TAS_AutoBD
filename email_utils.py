"""
TAS AutoBD — Email Utilities
==============================
Handles outbound email delivery via SendGrid and the Streamlit
widget for adding email addresses manually.

NOTE: This file is named *email_utils.py* (not email.py) to avoid
shadowing Python's built-in `email` standard library module.
"""

import re
import logging
import streamlit as st

from config import SENDGRID_API_KEY, SENDER_EMAIL

logger = logging.getLogger(__name__)

# Basic RFC-5322-compatible email pattern
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def is_valid_email(address: str) -> bool:
    """Return True if *address* looks like a valid email address."""
    return bool(_EMAIL_RE.fullmatch(address.strip()))


def send_email(to_email: str, subject: str, html_content: str):
    """
    Send an HTML email via SendGrid.

    Parameters
    ----------
    to_email     : recipient address
    subject      : email subject line
    html_content : full HTML body

    Returns
    -------
    (True, status_message)   on success
    (False, error_message)   on failure
    """
    if not SENDGRID_API_KEY:
        return False, (
            "SENDGRID_API_KEY is not configured. "
            "Add it to your .env file and restart the app."
        )

    if not is_valid_email(to_email):
        return False, f"Invalid recipient address: {to_email!r}"

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        message = Mail(
            from_email=SENDER_EMAIL,
            to_emails=to_email,
            subject=subject,
            html_content=html_content,
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        logger.info("Email sent to %s — status %s", to_email, response.status_code)
        return True, f"Email sent successfully (status {response.status_code})."
    except Exception as exc:  # noqa: BLE001
        logger.error("SendGrid error: %s", exc)
        return False, str(exc)


def add_email_manually(email_list: list) -> list:
    """
    Render a Streamlit widget that lets the user append an email address
    to *email_list*. Returns the (possibly updated) list.
    """
    col1, col2 = st.columns([4, 1])
    with col1:
        new_email = st.text_input("Add recipient email manually:", key="manual_email_input")
    with col2:
        st.write("")  # vertical alignment spacer
        st.write("")
        add_clicked = st.button("Add", key="add_email_btn")

    if add_clicked:
        if not new_email:
            st.warning("Please enter an email address.")
        elif not is_valid_email(new_email):
            st.warning(f"'{new_email}' does not look like a valid email address.")
        elif new_email in email_list:
            st.info("That email address is already in the list.")
        else:
            email_list = email_list + [new_email]
            st.success(f"Added: {new_email}")

    return email_list
