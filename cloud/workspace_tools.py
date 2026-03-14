"""
Rio Cloud — Google Workspace Integration Tools

Provides tool functions for Gmail, Google Drive, and Google Calendar
that the ToolOrchestrator can invoke via function calling.

Requires:
  - google-api-python-client
  - google-auth
  - Service account credentials or OAuth token

Set the following environment variables:
  - GOOGLE_APPLICATION_CREDENTIALS: path to service account JSON
  - GOOGLE_WORKSPACE_USER: email to impersonate (for domain-wide delegation)

If dependencies are missing, tools degrade gracefully (return error messages).
"""

from __future__ import annotations

import json
import os
import structlog
from typing import Any

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Google API client setup (lazy init)
# ---------------------------------------------------------------------------

_gmail_service = None
_drive_service = None
_calendar_service = None
_sheets_service = None
_docs_service = None
_available = False


def _get_credentials():
    """Get Google API credentials from service account or application default."""
    try:
        from google.oauth2 import service_account
        from google.auth import default

        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        user_email = os.environ.get("GOOGLE_WORKSPACE_USER", "")

        SCOPES = [
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/documents",
        ]

        if creds_path and os.path.isfile(creds_path):
            creds = service_account.Credentials.from_service_account_file(
                creds_path, scopes=SCOPES,
            )
            if user_email:
                creds = creds.with_subject(user_email)
            return creds
        else:
            creds, _ = default(scopes=SCOPES)
            return creds
    except Exception as exc:
        logger.warning("workspace.credentials_failed", error=str(exc))
        return None


def _init_services():
    """Initialize Google API services (lazy, called on first use)."""
    global _gmail_service, _drive_service, _calendar_service
    global _sheets_service, _docs_service, _available

    if _available:
        return True

    try:
        from googleapiclient.discovery import build
        creds = _get_credentials()
        if creds is None:
            return False

        _gmail_service = build("gmail", "v1", credentials=creds)
        _drive_service = build("drive", "v3", credentials=creds)
        _calendar_service = build("calendar", "v3", credentials=creds)

        try:
            _sheets_service = build("sheets", "v4", credentials=creds)
        except Exception:
            pass

        try:
            _docs_service = build("docs", "v1", credentials=creds)
        except Exception:
            pass

        _available = True
        logger.info("workspace.initialized")
        return True
    except ImportError:
        logger.info("workspace.not_available",
                     reason="install google-api-python-client google-auth")
        return False
    except Exception as exc:
        logger.warning("workspace.init_failed", error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Gmail Tools
# ---------------------------------------------------------------------------

async def gmail_search(query: str = "", max_results: int = 10) -> dict:
    """Search Gmail messages.

    Args:
        query: Gmail search query (same syntax as Gmail search bar).
               Examples: "from:boss@company.com", "subject:meeting is:unread"
        max_results: Maximum number of messages to return (default 10).
    """
    if not _init_services() or _gmail_service is None:
        return {"success": False, "error": "Gmail not configured. Set GOOGLE_APPLICATION_CREDENTIALS."}

    try:
        result = _gmail_service.users().messages().list(
            userId="me", q=query, maxResults=max_results,
        ).execute()
        messages = result.get("messages", [])

        # Fetch snippets for each message
        output = []
        for msg in messages[:max_results]:
            detail = _gmail_service.users().messages().get(
                userId="me", id=msg["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()
            headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
            output.append({
                "id": msg["id"],
                "from": headers.get("From", ""),
                "subject": headers.get("Subject", ""),
                "date": headers.get("Date", ""),
                "snippet": detail.get("snippet", ""),
            })

        return {"success": True, "count": len(output), "messages": output}
    except Exception as exc:
        return {"success": False, "error": f"Gmail search failed: {exc}"}


async def gmail_send(
    to: str = "",
    subject: str = "",
    body: str = "",
) -> dict:
    """Send an email via Gmail.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text (plain text).
    """
    if not _init_services() or _gmail_service is None:
        return {"success": False, "error": "Gmail not configured."}
    if not to:
        return {"success": False, "error": "Recipient 'to' is required."}

    try:
        import base64
        from email.mime.text import MIMEText

        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        result = _gmail_service.users().messages().send(
            userId="me", body={"raw": raw},
        ).execute()

        return {"success": True, "message_id": result.get("id", "")}
    except Exception as exc:
        return {"success": False, "error": f"Gmail send failed: {exc}"}


# ---------------------------------------------------------------------------
# Google Drive Tools
# ---------------------------------------------------------------------------

async def drive_list(query: str = "", max_results: int = 20) -> dict:
    """List/search files in Google Drive.

    Args:
        query: Drive search query (same syntax as Drive search).
               Examples: "name contains 'report'", "mimeType='application/pdf'"
        max_results: Maximum number of files to return (default 20).
    """
    if not _init_services() or _drive_service is None:
        return {"success": False, "error": "Google Drive not configured."}

    try:
        kwargs = {
            "pageSize": max_results,
            "fields": "files(id, name, mimeType, size, modifiedTime, webViewLink)",
        }
        if query:
            kwargs["q"] = query

        result = _drive_service.files().list(**kwargs).execute()
        files = result.get("files", [])

        return {
            "success": True,
            "count": len(files),
            "files": [
                {
                    "id": f["id"],
                    "name": f.get("name", ""),
                    "type": f.get("mimeType", ""),
                    "size": f.get("size", ""),
                    "modified": f.get("modifiedTime", ""),
                    "url": f.get("webViewLink", ""),
                }
                for f in files
            ],
        }
    except Exception as exc:
        return {"success": False, "error": f"Drive list failed: {exc}"}


async def drive_download(file_id: str = "", save_path: str = "") -> dict:
    """Download a file from Google Drive.

    Args:
        file_id: The Google Drive file ID.
        save_path: Local path to save the file (default: /tmp/<filename>).
    """
    if not _init_services() or _drive_service is None:
        return {"success": False, "error": "Google Drive not configured."}
    if not file_id:
        return {"success": False, "error": "file_id is required."}

    try:
        # Get file metadata
        meta = _drive_service.files().get(fileId=file_id, fields="name,mimeType").execute()
        name = meta.get("name", file_id)
        if not save_path:
            save_path = os.path.join(os.environ.get("TEMP", "/tmp"), name)

        # Download content
        import io
        from googleapiclient.http import MediaIoBaseDownload

        request = _drive_service.files().get_media(fileId=file_id)
        fh = io.FileIO(save_path, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        return {"success": True, "path": save_path, "name": name}
    except Exception as exc:
        return {"success": False, "error": f"Drive download failed: {exc}"}


# ---------------------------------------------------------------------------
# Google Calendar Tools
# ---------------------------------------------------------------------------

async def calendar_list_events(
    date_from: str = "",
    date_to: str = "",
    max_results: int = 10,
) -> dict:
    """List upcoming calendar events.

    Args:
        date_from: Start date in ISO format (default: now).
        date_to: End date in ISO format (default: 7 days from now).
        max_results: Maximum events to return (default 10).
    """
    if not _init_services() or _calendar_service is None:
        return {"success": False, "error": "Google Calendar not configured."}

    try:
        from datetime import datetime, timedelta, timezone

        if not date_from:
            date_from = datetime.now(timezone.utc).isoformat()
        if not date_to:
            date_to = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

        result = _calendar_service.events().list(
            calendarId="primary",
            timeMin=date_from,
            timeMax=date_to,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        events = result.get("items", [])
        return {
            "success": True,
            "count": len(events),
            "events": [
                {
                    "id": e["id"],
                    "summary": e.get("summary", ""),
                    "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date", "")),
                    "end": e.get("end", {}).get("dateTime", e.get("end", {}).get("date", "")),
                    "location": e.get("location", ""),
                    "status": e.get("status", ""),
                }
                for e in events
            ],
        }
    except Exception as exc:
        return {"success": False, "error": f"Calendar list failed: {exc}"}


async def calendar_create(
    summary: str = "",
    start: str = "",
    end: str = "",
    description: str = "",
    location: str = "",
) -> dict:
    """Create a new calendar event.

    Args:
        summary: Event title.
        start: Start time in ISO format (e.g., "2026-03-15T10:00:00+05:30").
        end: End time in ISO format.
        description: Event description (optional).
        location: Event location (optional).
    """
    if not _init_services() or _calendar_service is None:
        return {"success": False, "error": "Google Calendar not configured."}
    if not summary or not start or not end:
        return {"success": False, "error": "summary, start, and end are required."}

    try:
        event_body = {
            "summary": summary,
            "start": {"dateTime": start},
            "end": {"dateTime": end},
        }
        if description:
            event_body["description"] = description
        if location:
            event_body["location"] = location

        result = _calendar_service.events().insert(
            calendarId="primary", body=event_body,
        ).execute()

        return {
            "success": True,
            "event_id": result.get("id", ""),
            "link": result.get("htmlLink", ""),
        }
    except Exception as exc:
        return {"success": False, "error": f"Calendar create failed: {exc}"}


# ---------------------------------------------------------------------------
# Google Sheets Tools
# ---------------------------------------------------------------------------

async def sheets_read(
    spreadsheet_id: str = "",
    range_name: str = "Sheet1!A1:Z100",
) -> dict:
    """Read data from a Google Sheet.

    Args:
        spreadsheet_id: The spreadsheet ID (from the URL).
        range_name: A1 notation range (default "Sheet1!A1:Z100").
    """
    if not _init_services() or _sheets_service is None:
        return {"success": False, "error": "Google Sheets not configured."}
    if not spreadsheet_id:
        return {"success": False, "error": "spreadsheet_id is required."}

    try:
        result = _sheets_service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name,
        ).execute()

        values = result.get("values", [])
        return {
            "success": True,
            "range": result.get("range", range_name),
            "rows": len(values),
            "data": values[:100],  # Cap at 100 rows
        }
    except Exception as exc:
        return {"success": False, "error": f"Sheets read failed: {exc}"}


# ---------------------------------------------------------------------------
# Google Docs Tools
# ---------------------------------------------------------------------------

async def docs_create(
    title: str = "",
    body_text: str = "",
) -> dict:
    """Create a new Google Doc.

    Args:
        title: Document title.
        body_text: Initial text content for the document.
    """
    if not _init_services() or _docs_service is None:
        return {"success": False, "error": "Google Docs not configured."}
    if not title:
        return {"success": False, "error": "title is required."}

    try:
        doc = _docs_service.documents().create(
            body={"title": title},
        ).execute()

        doc_id = doc.get("documentId", "")

        # Insert body text if provided
        if body_text and doc_id:
            _docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={
                    "requests": [{
                        "insertText": {
                            "location": {"index": 1},
                            "text": body_text,
                        }
                    }]
                },
            ).execute()

        return {
            "success": True,
            "document_id": doc_id,
            "title": title,
            "url": f"https://docs.google.com/document/d/{doc_id}/edit",
        }
    except Exception as exc:
        return {"success": False, "error": f"Docs create failed: {exc}"}


# ---------------------------------------------------------------------------
# Workspace tool list — for orchestrator registration
# ---------------------------------------------------------------------------

WORKSPACE_TOOLS = [
    gmail_search,
    gmail_send,
    drive_list,
    drive_download,
    calendar_list_events,
    calendar_create,
    sheets_read,
    docs_create,
]


def get_workspace_tools() -> list:
    """Return workspace tools if Google API client is available."""
    try:
        from googleapiclient.discovery import build  # noqa: F401
        logger.info("workspace_tools.available")
        return WORKSPACE_TOOLS
    except ImportError:
        logger.info("workspace_tools.unavailable",
                     hint="pip install google-api-python-client google-auth")
        return []
