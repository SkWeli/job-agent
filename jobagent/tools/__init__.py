"""The agent's toolbox. Each tool does one thing, validates its input with a Pydantic schema, and
returns a short, model-readable string. Expected failures raise ToolError with a hint; the graph turns
them into 'ERROR: ...' messages so the model can recover (see graph.tools_node).

| tool                    | what it does                                  | side effects        |
|-------------------------|-----------------------------------------------|---------------------|
| search_jobs             | query one live feed, save new jobs            | writes tracker      |
| get_job_details         | full posting for one job                      | none                |
| save_assessment         | record score 1-5, verdict, reasons, gaps      | writes tracker      |
| list_jobs               | query the tracker                             | none                |
| draft_application       | tailored CV + letter, fact-checked (subgraph) | writes files        |
| read_web_page           | read one URL (robots.txt respected)           | none                |
| add_job_from_page       | save a job you just read                      | writes tracker      |
| add_job_from_text       | save a job ad the human pasted                | writes tracker      |
| write_note              | lesson for future runs                        | writes notes.md     |
| send_application_email  | email an application. NEEDS HUMAN APPROVAL    | IRREVERSIBLE        |
"""
from __future__ import annotations

from . import apply, search, tracker
from .context import ToolContext, ToolError


def build_tools(ctx: ToolContext):
    return [*search.build(ctx), *tracker.build(ctx), *apply.build(ctx)]


__all__ = ["build_tools", "ToolContext", "ToolError"]
