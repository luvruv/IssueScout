"""Unit tests for responsiveness analysis, claim detection, and setup checklist extraction."""
from core.claim_detector import detect_issue_claim_status
from core.responsiveness import calculate_maintainer_responsiveness
from core.setup_checklist import extract_setup_checklist


def test_claim_detector_closed_and_assigned():
    # 1. Closed issue
    mock_closed = {"state": "closed", "assignees": []}
    res_closed = detect_issue_claim_status("owner/repo", 1, mock_closed)
    assert res_closed["status"] == "CLOSED"
    assert res_closed["is_available"] is False

    # 2. Assigned issue
    mock_assigned = {"state": "open", "assignees": [{"login": "alice"}]}
    res_assigned = detect_issue_claim_status("owner/repo", 2, mock_assigned)
    assert res_assigned["status"] == "ASSIGNED"
    assert res_assigned["is_available"] is False
    assert "alice" in res_assigned["assignees"]


def test_claim_detector_discussion_and_intents():
    # Open issue with zero comments is available
    mock_fresh = {"state": "open", "assignees": [], "comments": 0}
    res_fresh = detect_issue_claim_status("owner/repo", 3, mock_fresh)
    assert res_fresh["status"] == "AVAILABLE"
    assert res_fresh["is_available"] is True


def test_responsiveness_offline_fallback():
    # Responsiveness should return valid structure even when network is offline
    res = calculate_maintainer_responsiveness("pallets/flask", sample_size=2)
    assert "tier" in res
    assert "badge" in res
    assert "is_responsive" in res


def test_setup_checklist_extraction():
    # Setup checklist should return valid terminal commands
    s = extract_setup_checklist("pallets/flask")
    assert "git clone" in s["clone_cmd"]
    assert len(s["install_cmds"]) > 0
    assert len(s["test_cmds"]) > 0
    assert len(s["lint_cmds"]) > 0

