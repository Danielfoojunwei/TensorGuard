"""
TensorGuard Dashboard QA Test Suite

Automated tests for dashboard functionality including:
- Page loading and static file serving
- Navigation between views
- KMS configuration UI
- API endpoint responses
"""

import pytest
import requests
import time
from unittest.mock import patch


DASHBOARD_URL = "http://localhost:8099"


class TestDashboardAvailability:
    """Test suite for dashboard availability and basic loading."""
    
    @pytest.fixture(scope="class")
    def dashboard_response(self):
        """Get dashboard home page response."""
        try:
            return requests.get(f"{DASHBOARD_URL}/", timeout=5)
        except requests.exceptions.ConnectionError:
            pytest.skip("Dashboard server not running")
    
    def test_dashboard_loads(self, dashboard_response):
        """Verify dashboard homepage returns 200."""
        assert dashboard_response.status_code == 200
    
    def test_dashboard_contains_title(self, dashboard_response):
        """Verify dashboard contains TensorGuard title."""
        assert "TensorGuard" in dashboard_response.text
    
    def test_dashboard_contains_version(self, dashboard_response):
        """Verify dashboard shows v2.0.0-FedMoE."""
        assert "v2.0.0" in dashboard_response.text or "FedMoE" in dashboard_response.text
    
    def test_dashboard_contains_navigation(self, dashboard_response):
        """Verify all navigation views are present."""
        html = dashboard_response.text
        assert 'data-view="overview"' in html
        assert 'data-view="settings"' in html
        assert 'data-view="usage"' in html
        assert 'data-view="versions"' in html


class TestDashboardViews:
    """Test suite for dashboard view elements."""
    
    @pytest.fixture(scope="class")
    def html_content(self):
        try:
            response = requests.get(f"{DASHBOARD_URL}/", timeout=5)
            return response.text
        except requests.exceptions.ConnectionError:
            pytest.skip("Dashboard server not running")
    
    def test_overview_view_elements(self, html_content):
        """Verify Overview view contains required elements."""
        assert 'id="view-overview"' in html_content
        assert 'Real-Time Fleet Telemetry' in html_content or 'Fleet Telemetry' in html_content
        assert 'Mixture of Intelligence' in html_content or 'MoI' in html_content
    
    def test_settings_view_elements(self, html_content):
        """Verify Settings view contains required elements."""
        assert 'id="view-settings"' in html_content
        assert 'Privacy Epsilon' in html_content or 'set-epsilon' in html_content
        assert 'LoRA Rank' in html_content or 'set-rank' in html_content
    
    def test_kms_section_elements(self, html_content):
        """Verify KMS/HSM configuration section is present."""
        assert 'kms-provider' in html_content
        assert 'AWS KMS' in html_content
        assert 'Azure Key Vault' in html_content
        assert 'GCP Cloud KMS' in html_content
        assert 'Test Connection' in html_content or 'btn-test-kms' in html_content
    
    def test_security_badge_present(self, html_content):
        """Verify security hardening badge is displayed."""
        assert 'CSPRNG' in html_content or 'msgpack' in html_content or 'security-badge' in html_content


class TestDashboardAPI:
    """Test suite for dashboard API endpoints."""
    
    def test_status_endpoint(self):
        """Verify /api/status returns valid JSON."""
        try:
            response = requests.get(f"{DASHBOARD_URL}/api/status", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert 'connection' in data or 'submissions' in data
        except requests.exceptions.ConnectionError:
            pytest.skip("Dashboard server not running")
    
    def test_start_endpoint(self):
        """Verify /api/start endpoint responds."""
        try:
            response = requests.get(f"{DASHBOARD_URL}/api/start", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("Dashboard server not running")
    
    def test_stop_endpoint(self):
        """Verify /api/stop endpoint responds."""
        try:
            response = requests.get(f"{DASHBOARD_URL}/api/stop", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("Dashboard server not running")


class TestKMSConfiguration:
    """Test suite for KMS/HSM configuration UI elements."""
    
    @pytest.fixture(scope="class")
    def html_content(self):
        try:
            response = requests.get(f"{DASHBOARD_URL}/", timeout=5)
            return response.text
        except requests.exceptions.ConnectionError:
            pytest.skip("Dashboard server not running")
    
    def test_kms_provider_selector(self, html_content):
        """Verify KMS provider dropdown contains all options."""
        assert 'value="local"' in html_content
        assert 'value="aws"' in html_content
        assert 'value="azure"' in html_content
        assert 'value="gcp"' in html_content
    
    def test_aws_config_fields(self, html_content):
        """Verify AWS KMS configuration fields exist."""
        assert 'aws-region' in html_content
        assert 'aws-cmk-arn' in html_content
    
    def test_azure_config_fields(self, html_content):
        """Verify Azure Key Vault configuration fields exist."""
        assert 'azure-vault-url' in html_content
        assert 'azure-key-name' in html_content
    
    def test_gcp_config_fields(self, html_content):
        """Verify GCP Cloud KMS configuration fields exist."""
        assert 'gcp-project' in html_content
        assert 'gcp-location' in html_content
        assert 'gcp-keyring' in html_content
        assert 'gcp-keyname' in html_content
    
    def test_kms_audit_log(self, html_content):
        """Verify KMS audit log section exists."""
        assert 'kms-audit-log' in html_content or 'Key Audit Log' in html_content


class TestStaticAssets:
    """Test suite for static asset loading."""
    
    def test_css_loads(self):
        """Verify CSS stylesheet loads."""
        try:
            response = requests.get(f"{DASHBOARD_URL}/styles.css", timeout=5)
            assert response.status_code == 200
            assert 'text/css' in response.headers.get('Content-Type', '')
        except requests.exceptions.ConnectionError:
            pytest.skip("Dashboard server not running")
    
    def test_js_loads(self):
        """Verify JavaScript file loads."""
        try:
            response = requests.get(f"{DASHBOARD_URL}/app.js", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("Dashboard server not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
