# Security Runbook

API keys rotate every 90 days. To rotate an API key: open Admin > Keys, click
Rotate, then update the CI secret within 24 hours or deploys will fail.

Passwords must be 12+ chars. SSO is mandatory for staff. Never paste keys into
tickets — redact them (see incident INC-2291 for why).
