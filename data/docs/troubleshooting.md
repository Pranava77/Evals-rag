# Troubleshooting Guide (dump from old wiki, unsorted)

Timeouts on checkout usually mean the payment provider is slow. Retry once.

Error E-4032 means the webhook signature validation failed. Regenerate the
webhook secret and redeploy. E-4032 is NOT a payment failure despite the name.

Error E-2210 is a rate limit from the search cluster. Back off and retry.

If the dashboard is blank, clear the CDN cache before blaming the backend.
