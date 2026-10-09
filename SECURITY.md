# Security disclosure and urgent key rotation

The old 'config copy.json' contained static JWT and WebSocket tokens, open network API settings and unsafe defaults. Current examples are sanitized, but the old commits remain readable. **Revoke and rotate all credentials ever committed**; editing files cannot revoke them or erase Git history.

Do not commit API keys, withdrawal-capable exchange credentials, chat tokens or customer information. Deploy only private credentials with least-privilege scopes, restricted access, and third-party execution safeguards. The default sample is stopped, dry-run and has Telegram/API disabled.

For vulnerabilities use GitHub private reporting if available; never disclose usable secrets in a public issue.
