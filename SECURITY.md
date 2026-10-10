# Security disclosure and urgent key rotation

The old 'config copy.json' contained static JWT and WebSocket tokens, open network API settings and unsafe defaults. Current examples are sanitized, but the old commits remain readable. **Revoke and rotate all credentials ever committed**; editing files cannot revoke them or erase Git history.

Do not commit API keys, withdrawal-capable exchange credentials, chat tokens or customer information. Deploy only private credentials with least-privilege scopes, restricted access, and third-party execution safeguards. The default sample is stopped, dry-run and has Telegram/API disabled.

For vulnerabilities use GitHub private reporting if available; never disclose usable secrets in a public issue.


## MLSD model artifacts (v0.4.1)

The v2 model loader rejects duplicates in JSON keys, malformed coefficients, missing provenance, invalid trading-pair identity and inconsistent training cutoffs. A SHA256 fingerprint of training **inputs** is not a signature: hostile or compromised model files can still contain attacker-chosen yet valid coefficients. Mount them from a trusted, read-only source, control file permissions and never enable trading solely because a JSON artifact loaded successfully. The v0.4.0 model JSON schema is deliberately not accepted by the v0.4.1 adapter: re-run offline fitting for the strict benchmark-lag contract.
